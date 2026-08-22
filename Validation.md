# nsdi-agent 正式实验验收规范

本文定义活动数据进入完整 RCA 框架和正式实验前必须满足的验收门禁。每个门禁包含固定策略、验证证据和通过条件，确保数据、语义、拓扑、训练知识和实验输出可以独立审计。

状态定义：

- `PASS`：已有可复核证据，允许下游使用。
- `READY`：实现和本地结构检查完成，等待正式 GPU 运行产生实验证据。
- `IMPLEMENT`：规范已确定，工程实现尚未完成。
- `BLOCK`：正式实验不得绕过该项。
- `MONITOR`：不阻塞实现，但必须在报告中持续审计。

## 1. 总览

| 编号 | 门禁 | 状态 | 阻塞范围 |
| --- | --- | --- | --- |
| V1 | 活动数据完整性 | PASS | 数据准备 |
| V2 | 时间切分与测试隔离 | PASS | 训练与评估 |
| V3 | 统一标签与端点语义 | PASS | 全框架 |
| V4 | Expert label 应用 | PASS | 训练标签与测试标签 |
| V5 | Manifest adapter | PASS | 框架数据入口 |
| V6 | 电压与数值边界语义 | IMPLEMENT | 物理约束 |
| V7 | 跨端 lane 对应关系 | PASS | lane_direction 特征 |
| V8 | SerDes SNR 量测语义 | IMPLEMENT | 特征解释与阈值 |
| V9 | host_snr 缺测语义 | PASS | 缺测路由 |
| V10 | 来源拓扑与 schema 漂移 | PASS | 特征与检索 |
| V11 | 训练知识重建 | READY | N3-N6 |
| V12 | Topology-aware 检索 | PASS | N3-N5 |
| V13 | Prompt 与输出协议 | PASS | LLM 正式运行 |
| V14 | 路由与置信度标定 | READY | N4/N6 |
| V15 | Fiber 证据与指标 | IMPLEMENT | 结论与报告 |
| V16 | 测试只读与防泄漏 | PASS | 所有实验 |
| V17 | 报告与 bad case 审计 | READY | 实验交付 |
| V18 | 可复现性与资源释放 | READY | 正式运行 |
| V19 | Legacy 回归隔离 | PASS | 合并代码 |

## 2. 验收门禁

### V1 活动数据完整性

活动数据固定为 `datasets/filtered_rule_temporal_2025_06_09_v1/`，共 608 条：

- `all_data`: 505
- `rule1_channel_not_4`: 103

验证证据：

- Manifest 记录每条源文件和输出文件的 SHA-256。
- `_metadata/quality_report.json` 记录来源数量、日期解析和 lane 宽度摘要。
- `python3 scripts/prepare_filtered_rule_temporal_split.py --check` 返回 `ok=true`、`case_count=608`、`errors=[]`。

通过条件：源文件存在且哈希一致；输出文件数量、case ID、标签和哈希与 manifest 一致。

固定策略：目录截图和外部汇总不覆盖逐文件 manifest；数据变更通过新数据集版本交付，不原地改写 v1。

### V2 时间切分与测试隔离

训练集使用 2025-06、2025-07、2025-08、2025-09，共 124 条。其余月份分别形成：

- `test/all_data`: 417
- `test/rule1_channel_not_4`: 67

通过条件：

- train 中不存在训练月份以外的 case。
- 两个 test 中不存在训练月份 case。
- case ID 在三个 split 间无重复。
- 两个测试集分别评估、分别报告。

固定策略：训练阶段可以合并两个来源；测试阶段不生成掩盖来源差异的单一主指标。需要总体数字时只作为按样本数加权的附加统计，并同时展示两个来源结果。

### V3 统一标签与端点语义

正式标签语义：

- `L1`：本端。
- `L2`：对端。
- `fiber`：链路介质。

固定别名：

- `all_data`: `l1/l2 -> L1/L2`
- `rule1_channel_not_4`: `l3/l4 -> L1/L2`

通过证据：

- 608 条输出标签均属于 `L1/L2/fiber`。
- 遥测端点键中不存在活动的 `l1/l2/l3/l4` 别名。
- `EvidencePack` 对 608 条 case 的 `no_telemetry` 数为 0。

固定策略：L1/L2 不绑定速率、设备型号和 lane 数。目录名、告警所在端和缺测字段不参与标签推断。

### V4 Expert label 应用

Expert label 只通过核心遥测精确指纹匹配：

- 审核 case 命中 49 条。
- 实际标签修正 27 条。
- 未精确命中 17 条。

通过条件：

- 一个遥测指纹最多对应一个显式 expert label。
- stable case ID 不因标签修正而变化。
- 所有修改保留原标签、审核 case ID 和审核元数据。
- 未精确命中的 case 不做推断式改标。

固定策略：Expert label 是版本化人工输入，不通过近邻、文件夹或多数投票扩散。

### V5 Manifest adapter

活动 manifest 使用 `output_file`，并包含 `train`、`test/all_data`、`test/rule1_channel_not_4` 三个 split。adapter 同时兼容活动字段 `output_file` 与 legacy 字段 `file`。

通过条件：

- 新 adapter 返回 124/417/67 三个稳定集合。
- 每条 case 保留 `source_dataset` 与 `_dataset_contract`。
- 推理入口不暴露测试标签。
- loader 单测锁定数量、标签分布、split 隔离和源文件哈希。
- legacy loader 行为不变。

验证证据：adapter 已锁定 124/417/67 数量、split 隔离和逐 case 数据契约；直接断言通过。

### V6 电压与数值边界语义

连续量必须使用字段级单位和边界，不建立全局 `value == 0` 异常规则。

固定规则：

- `txpower/rxpower = 0 dBm` 表示有效光功率读数，不自动判坏。
- `txpower/rxpower <= -39 dBm` 可作为断光或采集哨兵候选，需结合状态位区分。
- `bias = 0 mA` 表示激光器未驱动候选。
- `media_snr/host_snr <= 0` 作为触底候选。
- `serdes_snr <= 1` 作为失效状态候选，不解释为已确认的 dB 工程阈值。
- 3.10 V 单例作为量测异常和人工复核触发项，不单独生成根因标签。

通过条件：特征字典记录单位、缺测状态、抽取规则和边界来源；确定性物理边界与训练统计阈值分层保存。

### V7 跨端 lane 对应关系

数据中的 `transmission` 字段逐 lane 表示远端 Rx 与本端 Tx 的同编号差值；非哨兵值与该定义一致，因此光学同编号 lane 具备数据契约层面的逻辑映射。来源间仍存在 lane 宽度差异和部分指标缺失。

固定策略：

- 光学同编号 lane pairing 启用，只产生掉光状态、影响范围和 case 内相对离群证据。
- 原始 Tx/Rx 值不用于计算绝对链路损耗。
- SerDes lane 不映射到光学 lane。
- lane 宽度冲突显式编码，不截断后伪装成完整映射。

通过证据：`filtered-rule-topology-v1`、活动特征字典和单测共同锁定上述语义；两个测试集分别有 157 条和 15 条产生逻辑同 lane token。

### V8 SerDes SNR 量测语义

`serdes_snr` 当前按设备导出的数值质量指标处理，不声明未经数据契约支持的物理单位。

固定策略：

- 特征名称和报告使用“SerDes SNR 指标值”。
- 工程阈值与训练分位数分开记录。
- 缺失、零值和低值分开编码。
- 不与 `media_snr`、`host_snr` 共享阈值。

通过条件：字段字典明确单位状态、值域、异常哨兵和来源覆盖率。

### V9 host_snr 缺测语义

活动数据存在系统性 `host_snr` 缺测，`test/rule1_channel_not_4` 为 0/67 可观测。该现象属于 schema 差异，不代表链路正常或异常。

固定策略：

- 缺失编码为 `missing` 或 `not_applicable`。
- 缺失本身不生成根因支持票。
- 依赖 `host_snr` 的规则在字段缺失时标记为不可评估。
- 报告按来源展示字段覆盖率。

通过证据：规范化后 67 条该来源 test 均为 `partial_telemetry`，没有被误判为 `no_telemetry`。

### V10 来源拓扑与 schema 漂移

统一标签允许共享训练池，但不同来源的 lane 数、字段存在率和状态值域必须保留。

通过条件：生成活动数据 topology profile，至少包括：

- 每个来源和 split 的指标存在率。
- 每侧、每指标的 lane 宽度分布。
- `Lane number` 字段可靠性。
- 光功率哨兵、SNR 触底和状态位值域。
- train/test schema 差异。

固定策略：拓扑信息用于兼容性判断和分层审计，不直接作为根因标签特征。

验证证据：每条 EvidencePack 和 GraphCase 已保存 source、topology ID、lane profile 与实测宽度；manifest 固定 4×4 400G/200G 和 8×8 400G/400G 两类拓扑。

### V11 训练知识重建

活动训练集必须独立生成：

- 特征统计与异常阈值。
- IDF 和证据图。
- signature 纯度。
- learned SOP。
- measured constraint 支持数。
- 路由和置信度校准数据。

通过条件：每个 artifact 记录训练 case ID 集合、数据 manifest hash、版本号和内容 hash；生成过程不读取两个测试集标签。

固定策略：organized、l2fixed 和 expanded 数据产生的训练知识只作为参考，不进入活动数据正式推理。

### V12 Topology-aware 检索

统一训练池需要同时支持来源内历史匹配和跨来源通用物理关系共享。

固定策略：

- 历史候选保留 `source_dataset`、lane profile 和字段覆盖。
- N5a 完全匹配要求签名纯净且拓扑兼容。
- 跨来源候选默认不能仅凭 `sim=1.0` 直接复用历史结论。
- 跨来源共享优先限于不依赖 lane 数和端点编号的物理 token。

通过条件：报告两个测试集的跨来源 Top-N 数量、跨来源完全匹配数量、候选标签纯度和拓扑 veto 数量，并提供来源内检索消融。

验证证据：`build_packs` 优先读取逐 case 契约；`GraphCase` 与候选保存来源和拓扑。核心检索中 `all_data` 仅 2/417 使用显式跨拓扑兜底，`rule1` 为 0/67。

### V13 Prompt 与输出协议

活动 prompt 必须使用本端/对端语义：

- `L1 = local endpoint`
- `L2 = remote endpoint`
- `fiber = link medium`

通过条件：

- Prompt 不出现 L1=400G、L2=200G 的固定定义。
- 输入包含来源、lane profile、缺测字段、历史候选和物理约束。
- 输出固定为结构化候选、置信度、证据引用、冲突、缺失证据和建议动作。
- 输出 parser 对三个标签和降级状态有完整测试。
- Prompt 内容和变量顺序通过 hash 版本化。

验证证据：活动 Prompt 版本为 `filtered-rule-local-remote-v1`，hash 同时覆盖物理约束库、量测契约库和固定模板；拓扑上下文进入每个推理请求。

### V14 路由与置信度标定

N4/N6 必须在活动训练集内部标定，测试标签只用于最终评估。

通过条件：

- N5a 报告完全匹配数量、signature 纯度和混合标签覆盖。
- N5b 报告样本支持数和关键缺失证据。
- N5c 报告低匹配与证据不足原因。
- 每个自动结论阈值同时记录 coverage、precision at coverage 和置信区间。
- 小支持桶自动降级，不以单次高准确率进入正式策略。

固定策略：路由阈值、Wilson 下界和最小支持数均作为版本化配置，不继承旧数据默认值。

### V15 Fiber 证据与指标

Fiber 是少数类，活动训练集 11 条，两个测试集分别为 15 条和 1 条。

固定策略：

- 不用 L1/L2 证据不足自动推出 fiber。
- Fiber 结论需要介质侧直接证据、双向一致证据或经过审核的历史模式。
- 缺少 OTDR、镜检或可信同步功率标定时，允许输出补采或人工复核。
- 不承诺通过 prompt 单独解决 fiber。

通过条件：分别报告 fiber precision、recall、F1、预测数量、自动结案数量和降级数量；`rule1` test 只有 1 条 fiber，报告采用逐 case 解释，不下稳定统计结论。

### V16 测试只读与防泄漏

固定策略：

- 所有特征阈值、图、SOP、约束统计、prompt 示例和置信度策略只由 train 产生。
- 测试标签仅在预测落盘后进入 evaluator。
- 测试 bad case 不修改同一实验的知识包。
- N8 自动回灌关闭。

通过条件：`run_manifest.json` 记录训练 case ID 集合和 N8 状态；报告明确 `label_leakage=false`；测试运行从持久化只读知识包重新加载。

### V17 报告与 bad case 审计

每个测试集独立生成总览和逐 case 页面。

通过条件：

- 首页展示数据版本、流程图、模型版本和核心配置。
- 正确 case 按分支和关键证据归纳。
- Bad case 逐条标记失败步骤与原因。
- 原因分类至少覆盖：数据不可辨识、疑似标签、缺测、拓扑不兼容、图缺边、约束错误、阈值错误、prompt 错误和代码错误。
- `label_suspects.json` 与 `irreducible_cases.json` 独立保存。
- 两个测试集不共用一个模糊的 bad-case 汇总。

### V18 可复现性与资源释放

正式实验必须记录：

- Git commit 与工作树状态。
- 数据 manifest hash。
- 所有知识资产版本和 hash。
- 随机种子。
- LLM checkpoint、dtype、tensor parallel、max model length 和生成参数。
- 运行环境与依赖版本。
- GPU 运行前后快照。

通过条件：同一 manifest 和配置可以重放；vLLM 在 `finally` 中关闭；进程外复核显存释放；中断运行保留可审计日志，不把半成品标记为完整实验。

### V19 Legacy 回归隔离

旧数据和旧入口保留为工程回归资产。

通过条件：

- 活动 adapter 不改变 legacy loader 默认行为。
- 影响 legacy 路径的代码改动通过 `python -m pytest -q`。
- `tests/test_baseline_lock.py` 保持既有逐 case 锁。
- 活动数据指标和旧 organized/l2fixed 指标分表展示。

固定策略：legacy Prompt、活动 Prompt 和证据图内容指纹按数据契约分别版本化；活动
拓扑字段不得改变 legacy v1 图的冻结指纹。

验证证据：本机项目虚拟环境使用 pytest 9.1.1 完成全量回归，结果为
`347 passed in 15.72s`；`tests/test_baseline_lock.py`、legacy 图 hash
`5e10b5b25d559777`、legacy Prompt v14 以及活动 local/remote Prompt 隔离测试全部通过。

## 3. 正式实验执行与结果验收

活动实现已经具备正式 GPU 执行入口。运行时按以下顺序完成门禁：

1. 同步脚本确认工作树干净，切换并拉取 `origin/main`。
2. 实验机补跑完整 pytest；legacy 回归失败时不发布结果。
3. GPU 包装器验证数据 manifest、模型目录和空闲 GPU，直接启动 vLLM，不执行 CPU 模型 dry run。
4. 从 124 条 train 构建并持久化知识包，重新加载后分别运行 417 条和 67 条测试。
5. 两个测试集各自生成指标、traces 和逐 case HTML；测试标签不进入推理输入。
6. 进程内关闭 vLLM，并使用进程外 `nvidia-smi` 复核资源释放。
7. V11、V14、V17、V18 只有在正式 artifact 完整且可审计后从 READY 更新为 PASS。

正式结果的发布条件是 V1-V19 全部达到 PASS 或明确的非阻塞 MONITOR 状态。
