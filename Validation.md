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
| V11 | 训练知识重建 | PASS | N3-N6 |
| V12 | Topology-aware 检索 | PASS | N3-N5 |
| V13 | Prompt 与输出协议 | PASS | LLM 正式运行 |
| V14 | 路由与置信度标定 | READY | N4/N6 |
| V15 | Fiber 证据与指标 | IMPLEMENT | 结论与报告 |
| V16 | 测试只读与防泄漏 | PASS | 所有实验 |
| V17 | 报告与 bad case 审计 | READY | 实验交付 |
| V18 | 可复现性与资源释放 | READY | 正式运行 |
| V19 | Legacy 回归隔离 | PASS | 合并代码 |
| V20 | 高相似异标签人工复核 | READY | 标签质量审计 |
| V21 | 可解释特征置换不变性与复杂度 | PASS | N2 特征审计 |
| V22 | 证据图结构与诊断链有效性 | READY | N3/N5历史知识 |
| V23 | 训练增强排障决策图谱 | READY | N3-N6决策知识 |
| V24 | 决策图谱测试期逐case复盘 | PASS | 测试审计 |

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

验证证据：`scripts/build_filtered_rule_deterministic_knowledge.py` 已从 124 条 train 在约
5 秒内生成 `artifacts/filtered_rule_deterministic_knowledge_v1/`，训练 LLM 调用和 trace
均为 0。知识包 hash 为 `23a39fe3ced1910e`，证据图版本为
`evidence-graph-v1:124:affc399cf8706073`，逐 case 审计覆盖 124/124。118 个 signature 中
112 个为单例；数值 learned SOP 训练内命中 79/124、fiber 0/11，故二者均保留支持数与
不确定性，不作为物理真值。

测试分布预审计同样不调用 LLM：`all_data` 和 `rule1_channel_not_4` 的双相似度精确匹配
分别只有 12/417 和 1/67，最近历史标签准确率分别为 52.76% 和 49.25%。正式入口必须在
vLLM 初始化前完成确定性知识落盘、重新加载，并与仓库参考 hash 比对；不允许因相似度非零
就把历史标签当作测试结论。

追加盲测证据：冷启动物理推理在两个 split 为 66.19%/64.18%，加入当前训练证据图、
learned SOP 与训练可靠性后降为 58.75%/50.75%。因此当前知识融合不能作为准确率提升门禁，
正式优化必须回到 train LOO 或新增独立 validation split；禁止用本轮测试结果调整权重后
再次把同一测试集报告成盲测。两条精确纯净历史冲突仅列为标签疑点，不自动改标。

追加专家文档复判证据：当前模型逐 case 冷启动预测先冻结，再固定应用
`expert-model-document-v1`，专家规则执行阶段只读取去标签遥测。两个 split 从
41.97%/46.27% 提升至 53.96%/64.18%，但规则输出存在 L1=291/484 的结构性偏置，来源包括
144 条无异常默认 L1 和双端均 down 默认 L1。该结果验证专家规则可作为训练外对照与候选知识，
但不能直接成为正式自动终裁器；后续改进必须在 train LOO 或独立 validation 上完成，不能根据
本轮测试正误修改阈值、优先级或双端裁决后再次报告为同轮盲测。

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

验证证据：活动 Prompt 版本为 `filtered-rule-general-structured-retry-v4`，hash 同时覆盖物理约束库、量测契约库和固定模板；拓扑上下文、双相似度、当前物理路径、历史证据链和差异清单进入对应分支请求。N5a/N5b 只注入相关物理约束与量测 veto，N5c 注入完整专家 SOP。统一模板不要求固定步骤数，SOP 与谓词引用字段均为可选。

正式生成契约：`max_new_tokens=16384`、`max_model_len=32768`、`max_attempts=3`。
正式 vLLM 固定启用 JSON Schema guided decoding，并优先使用模型 tokenizer 的原生 chat template。
每条 case 只产生一个 trace；首轮后仅重试未通过 parser/checker 的 case，已通过 case 不再请求模型。
`attempt_count` 必须为 1–3；三轮后仍失败的输出进入低置信 fallback 和 N6 门禁。

### V14 路由与置信度标定

N4/N6 必须在活动训练集内部标定，测试标签只用于最终评估。

活动固定策略为 `filtered-rule-three-channel-v2`：N4 只能输出 N5a、N5b、N5c。
`S_feature` 使用完整可解释 token 的 IDF-Jaccard，`S_graph` 使用 token 语义前缀关系图的
IDF-Jaccard；N5a 要求两者均为 1.0，N5b 要求两者均不低于 0.70，其余进入 N5c。
冲突、缺失证据和历史证据链进入分支推理载荷；N6 是三通道
单次推理之后的置信度门禁，不是第四个推理通道。

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
`367 passed in 20.85s`；`tests/test_baseline_lock.py`、legacy 图 hash
`5e10b5b25d559777`、legacy Prompt v14 以及活动 local/remote Prompt 隔离测试全部通过。

### V20 高相似异标签人工复核

固定训练知识包先对测试 case 计算标签无关的 `S_feature` 与 `S_graph`，完成候选排序后才
读取训练/测试标签，把 `min(S_feature, S_graph) >= 0.70` 且标签不同的组合写入人工队列。

通过条件：

- 精确冲突和近似冲突分开显示，不把相似自动解释为标签错误。
- 每个测试主 case 展示全部入选训练近邻的共享、单侧独有和互斥证据，以及原始遥测。
- 人工结论与原始数据、知识包隔离保存，支持导入/导出和审核完成状态。
- 只有人工确认结果才能进入后续 expert label 版本；当前 v1 数据和 N8 保持冻结。

当前证据：工作台包含 125 个冲突组，其中双精确冲突 7 个；该项保持 READY，等待领域
专家完成复核并导出签名标注结果。

### V21 可解释特征置换不变性与复杂度

活动特征不能绑定具体 lane 编号。验证采用保持同号跨端逻辑配对的同步置换：对每条 case
反转所有 lane-valued 字段的数值 key，使用冻结训练阈值和 FeatureModel 重新抽取 token。

通过条件：

- 608 条 case 的置换前后 token 集合完全相同。
- 报告逐特征族展示物理定义、通俗解释、标签分布、token 支持度和改造结论。
- 明确区分 lane 编号置换不变性与 4/8 lane 宽度分布可比性。
- 标签关联只作为统计审计，不写入物理约束或自动回灌知识包。

验证证据：608/608 条同步置换不变，变化 case 为 0；7 个活动家族共产出 87 种 token，
形成 474 个 signature，其中 403 个为 singleton。审计结论已写入独立 JSON 与 HTML；
审计结论已落实为 `filtered_rule_v2`：4 个根因 signature 家族、72 种 token、平均 5.83
token/case、425 个 signature 和 334 个 singleton；v2 同步置换仍为 608/608 通过。
正式入口已切换到 v2，v1 与 legacy 指纹保持可复现。

### V22 证据图结构与诊断链有效性

证据图需要区分用于检索的 Case—FeatureToken 事实层，以及用于解释的 SOP—Constraint—Outcome
诊断过程层。过程 trace 只有在包含有效结论并与确认标签一致时，才能升级为可复用历史因果链。

通过条件：

- 检索相似度不读取训练 label，N5a 复用前执行 signature 标签纯度检查。
- 混合标签与单例 signature 不直接作为高置信自动复用模式。
- 统计树节点与物理约束节点使用不同类型，不把统计先验写成物理证据。
- 每条可复用诊断链必须包含有效 Outcome、确认来源、关键证据引用和显式 ConstraintCheck。

当前证据：事实层有 124 Case、64 FeatureToken 和 708 条边；111/117 signature 为单例，
3 个混合组覆盖 7 条case。诊断层只有 13/124 个 Outcome 含 verdict、4 个显式
ConstraintCheck，尚不满足“124条确认历史因果链”的门禁，因此状态保持 READY。

### V23 训练增强排障决策图谱

决策知识按“专家骨架—物理因果门—训练路径统计”三层保存。专家阈值和方向关系不从训练
标签拟合；训练标签只用于路径支持度、冲突和可靠性标定。测试case与测试label不得参与构图。

通过条件：

- L1/L2使用同一套side参数化谓词，lane影响范围使用比例或同步置换不变表达。
- 无异常、关键缺测、端口双Down和未解决冲突进入证据不足，不默认任一端。
- 接收侧症状必须经过发送健康、paired-lane和介质证据门后才能定向。
- fiber必须有正向介质证据；规则冲突本身不是fiber证据。
- 训练路径的support、标签分布和Wilson下界与物理规则字段严格分层。
- singleton或混合标签路径不进入自动终裁。

当前证据：v1图谱有35节点、55边、68个训练路径模板；关闭兜底后覆盖111/124，覆盖内
准确率63.96%。47个路径为singleton、12个路径标签混合，因此图谱结构已经实现，但自动
终裁准入仍需后续训练内折外标定与专家审核，状态保持READY。

### V24 决策图谱测试期逐case复盘

测试必须先对去标签case生成并冻结预测，再揭示真实标签分类复盘。复盘分类至少覆盖正确、
决策图错误、特征问题、证据缺失、过拟合/不可辨识和疑似标签问题；没有命中的类别允许为空。

通过条件：

- 两个测试split分别报告coverage和accuracy at coverage。
- 484条case均有独立HTML，展示盲推理与标签揭示后分析。
- `blind_predictions.json`不含actual/label，`blind_freeze.json`记录预测hash与知识版本。
- 标签疑点和不可辨识case独立保存，不自动回灌。
- 主HTML提供错误归因分布和按优先级排列的优化建议。

验证证据：484条盲预测hash为
`1051c34640154ec0fbbe3af0ee0d1c69c3ebadf2fce37697be8603309a8b78e6`；总体覆盖134/484，
覆盖内89/134（66.42%）。all_data为69/94（73.40%），rule1为20/40（50.00%）。
484个逐case页面、主HTML、label_suspects与irreducible清单均通过结构检查，N8冻结。

### V25 决策图谱因果门与测试知情迭代

接收侧单指标规则必须同时通过总体统计、当前 topology_id 分层统计和发送端正向因果门。
接收异常本身不得作为对端故障的充分条件；高等级候选与任何不同方向的物理候选冲突时降级。

通过条件：

- rxpower/media_snr 只有在对端 TxLOS/TxLOL、同方向 tx_down 或发送触底成立时产生端点票。
- `tx_ok_rx_down` 只形成传播路径/介质候选，不反向证明发送端故障。
- 规则可靠性按 `400g-200g-logical4` 与 `400g-400g-logical8` 分层保存并执行。
- 冲突候选不得因证据等级较低而被静默忽略。
- 测试复盘驱动的后续迭代必须标记 `prior_test_informed_iteration=true`，不得冒充独立盲测。

验证证据：训练集 receive-context 统计中没有 `opposite_tx_fault` 样本；聚焦回归覆盖
无因果支持、`tx_ok_rx_down`、TxLOS 和候选冲突四类边界。v3 冻结预测 hash 为
`d4bbcd0f034386d94d3757e3f50ce8a4d81eefce0bf4b7198c76ce0666a415cc`，484 个逐case页面
齐全。错误终裁由 v1 的 45 条降至 v3 的 1 条，但覆盖率同时由 27.69% 降至 0.83%；
因此安全门实现为 PASS，自动定界可用性仍为 READY，不得进入正式发布。

### V26 训练留一正向路径与 logical8 恢复覆盖

正向路径必须同时满足同拓扑、同专家规则、同方向、token相似度、最小近邻数和标签纯度；
测试case不得参与阈值选择。物理候选、专家候选和训练路径候选存在方向冲突时继续降级。

通过条件：

- multi_metric、SerDes、media_snr、rxpower分别保存版本化训练留一配置和通过case。
- 每个测试预测保存实际命中的训练近邻、相似度、支持数和纯度。
- logical4与logical8不共享近邻或路径统计。
- 缺测case输出字段级补采建议，不进行值填充或强制三分类。
- 单近邻错误归类为overfitting；高支持路径错误归类为decision_graph_error。
- 强发送端物理证据与未审核标签冲突时进入annotation_queue，不自动回灌。

验证证据：v3图谱包含38节点、64边；四类训练留一路径选择性准确率均为100%，但覆盖仅
6/7/4/2条。v4测试冻结hash为
`ff62dfcd58a6cf89a4a620f030d101fb3fe8db082ff8805d90118de3b612f166`；总体覆盖66/484，
覆盖内49/66（74.24%），logical8为7/8（87.50%）。测试中仍有8条决策路径错误和8条
单近邻过拟合，因此结构实现为PASS，泛化与发布门禁保持READY。

### V27 rxpower/SerDes 风险收紧与覆盖恢复

通过条件：

- rxpower不得以单条训练近邻终裁，最小同拓扑支持数不少于2。
- SerDes历史匹配只形成advisory候选，不直接进入强/校准终裁集合。
- 新增覆盖不得通过降低rxpower或SerDes相似度、纯度或支持门获得。
- 拓扑统计路径仅在本拓扑support和Wilson下界通过时启用，不跨拓扑借用。
- 主报告分别统计候选命中、终裁票和最终输出，避免把advisory误报为覆盖。

验证证据：v5中rxpower与SerDes自动终裁均为0；SerDes有8条advisory候选。
logical4 media_snr训练拓扑路径覆盖32条、正确27条。总体覆盖71/484、正确57/71
（80.28%），较v4同时提升覆盖和选择性准确率。预测冻结hash为
`4b87a38baccb87a1643a57720e93a2afdb301433919218b83e31deb9dc6cea40`。
门禁实现为PASS；由于v5仍是测试知情迭代，独立泛化状态保持READY。

## 3. 正式实验执行与结果验收

活动实现已经具备正式 GPU 执行入口。运行时按以下顺序完成门禁：

1. 同步脚本确认工作树干净，切换并拉取 `origin/main`。
2. 实验机补跑完整 pytest；legacy 回归失败时不发布结果。
3. GPU 包装器验证数据 manifest、模型目录和空闲 GPU，直接启动 vLLM，不执行 CPU 模型 dry run。
4. 从 124 条 train 以确定性代码构建并持久化知识包，训练侧 LLM 调用固定为 0；重新加载后
   才分别对 417 条和 67 条测试运行 LLM。
5. 两个测试集各自生成指标、traces 和逐 case HTML；测试标签不进入推理输入。
6. 进程内关闭 vLLM，并使用进程外 `nvidia-smi` 复核资源释放。
7. V11、V14、V17、V18 只有在正式 artifact 完整且可审计后从 READY 更新为 PASS。

正式结果的发布条件是 V1-V20 全部达到 PASS、READY（仅人工复核项）或明确的非阻塞
MONITOR 状态。
