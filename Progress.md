# nsdi-agent 当前交付状态

本文记录活动数据、框架适配、验证结果和正式实验状态。长期约束见 `AGENTS.md`，验收标准见 `Validation.md`。

## 1. 当前目标

项目使用历史证据图、物理约束、专家 SOP 和 LLM 校验完成光链路 RCA。统一根因标签为：

- `L1`：本端根因。
- `L2`：对端根因。
- `fiber`：两端之间的链路介质根因。

活动阶段已经完成数据固定和框架适配。下一步是在 GPU 实验机执行正式全流程，分别验收两个测试集。

## 2. 活动数据

固定数据位于 `datasets/filtered_rule_temporal_2025_06_09_v1/`，共 608 条。

| 划分 | 来源 | L1 | L2 | fiber | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | 两个来源合并 | 50 | 63 | 11 | 124 |
| test | `all_data` | 144 | 258 | 15 | 417 |
| test | `rule1_channel_not_4` | 37 | 29 | 1 | 67 |

训练月份固定为 2025-06 至 2025-09，其余月份进入测试。两个测试来源独立评估。

来源标签统一规则：

- `all_data`: `l1 -> L1`, `l2 -> L2`
- `rule1_channel_not_4`: `l3 -> L1`, `l4 -> L2`
- `fiber -> fiber`

Expert label 通过核心遥测精确指纹应用：命中 49 条，修正 27 条，其中训练 6 条、测试 21 条；未命中 case 不做推断式改标。

## 3. 拓扑与 lane 契约

活动拓扑版本为 `filtered-rule-topology-v1`。

| 来源 | 本端 L1 | 对端 L2 | 光学 lane | SerDes lane |
| --- | --- | --- | --- | --- |
| `all_data` | 400G | 200G | 4×4 | 4×4 |
| `rule1_channel_not_4` | 400G | 400G | 8×8 | 通常 4×4，部分缺失 |

数据审计确认 `transmission` 是同编号光学 lane 的跨端差值字段，因此同编号光学 lane 是明确的逻辑配对。当前实现保留：

- `tx_ok_rx_down`、`tx_down`、同 lane 双向触底等状态证据。
- single / partial / all-lanes 影响范围。
- case 内单 lane 相对离群证据。
- 来源、拓扑 ID、每指标实际 lane 宽度和缺测状态。

当前实现不使用绝对 Tx/Rx 差值判定链路损耗，不把 SerDes lane 映射到光学 lane。历史检索优先同拓扑正相似候选；同拓扑没有任何证据重叠时才显式启用跨拓扑兜底。

## 4. 已完成实现

### 4.1 数据与标签

- 数据准备脚本生成固定 split、统一标签、expert label 审计和文件哈希。
- Manifest adapter 同时支持活动字段 `output_file` 与 legacy 字段 `file`。
- 数据 API 支持 `train`、`test/all_data`、`test/rule1_channel_not_4`。
- `EvidencePack` 从每条 case 契约恢复真实来源、拓扑和 lane profile。

### 4.2 特征与证据图

- 活动特征 profile：`filtered_rule_v2`。
- 特征字典：`filtered-rule-feature-dictionary-v2`，当前 hash 为 `7764aa3a23d2ac2e`；
  `filtered_rule_v1` 只保留旧实验复现。
- 新增逻辑同 lane 状态与范围 token，不生成绝对链路损耗 token。
- `GraphCase`、候选和匹配结果保存来源、拓扑及 lane profile。
- Top-K 截断前执行同拓扑优先，避免小 K 丢弃兼容历史。

### 4.3 Prompt 与推理

- L1/L2 在协议中统一为 local/remote endpoint。
- Prompt 输入包含来源拓扑、lane profile、lane 宽度、同 lane 逻辑契约和禁止推断项。
- 活动 Prompt 使用物理约束库与量测契约库，不使用旧数据统计型 measured constraints。
- Prompt 路由按数据契约隔离：legacy N5c 保持 400G/200G 语义和
  `rca-dual-sop-multidim-v14-full-step-ids`，活动数据使用
  `filtered-rule-general-structured-retry-v4`；推理 trace 和 manifest 分别记录实际版本。
- 活动正式流程使用 `filtered-rule-three-channel-v2`：先由训练集冻结可解释特征模型和
  证据图，再分别计算完整 token 的 `S_feature` 和语义前缀图的 `S_graph`，把每条 case
  唯一分到 N5a/N5b/N5c。N5a 要求双相似度均为 1.0，N5b 要求双相似度均不低于 0.70。
  N6 只做受约束推理后的置信度与降级门禁，不再作为推理前第四通道。
- 三个分支使用独立载荷：N5a 注入历史证据链，N5b 注入 shared/missing/conflict 与
  关键缺失证据，N5c 注入完整专家 SOP；每个请求同时携带当前五层物理路径和真实 lane 数值。
- 正式 vLLM 开启 JSON Schema 结构化解码。每条 case 首轮生成一次；仅对 JSON 解析失败
  或物理 checker 未通过的 case 重试，最多 3 轮。三轮后仍失败时保留最后一个可解析候选
  并把物理合规分降为 0；完全不可解析时进入低置信 forced/fallback。
- 活动 Prompt 使用一套通用推理协议，不强制固定步骤数。`sop_step_id` 与
  `cited_predicates` 只在输入确实提供相应内容时引用，不再混用 S1-S5 与 Q0/P/R/L/D。
- N8 自动回灌保持关闭；测试标签只在推理完成后参与指标计算。

### 4.4 正式实验入口

`scripts/run_filtered_rule_temporal_experiment.py` 从 124 条 train 以确定性代码构建知识包，
训练阶段不调用 LLM；知识包落盘并重新加载后，GPU vLLM 只处理两个测试集。每个测试集
独立输出 summary、outcomes、traces 和逐 case HTML。

`scripts/build_filtered_rule_deterministic_knowledge.py` 是本地训练知识构建与逐 case 审计入口。
固定产物位于 `artifacts/filtered_rule_deterministic_knowledge_v1/`，包含知识包、124 条
逐 case 特征/数值/SOP/留一法历史候选、signature 分组和 token 支持统计。

`scripts/run_filtered_rule_temporal_gpu_experiment.sh` 不执行 CPU 模型 dry run；它检测空闲显存和模型结构，在 1–4 张 GPU 中选择最大的合法 tensor parallel size，并保存运行前后 GPU 快照、命令和日志。

`scripts/run_synced_filtered_rule_experiment.sh` 要求干净工作树，固定切换本地 `main`，从 `origin/main` 执行 `pull --ff-only`，成功后只提交本轮结果目录并推送远端 `main`。

## 5. 已验证结果

数据完整性检查：

```text
python3 scripts/prepare_filtered_rule_temporal_split.py --check
ok=true, case_count=608, errors=[]
```

针对 manifest、标签映射、拓扑、EvidencePack、同拓扑 Top-K 优先和跨拓扑兜底的 10 个无 fixture 断言均通过。新增 Python 文件通过 `py_compile`，两个 shell 入口通过 `bash -n`。

活动训练核心构建检查：

- 124 条训练 case 可在约 5 秒内构建活动特征字典、证据图和 learned SOP，LLM 调用数为 0。
- 图中来源分布为 `all_data=88`、`rule1_channel_not_4=36`。
- v2 证据图版本为 `evidence-graph-v1:124:b60df2407a47cbde`，包含 124 个历史 case 和
  124 条训练确认诊断子图；可跨机器复现的知识包 hash 为 `565a61e23207d798`。
- v2 训练集 117 个 signature 中 114 个标签纯净，3 个混合标签 signature 覆盖 7 条 case；
  单例支持仍占多数，因此训练纯度不能直接解释为泛化能力。
- 数值 learned SOP hash 为 `2e84eb36c2257ea7`；训练内命中 79/124，fiber 为 0/11。
  SerDes SNR 数值尺度仍未完成量测语义确认，该树只作为统计先验和审计路径。
- 417 条 `all_data` 核心检索中 415 条 Top-1 来自同来源，2 条显式跨拓扑兜底。
- 67 条 `rule1_channel_not_4` 核心检索全部 Top-1 来自同来源，无跨拓扑兜底。
- 逻辑同 lane token 在两个测试集中分别覆盖 157 条和 15 条。

无 LLM 的 v2 train/test 分布审计确认：

- `all_data` 测试双相似度精确匹配 33/417，S_feature/S_graph 中位数为 0.664/0.721，
  最近历史标签直接复用准确率为 51.80%，有 10 种同来源训练未见 token。
- `rule1_channel_not_4` 测试精确匹配 3/67，S_feature/S_graph 中位数为 0.500/0.635，
  最近历史标签直接复用准确率为 46.27%，有 19 种同来源训练未见 token。该来源的时间
  schema 漂移仍然显著，不能因为 signature 精简而提高历史候选权重。
- `rule1_channel_not_4` 存在显著时间 schema 漂移：同来源训练仅 6/36 的两端 SerDes
  缺失，测试为 67/67 缺失。缺测只降低完整度，不参与根因投票。
- 两个来源均存在标签先验漂移；统一训练池共享物理知识，但历史标签、阈值和正式指标
  必须按来源分层解释。

标签盲测已按“预测先冻结、评估后读标签”的两阶段协议覆盖 484/484 条测试 case，产物位于
`artifacts/filtered_rule_blind_case_review_v1/`：

| split | 冷启动物理推理 | 训练知识增强 | 净变化 |
| --- | ---: | ---: | ---: |
| test/all_data | 276/417，66.19% | 245/417，58.75% | -31，-7.44pp |
| test/rule1_channel_not_4 | 43/67，64.18% | 34/67，50.75% | -9，-13.43pp |

知识增强在两个 split 分别纠正 46/3 条，同时干扰 77/12 条。当前证据图近邻与 learned SOP
没有带来净提升，也没有识别任何 fiber；不能宣称达到 80%-90%。本轮预测 hash 分别为
`96d6bb7a5ab94aff` 和 `2de580038d7b79d0`，不得依据已揭示的测试标签回调同轮融合权重。
两条纯净精确历史模式冲突进入 `label_suspects.json`，其余错误按知识干扰、不可辨识或
模型/知识错误逐 case 保存在 HTML 报告中。

独立的当前模型逐 case 盲审与专家文档增强复判位于
`artifacts/current_model_case_review_v1/`。冷启动预测先覆盖并冻结 484/484 条，随后才读取
测试标签；专家增强阶段固定读取 `/Users/ziangchen/Downloads/专家模型.md` 所述阈值、
短路异常检测、单端规则优先级和双端裁决，不使用测试标签调参。结果如下：

| split | 当前模型冷启动 | 固定专家规则增强 | 净变化 |
| --- | ---: | ---: | ---: |
| test/all_data | 175/417，41.97% | 225/417，53.96% | +50，+11.99pp |
| test/rule1_channel_not_4 | 31/67，46.27% | 43/67，64.18% | +12，+17.91pp |

专家增强在两个 split 分别救回 142/19 条，同时干扰 92/7 条。其结构性输出分布为
L1=291、L2=185、fiber=8：144 条无异常 case 按文档默认返回 L1，形成明显本端偏置；
fiber 仅在双端最高规则同优先级且定界相反时输出。`all_data` 的 fiber precision/recall/F1
为 28.57%/13.33%/18.18%，`rule1_channel_not_4` 唯一 fiber 未命中。该专家规则显著优于
本轮当前模型冷启动，但仍未达到 80%-90%，也不能替代训练证据图与受约束推理主链路。

冷启动和专家增强预测 SHA-256 分别为
`30674155eaa733cebbf43c8c0dfa12282bdd832763eaa5264f9b85d29aba1998` 与
`265a72addd646d1aae1ad054bcff4eb3bf238666268bb0b3bd7de9b1da573328`。
统一 HTML 提供 484 条真实标签、冷启动判断、专家规则、专家增强判断和正误对照。

活动标签冲突复核工作台位于 `artifacts/filtered_rule_label_annotation_v1/`。它使用冻结训练
知识包先完成特征与语义图检索，再揭示测试标签构造人工复核队列，不修改数据或知识包：

- 共发现 125 个 `min(S_feature, S_graph) >= 0.70` 的训练/测试异标签冲突组，其中双精确
  冲突 7 个；`all_data` 99 个，`rule1_channel_not_4` 26 个。
- 工作台按测试主 case 聚合最多 8 个异标签训练近邻，展示共享、测试独有、训练独有、
  互斥证据，原始遥测，以及冷启动/专家增强判断。
- 人工复核支持筛选、检索、建议标签、证据充分性、问题类型、决定性证据、缺失证据、
  审核人与完成状态；浏览器本地自动保存并支持 JSON 导入/导出。
- `all_data` 的 192 条专家增强 bad case 已形成 5 条可审计经验。最大风险是 83 条
  `no_anomaly_default_L1` 和 137 条 `L2 -> L1` 错误，说明“未检出异常”必须降级而不能
  默认本端，单侧接收质量异常也必须通过 Tx/Bias、Rx/LOS、transmission 与同 lane 关系
  构成方向证据链后才能终裁。

活动可解释性特征逐维审计位于 `artifacts/filtered_rule_feature_review_v1/`，覆盖固定数据集
全部 608 条 case 和 `filtered_rule_v1` 的 7 个活动特征族：

- 对每条 case 同步反转所有数值 lane key 后重新抽取特征，608/608 条 token 完全不变；
  当前特征不绑定具体 lane 编号。该结论只覆盖保持跨端同号逻辑配对的同步置换，不允许
  独立打乱任一端后仍声称物理证据等价。
- 当前实际产出 87 种 token，平均每 case 9.16 个，组合成 474 个 signature，其中 403 个
  只出现一次，存在明显 signature 过度碎片化。
- `status_fault` 可保留；`paired_lane_state` 在同号 lane 数据契约门禁下保留；
  `signal_drop`、`lane_imbalance`、`level_tail` 需要按指标语义和 4/8 lane 宽度改造；
  `telemetry_gap`、`serdes_state` 应移出主检索 signature，进入 N6 量测质量门禁。
- 置换不变不代表跨 lane 宽度分布不变：`min`、`max-min` 和异常 lane 数都会随 lane 数变化。
  后续应使用异常 lane 比例、MAD/IQR、低分位，并按来源/拓扑/lane width 标定。

上述审计已落成 `filtered_rule_v2`，实现前已将远端 `main` fast-forward 到 `0324759`：

- signature 只保留 `status_fault`、`paired_lane_state`、`signal_drop_ratio`、
  `topology_level_tail` 四个根因证据族；量测缺失和 SerDes 有效性改由 EvidencePack/N6 读取。
- `signal_drop_ratio` 使用异常 lane 比例的 minority/majority/all 分档；连续量按 topology、
  side、statistic 和实际 lane width 冻结训练边界，media SNR 使用 25% 低分位替代最小值。
- 608 条 v2 审计仍保持同步 lane 置换 0 变化；token 从 87 降到 72，平均每 case 从 9.16
  降到 5.83，signature 从 474 降到 425，singleton 从 403 降到 334。
- 正式入口、确定性知识构建和分布审计默认均切换到 v2；v1 序列化与 legacy 图 hash 保持冻结。

v2 证据图逐层审计与交互可视化位于
`artifacts/filtered_rule_evidence_graph_review_v2/`：

- 检索主图包含 124 个历史 Case、64 个 FeatureToken 和 708 条 `has_token` 边；标签挂在
  Case 上，但倒排索引与相似度只读取 token，标签读取仍是 N5a 的显式动作。
- 117 个训练 signature 中 111 个是 singleton；3 个混合标签 signature 覆盖 7 条 case。
  当前图首先是稀疏案例索引，尚不能描述为高复用的规则图。
- 124 个诊断子图共有 1379 个节点、2681 条边，但只有 4 个显式 ConstraintCheck；79 条
  numeric decision tree 步骤属于统计先验，不是物理证据。
- 124 个 Outcome 中只有 13 个含自动 verdict，其中 12 个匹配训练确认标签；其余 111 个
  是降级/空结论。因此当前 diagnosis 子图主要保存确定性流程 trace，不能整体宣称为
  124 条已经确认的历史因果排障链。
- HTML 支持按分支/Case 检索并绘制单 case 的 Feature—SOP—Constraint—Outcome 子图，
  同时提供 signature 纯度和 FeatureToken 倒排索引审计。

基于上述问题，新增训练增强排障决策图谱
`artifacts/filtered_rule_decision_graph_v1/`。它不再把 Case—Token 相似度图当成全部知识：

- 专家文档提供量测阈值、异常谓词、单端模式与双端仲裁骨架；物理/量测契约增加质量门、
  同 lane 方向因果门和 fiber 正证据门；124 条训练case只给路径附支持数、标签分布和
  Wilson下界，不把训练统计写成物理规则。
- 图谱包含35个可执行语义节点、55条决策边和68种训练观测路径。所有指标谓词使用side占位，
  对L1/L2完全对称，也不绑定4/8 lane编号。
- 关闭专家兜底后，骨架覆盖111/124（89.52%），覆盖内命中71/111（63.96%）。剩余case
  进入`证据不足`，不再用“无异常默认L1”填充覆盖率。
- 68种训练路径中47种为单例、12种存在混合标签，说明训练case可以丰富路径，但尚不能把
  每个观测组合升级成自动规则；低支持与混合路径只进入审计/候选。
- 原多指标规则支持43、准确率53.49%、Wilson下界38.92%；SerDes单指标支持35、准确率
  57.14%、下界40.86%，两者必须经过因果门禁。media_snr与rxpower单指标在训练内较好，
  但支持仅12/10，仍保留为候选而非无条件终裁。
- 端口Down改为严重度上下文；同优先级相反定界改为冲突；fiber只有在发送健康与介质侧
  正证据成立时输出。HTML可按知识来源和节点类型浏览图，并逐条查看训练路径可靠性。

决策图谱与v2可解释特征的首次测试期 `/loop` 位于
`artifacts/filtered_rule_decision_graph_test_v1/`。预测阶段只读取冻结训练知识，484条盲预测
先落盘并以 SHA-256 `1051c34640154ec0fbbe3af0ee0d1c69c3ebadf2fce37697be8603309a8b78e6`
冻结，之后才读取真实标签逐case复盘：

| split | case | 自动结论 | 覆盖率 | 覆盖内正确 | 选择性准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| test/all_data | 417 | 94 | 22.54% | 69 | 73.40% |
| test/rule1_channel_not_4 | 67 | 40 | 59.70% | 20 | 50.00% |
| 总体 | 484 | 134 | 27.69% | 89 | 66.42% |

- 350条降级为证据不足；逐case复盘归因为：正确89、决策图覆盖缺口200、决策图错误40、
  关键证据缺失150、可解释特征问题5。本轮没有case达到严格“疑似标签问题”门禁，也没有
  依赖单条精确历史产生的自动结论，因此对应清单为空。
- `all_data`的选择性准确率较好但覆盖率低；`rule1_channel_not_4`覆盖较高但准确率仅50%，
  说明同一决策路径在400G-400G来源上存在明显适用域问题，必须按来源/拓扑标定。
- 40条错误全部来自通过训练可靠性门禁的rxpower/media_snr方向规则，其中rxpower 29条、
  media_snr 11条。接收侧症状不能只按“指向对端”终裁，需加入对端Tx健康、同lane传播方向、
  LOS/LOL和介质候选排除边。
- 200条图谱缺口主要来自训练下界不足的multi_metric、SerDes和端口状态路径。它们需要丰富
  因果链和补采策略，不应简单降低门禁换取覆盖率。
- 主HTML提供总览、错误分类与优化建议；484个独立页面展示盲推理步骤、完整特征、专家路径、
  物理/历史候选和真实标签揭示后的复盘。N8保持冻结。

本机项目虚拟环境已安装 pytest 9.1.1。完整回归结果为：

```text
.venv/bin/python -m pytest -q
367 passed in 20.85s
```

回归同时锁定 legacy 证据图 hash `5e10b5b25d559777`、legacy Prompt v14、活动
local/remote Prompt 独立版本与 topology-aware hash。正式实验机仍需在拉取最新 `main`
后按同步入口再次执行门禁。

活动数据三通道静态路由分布：

| split | N5a | N5b | N5c | 推理前 N6 | LLM 请求数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train LOO | 9 | 36 | 79 | 0 | 0 |
| test/all_data | 33 | 133 | 251 | 0 | 417 |
| test/rule1_channel_not_4 | 3 | 8 | 56 | 0 | 67 |

首轮生成请求固定为两个测试集共 484 条。后续批次只包含前一轮解析或校验失败的 case；已经通过的
case 不重复生成。每条 case 的 `attempt_count` 必须位于 1–3。

## 6. 正式配置

默认模型为 `/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B`。

- routing policy：`filtered-rule-three-channel-v2`
- feature profile：`filtered_rule_v2`
- M9 candidate order：仅 `branch`
- Top-K：全量候选
- N8：冻结
- seed：42
- dtype：BF16
- max model length：32768
- max new tokens：16384
- max attempts：3（仅失败 case 重试）
- structured output：JSON Schema guided decoding
- tensor parallel：根据空闲 GPU 和模型结构自动选择，最多 4

正式同步运行：

```bash
scripts/run_synced_filtered_rule_experiment.sh
```

跳过 Git 同步直接运行：

```bash
scripts/run_filtered_rule_temporal_gpu_experiment.sh
```

## 7. 待完成工作

1. 在 GPU 实验机通过同步入口执行正式实验。
2. 审核两个独立 HTML 报告、fiber 个案、降级比例和跨拓扑兜底 case。
3. 根据训练内标定和正式 bad case 归因开展消融，不修改同轮测试知识。
4. 正式结果稳定后归档不再使用的旧实验说明和重复文档；legacy 代码与基线 artifact 暂时保留。

旧 organized、l2fixed 和 expanded 结果只作为历史参考，不与活动数据指标混表。

## 8. 决策图谱 P0 修复与测试知情 `/loop` 复盘

活动决策图升级为 `artifacts/filtered_rule_decision_graph_v2/`：37 个节点、61 条边。新增
接收症状发送端校验门和 topology_id 分层可靠性门。训练审计确认 rxpower/media_snr 的
26 条路径全部属于 `uncorroborated_receive_symptom`，没有一条包含明确
`opposite_tx_fault`；因此接收异常不再直接产生对端终裁票。高等级候选与任一物理候选
方向冲突时统一降级。

测试知情迭代报告位于 `artifacts/filtered_rule_decision_graph_test_v3/`。每一轮仍先对去标签
输入冻结预测，再揭示标签，但修复方向来自 v1/v2 测试复盘，不能作为新的独立盲测：

| 迭代 | 覆盖 | 覆盖内正确 | 选择性准确率 | 证据不足 |
| --- | ---: | ---: | ---: | ---: |
| v1 基线 | 134/484 | 89 | 66.42% | 350 |
| v2 因果/拓扑门 | 9/484 | 4 | 44.44% | 475 |
| v3 冲突否决 | 4/484 | 3 | 75.00% | 480 |

v3 的 `all_data` 为 3/4（覆盖率 0.96%，选择性准确率 75%）；
`rule1_channel_not_4` 为 0/0（覆盖率 0%）。复盘分类为正确 3、决策图缺口 330、关键证据
缺失 150、特征问题 1。修复阻断了已知错误路径，但覆盖率已低到不可用于全量自动定界；
下一阶段的 P0 是在训练边界内补齐 multi_metric、SerDes 和 logical8 paired-lane 的正向
可执行路径，而不是放宽门禁。

主报告、`report.html`、冻结预测、逐 case JSON 以及 484 个独立 case 页面均已生成。v3
预测 SHA-256 为 `d4bbcd0f034386d94d3757e3f50ce8a4d81eefce0bf4b7198c76ce0666a415cc`，
知识包 hash 为 `565a61e23207d798`，N8 保持冻结。

## 9. 决策图谱 v3 正向路径与 `/loop` v4

`artifacts/filtered_rule_decision_graph_v3/` 在因果门、拓扑门和冲突否决之外，新增训练留一
正向路径门。路径只检索同 topology_id、同专家规则、同方向的训练近邻，并分别为
multi_metric、SerDes、media_snr、rxpower设定训练留一相似度、最小支持、纯度和Top-K。
四类路径在训练留一中覆盖6、7、4、2条，均为100%正确；该数字只用于门禁，不作为测试指标。

`artifacts/filtered_rule_decision_graph_test_v4/` 的484条预测先冻结，SHA-256为
`ff62dfcd58a6cf89a4a620f030d101fb3fe8db082ff8805d90118de3b612f166`，之后才揭示标签：

| split | 覆盖 | 覆盖率 | 覆盖内正确 | 选择性准确率 |
| --- | ---: | ---: | ---: | ---: |
| test/all_data | 58/417 | 13.91% | 42 | 72.41% |
| test/rule1_channel_not_4 | 8/67 | 11.94% | 7 | 87.50% |
| 总体 | 66/484 | 13.64% | 49 | 74.24% |

v4相对v3恢复62条自动覆盖；logical8由0覆盖恢复到8条、正确7条。分路径测试结果为：
multi_metric 20/27、media_snr 16/17、rxpower 6/10、SerDes 4/8。后两类泛化明显不足，
下一轮必须提高支持并加入方向性否定证据，不能继续降低相似度。

复盘分类为正确49、决策图缺口268、决策图错误8、单近邻过拟合8、关键证据缺失150、
疑似标签问题1。`case_5f9fb799fec41356`表现为L2发送触底并有Tx状态支持，但当前标签为
fiber，已写入`annotation_queue.json`等待人工审核，未自动改标。每个case页面新增针对性的
补采清单；缺测保持missing，不做多数类或正常值填充。

## 10. rxpower/SerDes 收紧与 `/loop` v5

决策图谱 v4 将 rxpower 的训练近邻门由1条提高为至少2条；当前训练留一没有路径满足该门，
因此测试自动终裁为0。SerDes保留原3近邻门，但输出等级改为advisory，8条测试命中全部只作
候选、自动终裁为0。两类均不再贡献覆盖或错误终裁。

覆盖恢复不借用这两类路径，而是使用训练内通过 support>=8、Wilson下界>=0.50 的
logical4 media_snr拓扑路径；logical8不共享该统计。v5冻结预测SHA-256为
`4b87a38baccb87a1643a57720e93a2afdb301433919218b83e31deb9dc6cea40`：

| split | 覆盖 | 覆盖率 | 覆盖内正确 | 选择性准确率 |
| --- | ---: | ---: | ---: | ---: |
| test/all_data | 63/417 | 15.11% | 50 | 79.37% |
| test/rule1_channel_not_4 | 8/67 | 11.94% | 7 | 87.50% |
| 总体 | 71/484 | 14.67% | 57 | 80.28% |

相对v4，覆盖由66增至71，正确数由49增至57，选择性准确率由74.24%增至80.28%。
logical4 media_snr路径覆盖32条、正确27条（84.38%）。复盘分类为正确57、决策图缺口263、
决策图错误4、特征/量测问题5、单近邻过拟合4、关键证据缺失150、疑似标签问题1。
主HTML和484个逐case页面位于`artifacts/filtered_rule_decision_graph_test_v5/`。

## 11. 未覆盖样本 LLM + 物理约束可行性复盘

`artifacts/filtered_rule_uncovered_llm_feasibility_v1/` 将 v5 的413条未覆盖case与远端
`filtered_rule_temporal_20260823T122701Z`已冻结的真实LLM推理结果按case_id连接。连接时先构造
不含真实标签的`blind_join.json`，之后才生成标签感知的`case_reviews.json`；没有修改知识包、
决策图、预测或标签。

强制使用LLM建议标签时命中225/413（54.48%）；旧LLM门禁实际放行216条、命中119条
（55.09%）。直接合并v5后覆盖率可达59.30%，但覆盖内准确率降至61.32%，因此不能把LLM
强制三分类作为提升覆盖率的正式方案。150条missing/no-anomaly仅52.67%，logical8的59条仅
42.37%；280条回答至少触发一项约束校验问题，fiber 14条没有一条被正确建议为fiber。

探索性子集`LLM final ∩ N5c ∩ 物理专家候选一致`覆盖29条、正确24条（82.76%）；与v5合并后
覆盖100/484（20.66%）、正确81/100（81.00%）。该门来自测试复盘，95% Wilson下界仅65.45%，
只能作为下一轮训练留一/独立验证假设，不能直接写入正式策略。主HTML和413个逐case页面记录
盲态推理链、约束违规、建议标签、真实标签及复盘结论。

逐case页面支持对可解释特征和LLM引用证据执行hover/键盘聚焦：浮层展示对应端点、原始lane
数值、状态字段、同拓扑训练阈值和触发依据。页面底部另有完整原始遥测折叠区；该区域已剥离
`_dataset_contract`和真实标签，避免标签信息混入证据展示。

## 12. 专家因果规则修复与未覆盖样本全量重判

活动数据新增 `filtered-rule-expert-causal-v2` 专家评估层，保留旧专家结果用于审计，但不再
沿用“某端接收异常即归因另一端”的固定映射。新版规则区分发送端直接故障、远端发送故障
导致的本端接收异常、双向接收掉光、主机侧状态佐证以及仅有接收链相关性等不同因果层级。
训练集校准表明，18 条本端 Rx/media/SerDes 对齐异常的标签分布为本端 5、对端 10、fiber 3；
该签名不具备唯一方向性，因此现在只形成有序候选集合并进入人工复核，不能自动反转标签。

`decision.py` 同时增加致命输出门禁：存在 fallback、`physical_compliance<=0`、零上限惩罚或
推理步骤汇总与最终 verdict 不一致时，LLM 的标量置信度不能重新取得自动终裁资格。由此修复
了 `case_1990585214d8ea2b` 一类“正文指向 L2、最终却输出 L1 且物理校验失败”的错误放行。

新版复盘位于 `artifacts/filtered_rule_uncovered_llm_feasibility_v2/`。413 条未覆盖样本全部
重新生成页面；80 条旧终裁因致命校验失败被降级，16 条存在推理步骤与最终 verdict 冲突，
21 条表现为独立物理候选命中真实标签而旧最终标签错误，已单列到
`reasoning_label_mismatches.json`。标签只在盲态修正冻结后用于复盘分类。

当前修正规则自动终裁 68 条、正确 39 条（57.35%），仍不满足发布要求；其价值是阻断错误
终裁并给出可审计候选，而不是宣称已恢复高覆盖率。v5 的 71 条决策图终裁仍是当前有效基线，
新版因果层需经过训练留一冻结和独立验证后才能进入正式自动覆盖策略。

## 13. 完整测试集非空分析与 Host SNR 可选化

完整测试集复盘位于 `artifacts/filtered_rule_full_test_analysis_v3/`。两个来源共484条case均
输出非空 `analysis_verdict`，取值固定为L1/L2/fiber；同时独立保存`corrected_action`和
`analysis_confidence_tier`。因此人工复核或补采不再表现为null，但也不会被误报成安全自动终裁。

Host SNR在活动链路中改为纯增强证据。原始缺失状态继续保留用于数据质量审计，但
`diagnostic_missing_fields`、活动Prompt和补采清单不再把host_snr缺失视为关键缺失；仅当存在
有效观测且与本端电口方向一致时，才增强本端候选。接收症状物理约束也由“默认支持对端”收紧
为“约束在对端发送链、介质、本端接收链三者内”，必须有独立方向证据才能唯一归因。

本轮冻结结果为：完整测试集288/484（59.50%，Macro-F1 39.39%）；`all_data`为255/417
（61.15%），`rule1_channel_not_4`为33/67（49.25%）。未覆盖413条为231/413（55.93%）。
高置信动作覆盖139条、正确96条（69.06%）；原v5冻结路径仍为57/71（80.28%）。完整强制分析
满足“每条都有结论”的产品展示要求，但当前不具备替代v5安全终裁门的精度。

`case_1990585214d8ea2b`现按同lane L2接收链证据输出最终分析L2，安全动作保持human_review；
这体现了“必须给出最佳判断”和“是否允许自动执行”两个字段的职责分离。
