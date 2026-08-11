# nsdi-agent 重构进度

本文是 `nsdi-agent/` 的重构进度台账。后续 AI 写代码前必须先阅读 `AGENTS.md` 和本文，确认当前任务、已完成改造、legacy 回归锚点和新框架实验门禁。

三份文档的分工：`AGENTS.md` 写约束，本文写已完成的事和已测出的数字，`Validation.md` 写还需要人拍板、需要外部输入或需要新数据才能验证的事。遇到「这个数该定多少」「这个字段什么意思」这类问题，先查 `Validation.md`，那里每条都写了未确认前的默认取值。

## 0. 2026-08-09 进度审计摘要

本节按当前工作树、测试、脚本和 `artifacts/` 重新核对，不沿用已经过时的任务描述。

### 0.4 2026-08-10 RCA v2 / l2fixed 重构落地

本节记录基于 `datasets/rca_v2_l2fixed` 的 v2 重构事实。旧 organized 126/85 与
legacy 58/85 仍是回归锚点，不与本节数字混比。

- **新数据契约已建立：** `scripts/prepare_l2fixed_stratified.py` 只写
  `datasets/rca_v2_l2fixed/_metadata/manifest.json` 与 `quality_report.json`，
  不改写 case 文件。manifest 为 seed=42、60/40 分层切分：train 161、test 107；
  标签分布为 train L1 49 / L2 100 / fiber 12，test L1 32 / L2 67 / fiber 8。
- **l2fixed 数据质量边界：** 全量 268 条，标签分布 L1 81 / L2 167 / fiber 20；
  54 条缺 `alarm_ip_interface`，54 条缺 `Lane number`，31 条仍有 L1 侧指标宽度大于 4；
  L2 侧宽度已由修复脚本收敛到不超过 4。L1 是 400G，不再把 L1 8-lane 自动视为错误。
- **EvidencePack 契约扩展：** `EvidencePack` 增加 `field_states`，把 observed / missing /
  not_applicable / invalid 显式记录下来。标签仍在构造入口被结构性剥离。
- **v2 特征字典已显式接入：** `feature-dictionary-v2` 不是默认字典，必须通过
  `dictionary_for("v2")` 或 `evaluate_routing.py --feature-profile v2` 选择；它在 v1 基础上
  加入 `lane_direction`、`telemetry_gap` 与 `serdes_state`，其中 `serdes_state` 只表示
  有效 / 失效 / 缺失，不解释为 dB SNR。
- **learned SOP 已实现：** `rca_framework/sop/` 中的 `learned-sop-v1` 是训练集标签归纳的
  浅层可解释树，不是专家 SOP。每个叶子记录支持数、标签分布、路径条件与 Wilson 下界；
  低支持或低下界由 M9 降级。
- **证据图 v2 原语已实现：** `EvidenceGraph` 保留 case-token 全局图，并新增
  `CaseDiagnosis` / `DiagnosisNode` / `DiagnosisEdge`，可表达 FeatureToken、SOPStep 与 Outcome
  的 per-case 诊断链。`feedback.py` 只允许带 `confirmed_by` 的人工确认结果回灌。
- **N7/N8 基础闭环已补：** `report.py` 可从分支输出和 M9 决策渲染报告；`feedback.py` 可把
  报告证据链转成诊断子图并附加到 evidence-graph-v2。
- **skill 已拆分：** `skills/rca-domain`、`skills/rca-constraints`、`skills/rca-sop`、
  `skills/rca-evidence-graph`、`skills/rca-workflow` 由 `scripts/render_rca_skills.py`
  统一生成。`rca-constraints` 仍由约束库单点渲染，避免物理约束、SOP 和证据图说明继续耦合。
- **约束库升级：** `constraint-library-v3` 保留 15 条约束，把实测口径改为
  `rca_v2_l2fixed` manifest train split，并为 M7 增加 `allowed_effects`、`allowed_targets`
  和 token 前缀适用范围字段。
- **无 LLM v2 门禁首跑：** 命令
  `python scripts/evaluate_routing.py --data-dir datasets/rca_v2_l2fixed --manifest-split --feature-profile v2 --learned-sop --llm-backend none --policies coverage-v2 --output-dir artifacts/l2fixed_v2_gate_none`
  已生成正式四件套。`coverage-v2` 分流为 N5a 15、N5b 26、N5c 64、N6 2；M9 最终回答
  38/107，答对 27，给结论时准确率 71.05%，全量准确率 25.23%。该数字只是 v2 dry-run
  工作点，不代表方法已优于 legacy。

### 0.1 与目标架构的总体距离

- **主干已打通到 N6：** N1 `EvidencePack`、N2 可解释特征、N3 证据图检索、N4 两套路由、
  N5a/N5b/N5c 处理器、M7 校验器、M8 真实 vLLM 推理和 M9 统一决策出口均已有代码。
- **还不是完整端到端系统：** M6 专家 SOP、M10 报告、M11 回灌脚本和 M12 实验框架
  尚未交付；当前新框架主要靠 `scripts/evaluate_routing.py` 装配。该脚本已能写正式
  manifest/逐 case 产物，但还没有独立 CLI / pipeline。
- **T6 暴露的三处工程缺口已在 T7 代码中闭环：**
  1. N5a 混合桶、N5b 关键缺失/候选冲突和 N5c 现在统一批量进入受约束推理。
  2. LLM confidence 按分支与置信度分桶，在训练集留一法输出上独立标定；未标定输出的
     Wilson 下界固定为 0，不能直接进入最终报告。
  3. `decision.py` 按 Wilson 下界与最小支持数统一输出结论、补采或人工介入。
  真实 32B 已用 prompt v3 和新标定链路完成正式复跑；M9 最终回答 33/85、答对 26，
  给结论时准确率 78.79%，但 `fiber` 仍为 0/6。
- **M7 当前只能保证结构合规，不等于物理语义正确。** 真机输出已经出现“用接收功率约束解释
  发送功率”“用禁止按量纲解释的 C13 反过来支持根因”等语义错配，但现有 checker 只校验
  token/约束编号是否存在、少数禁句和最终排除条件，因此仍会判通过。
- **证据图当前是 case-token 二部索引。** 它已足够支持加权 Jaccard 与版本化，但还没有目标图中
  更强的因果关系、处置动作、人工确认记录和增量审计闭环。
- **版本库交付风险：** 当前 `git log` 只提交到阶段 1；T1-T6 的主体代码、测试、脚本与文档
  仍大量处于 modified/untracked 工作树。本文描述的是“本地工作树可运行状态”，不是已提交版本。
  在继续大改或跨机器实验前，应先由用户决定如何分批提交；本次审计未擅自创建 commit。

### 0.2 当前可信实验结论

1. legacy 锚点仍是 deterministic 58/85（68.24%）和 DeepSeek-32B 59/85（69.41%）。
2. v1 特征字典显著提高 signature 纯净度，但没有突破整体特征上限：
   混合标签覆盖 65.87% → 7.94%，N5a 桶内多数投票 60.87% → 76.19%，
   全测试集纯匹配仍为 58/85；`fiber` 只判对 1/6，F1=0.20。
3. `coverage-v2` 在不接 LLM 时提供 37/85 的覆盖，给结论时 29/37=78.38%；
   它是“高精度、低覆盖”的选择性出口，不是 85 条全量三分类器。
4. 真实 DeepSeek-R1-Distill-Qwen-32B/vLLM 已于 2026-08-07 跑通，推翻了本文旧版
   “本机无 GPU、尚未真机验证”的描述。`coverage-v2` 下 N5c 回答 22 条只对 5 条
   （22.73%），整条新链路回答 59/85、答对 34 条，给结论时准确率 57.63%，
   全量准确率 40.00%。模型会主动弃权 24/46，但新增回答的质量不足，不能作为方法有效性结果。
5. 上述 T6 产物只能算**工程真机首测**，还不满足正式实验门禁：
   `t6_llm_evaluation.json` 未记录模型路径、prompt hash、约束/SOP 版本、生成参数和 seed，
   未保存逐 case trace/prediction，也没有三类分层指标，不能直接进论文结果表。
6. 2026-08-10 的独立规则 empirical study 显示：只给证据时模型回答 19/85、答对 7；
   加当前规则后回答 15/85、答对 8；再加 checker 后回答 14/85、答对 7。
   规则把“给结论时准确率”从 36.84% 提到 53.33%，但只净增 1 个正确 case，
   配对 McNemar `p=1.0`，不能宣称有显著提升；三组 `fiber` 都是 0/6。

### 0.3 当前实验设计完成度

- **已完成：** legacy 产物；特征家族 1023 子集消融；N4 阈值标定；
  两套路由比较；合成失效模式 checker 审计；一次真实 32B 受约束推理首测；
  同 case / 同模型 / 同 seed 的“证据-only、规则 prompt、规则 prompt+checker”三臂配对实验。
- **仅有文档数字：** 多数类 55/85 与 RF 约 70.14% 尚无独立 artifact 路径，
  在补齐可复跑脚本/产物前只能作背景参照。
- **未完成：** Top-N 消融、置信度阈值与 coverage-risk 曲线、训练规模消融、
  SOP 消融、多随机种子或重复切分、
  `fiber` 分层误差分析和统计区间。
- **当前无法直接执行的设计：** 原计划训练规模 50/100/150/200 与现有固定切分冲突；
  当前训练池只有 126 条，150/200 必须等待合并数据集，或重新设计成固定测试集上的嵌套训练子集。
- **下一实验优先级：** 先修正评估记录与 N5a/N5b/N5c 装配缺口，再跑无约束基线和约束消融；
  否则继续调 prompt 只能得到不可归因、不可复现的数字。

## 1. 使用约定

状态图例：

- `未开始`：尚未动代码或文档。
- `进行中`：已有部分改动，但未通过该项验收。
- `已完成`：实现完成且通过对应验收。
- `已冻结`：作为历史规划或兼容锚点保留，不应作为下一步继续执行。
- `已放弃`：经确认不再执行。

更新规则：

1. 每次代码改动前，先看"当前阶段指针"和"门禁清单"。
2. 每完成一个模块或任务，更新对应表格的状态、结果和日期。
3. 每次运行测试或实验，把命令、结果、产物路径写入"门禁运行记录"。
4. 每次引入行为变化，必须说明是否影响 legacy 基线。
5. 不要把 `/home/chenziang/nsdi/` 的状态写成当前进度；这里记录的是 `/home/chenziang/nsdi-agent/`。

## 2. 当前阶段指针

当前任务：`T10 - 独立 LLM 规则消融与经验研究`

当前状态：`首轮三臂配对实验已完成 / 结果未证明当前规则显著提高判断正确数`

阶段 0、阶段 1 均已于 2026-08-06 完成。原阶段 2-5 规划已冻结，不再作为主路线继续推进。

- T1 已于 2026-08-07 完成，冻结特征字典 v1（`feature-dictionary-v1`，`content_hash=1b2e66ed650ce60e`），产物与原始数据见第 9.1-9.6 节。
- T2 已于 2026-08-07 完成代码侧交付；约束库最初冻结为 v1（14 条），
  后因 T5 的 C15 发现升级到 v2（15 条）。15 条全部待夏思博审核，这是 T2 唯一的未完成项。
- T3 已于 2026-08-07 完成，交付 `EvidencePack`（`evidence-pack-v1`）与缺失 / 冲突证据结构，产物见第 9.10-9.11 节。
- T4 已于 2026-08-07 完成，交付证据图 `evidence-graph-v1`（`content_hash=5e10b5b25d559777`）、Top-N 检索与 N4 阈值标定曲线，产物见第 9.12-9.14 节。
- T5 已于 2026-08-07 完成，交付 M4 路由器（两套可配置规则）、N5a/N5b/N5c 三分支处理器、N6 弃权出口与置信度标定，产物见第 9.16-9.20 节。
- 约束库因 T5 的发现升到 `constraint-library-v2`（`content_hash=abb395e9371abc36`，15 条），新增 `C15`，原因见第 9.19 节。
- T6 已于 2026-08-07 完成代码侧交付并完成 DeepSeek-R1-Distill-Qwen-32B/vLLM 真机首测；
  工程链路可运行，但真实模型效果未达标，且真机输出暴露出 M7 语义校验覆盖不足。
  产物与结果见第 9.21-9.26 节。

下一步：

1. 扩展 M7 的约束-证据适用性与 effect/target 一致性校验；把本次 15 条首轮违规和
   8 条重写后仍违规 case 连同 T6 badcase 固化成回归集。
2. 把约束库交夏思博逐条审核（现在是 15 条），并请黄泽舜确认 `Validation.md` V16 的哨兵语义；
   当前规则更偏量测有效性和禁止误推，缺少能区分 L1/L2/fiber 的强阳性规则。
3. 不在当前结果上继续调 checker 阈值；先补专家认可的可判别规则或新证据，再重复相同三臂实验。
4. 之后进入 Top-N、置信度和训练规模消融；150/200 仍需等待合并数据集。

## 3. 任务进度表

| 编号 | 任务 | 对应模块 | 工作量 | 交付物 | 依赖 | 状态 | 验收 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 冻结特征字典 v1 | M1 | 1 天 | 特征字典表、版本号、signature 分辨率报告 | 无 | 已完成 | 已达标：混合标签覆盖 65.87% → 7.94%；N5a 桶内多数投票 60.87% → 76.19%（> 64.71%）；10 个家族全部声明物理含义 / 单位 / 取值域 / 抽取规则 |
| T2 | 整理通用物理约束规则 | M5 | 1-2 天 | 约束规则清单、prompt 化模板 | 无 | 代码已完成 / 待审核 | 约束库已升到 v2 共 15 条（含 C15 哨兵有效性前置条件），全部 `pending_expert_review`，待夏思博审核 |
| T3 | 实现证据到特征向量抽取 | M1 | 2 天 | 抽取器代码、单测 | T1 | 已完成 | `EvidencePack` 使标签隔离成为结构性保证；正常 / 缺失 / 冲突 / lane 级四条路径各有单测（13 个）；`anomaly_id` 有测试锁定不变；211 条真实 case 抽取零冲突 |
| T4 | 构建证据图与 Jaccard 检索 | M2 / M3 | 2-3 天 | 图存储、检索 API、Top-N 结果结构 | T1 / T3 | 已完成 | Top-N 输出相似度、证据覆盖率、缺失 / 多余 / 冲突证据；IDF 权重可注入；图版本号 `evidence-graph-v1:126:5e10b5b25d559777` 可复现；N4 分布 21/26/38 与 T1 一致 |
| T5 | 实现分流路由与三分支处理器 | M4 / N5a-c | 3 天 | 路由器、完全匹配 / 部分匹配 / 通用排障处理器 | T4 | 已完成 | 两套路由规则（`board-100-70` / `coverage-v2`）可配置可统计；N5a 按桶纯净度拆分；置信度由训练集留一法标定而非常数；N5c 无 LLM 时不猜；确定性排除有全量校验 |
| T6 | 接入 LLM 推理与约束校验器 | M7 / M8 | 2 天 | prompt 模板、校验断言、逐步推理日志 | T2 / T5 | 已完成真机首测 / 效果未达标 | 45 个单测覆盖（checker 19 + LLM reasoning 26）；DeepSeek-R1-Distill-Qwen-32B/vLLM 已跑通。`coverage-v2` 的 N5c 回答 22、判对 5、弃权 24；证明模型会弃权，但也证明当前语义校验和推理质量不足 |
| T7 | 置信度 / 降级策略 | M9 | 1 天 | 阈值配置、低置信度出口 | T5 / T6 | 已完成正式真机门禁 | `decision.py` 已统一结论/补采/人工出口；默认 Wilson 下界 0.50、支持数 10（V18 待确认）。32B prompt v3 正式运行最终回答 33/85、答对 26、回答准确率 78.79%，fiber 0/6 |
| T8 | 报告生成器改造 | M10 | 1 天 | 报告模板 | T5 / T7 | 未开始 | 报告显式区分完全 / 部分 / 低匹配来源，并渲染证据链 |
| T9 | 证据图回灌闭环 | M11 | 1 天 | 回灌脚本、版本记录 | T4 / T8 | 部分完成 | `EvidenceGraph.extend` 已提供不可变追加原语并有单测；缺人工确认输入、落盘脚本、审计记录与回放 |
| T10 | 实验 / 消融框架 | M12 | 1-2 天 | 参数化脚本、结果汇总模板 | T5 | 部分完成 | 已新增独立三臂规则 empirical study、正式 manifest/trace、Wilson 区间和配对 McNemar 检验；缺 Top-N/置信度/训练规模/SOP 消融。150/200 受当前 126 条训练池阻塞 |
| T11 | 系统架构图 / 过程说明图 | 架构交付 | 并行 | 论文可用框图 | 无 | 进行中 | 与画板 N1-N8 一致，可复用到论文 |
| T12 | 论文初稿 | 论文交付 | 第 3-4 周 | 方法 / 实验 / 讨论章节草稿 | T10 实验结果 | 未开始 | 与代码版本、实验版本、图表一致 |

### 3.1 历史阶段（已冻结）

| 阶段 | 内容 | 状态 | 结果 |
| --- | --- | --- | --- |
| 0 | `RuntimeConfig` + `build_case_context` 去重 + 基线锁定测试 | 已完成 | pytest 15 passed；58/85；`predictions.json` 与基线字节一致 |
| 1 | 观测字段、证据结构、覆盖分档、支持度分级 | 已完成 | pytest 46 passed；legacy 值不变，只新增观测键 |
| 2 | `evidence_status`、lane pair、lane directional signature 影子模式 | 已冻结 | 路线冻结；未提交代码改归 T1 / T3 |
| 3 | `rca_framework/agent/` + deterministic Agent | 已冻结 | 被 N1-N8 / M1-M12 新主线替代 |
| 4 | LLM 子包拆分 + `abstain` + 选择性评估 | 已冻结 | 被 M8 / M9 / M12 新主线替代 |
| 5 | `skills/` + playbook + trace 回放 | 已冻结 | 被约束库 / SOP / 证据图回灌替代 |

## 4. 逐模块进度表

### 4.1 现有模块

| 模块 | 初始行数 | 当前角色 | 新框架归属 | 状态 | 验收标准 |
| --- | ---: | --- | --- | --- | --- |
| `rca_framework/types.py` | 84 | `ROOT_CAUSES`、`Anomaly`、`CaseEvidence`、`EvidenceItem`、`Verdict` | legacy 与新框架基础协议 | 已完成 | 旧类型行为不变；新增协议有测试覆盖 |
| `rca_framework/data.py` | 332 | 数据清单、脱敏、L1/L2 归一化、数据集加载、`case_side_mapping` | N1 证据标准化与 lane 对齐输入 | 已完成 | 现有数据集无需重生成；旧 loader 签名不变 |
| `rca_framework/anomaly.py` | 262 | 阈值拟合、异常提取、方向性损耗、lane 报告 | M1 特征抽取底层 | 已完成（候选特征保留） | lane 特征已做 1023 子集消融，因无增量未进入 v1；legacy `anomaly_id` 不变 |
| `rca_framework/retrieval.py` | 45 | IDF 加权 Jaccard 检索 | M2 / M3 检索内核 | 已完成 | 支持 Top-N、`hide_labels=True`，后续补缺失 / 冲突证据结构 |
| `rca_framework/graph.py` | 334 | legacy KG 学习、路径评分、feature rules、RAG 检索 | legacy-only + M2 / M3 可复用统计 | 已完成 | `max_retrieval_similarity` 可供 N4 分流；legacy `scores` 不漂移 |
| `rca_framework/rules.py` | 182 | legacy 互斥符号规则学习与匹配 | legacy-only，可作为 M5 约束候选来源 | 已完成 | `rule_overlap` 仍为 0；旧 match 分数一致 |
| `rca_framework/evidence.py` | 新增 | `EvidenceItem` 聚合、同源 / 独立 / 冲突判定 | N6 决策辅助 | 已完成 | `same_source_agreement` 可统计 |
| `rca_framework/llm/` | 原 `llm.py` 218 | legacy 兼容层 + 新协议、prompt、vLLM 后端、受约束推理循环 | M8 + legacy 兼容 | 已完成真机首测 | legacy import 不破；新 prompt 固定版本；真实 32B 已运行，但效果未达标 |
| `rca_framework/fusion.py` | 100 | legacy 两路融合与冲突仲裁 | legacy-only 回归锚点 | 已冻结 | `fuse_results` 不改；legacy 58/85 不漂移 |
| `rca_framework/pipeline.py` | 233 | fit / infer / evaluate / save / load、reasoner 缓存 | legacy pipeline；部分观测统计可复用到 M12 | 部分完成 | legacy 输出逐 case 一致；新框架另设入口 |
| `rca_framework/cli.py` | 169 | `prepare` / `train-evaluate` / `infer` 入口 | legacy CLI；新框架后续新增独立入口 | 未开始 | 旧参数、默认值和命令行为不变 |
| `rca_framework/runtime.py` | 新增 | `RuntimeConfig` | legacy 配置对象 | 已完成 | 旧 kwargs 调用等价；未知参数报错 |

阶段 0 完成后 `pipeline.py` 为 235 行，新增 `CaseContext`、`build_case_context`、`finalize_prediction` 三个装配单元，`infer` 与 `evaluate` 不再各自复制一份装配代码。`cli.py` 改为通过 `runtime_from_args` 传 `RuntimeConfig`，命令行参数、默认值与 `run_manifest.json` 内容均未变。

阶段 1 只往输出里加键，不改任何 legacy 值。新增键共三处：逐 case 的 `observation`、`KG_RCA.matched_rules[*].support_tier`、`KG_RCA.support_tier_counts`，以及 `evaluation_summary.json` 的 `observations`。`CaseContext` 增加 `coverage` 与 `evidence_view` 两个观测量，legacy 决策不读取它们。

### 4.2 待建模块

| 目标路径 | 对应模块 | 目标职责 | 复用来源 | 状态 | 验收标准 |
| --- | --- | --- | --- | --- | --- |
| `rca_framework/features/dictionary.py` | M1 | 特征字典 v1，记录维度 / 物理含义 / 单位 / 取值域 / 版本号 | `anomaly_id`、lane 影子字段 | 已完成 | 10 个家族声明齐全；`content_hash=1b2e66ed650ce60e`；分辨率验收达标 |
| `rca_framework/features/extractor.py` | M1 | 证据包到可解释稀疏特征向量 | `extract_evidence` | 已完成 | 10 个家族抽取 + `FeatureModel` 分位拟合；`EvidencePack` 接口与冲突结构已接入 |
| `rca_framework/evidence_graph/store.py` | M2 | case-token 二部证据图构建、索引、版本与持久化 | `AnomalyKnowledgeGraph.fit` | 已完成基础版 | case / feature token 可持久化，支持 `extend`；尚无处置/因果/人工审计边 |
| `rca_framework/evidence_graph/match.py` | M3 | Top-N 检索与缺失 / 多余 / 冲突证据清单 | `retrieval.retrieve` | 已完成 | 输出候选、相似度、覆盖率、差集与冲突；正式 Top-N 消融未跑 |
| `rca_framework/evidence_graph/router.py` | M4 | N4 分流路由 | `max_retrieval_similarity` | 已完成 | `board-100-70` 与 `coverage-v2` 可配置；独立实验 manifest 尚未实现 |
| `rca_framework/constraints/library.py` | M5 | 结构化物理约束库 | 专家规则、阶段 1 观测 | 代码已完成 / 待审核 | 15 条，版本 `constraint-library-v2`、`content_hash=abb395e9371abc36`；全部待专家审核 |
| `skills/rca-constraints/SKILL.md` | M5 | prompt 化约束说明 | `constraints/library.py` | 已完成 | 由 `scripts/render_constraint_skill.py` 自动生成；测试比对渲染结果，门限只有一处定义 |
| `skills/rca-sop/SKILL.md` | M6 | 专家排障 SOP | 黄泽舜 / 其桐文档 | 未开始 | 指标查看顺序和兜底 SOP 可读 |
| `rca_framework/constraints/checker.py` | M7 | 可执行约束断言 | `constraints/library.py` | 部分完成 | 结构、引用和少数禁句校验已完成；真机输出证明“约束是否适用于该证据/作用方向”仍未校验 |
| `rca_framework/llm/` | M8 | 新框架 LLM 后端、prompt、协议、重写循环 | legacy `llm.py` | 已完成真机首测 | legacy import 不破；真实 32B 可批量推理和主动弃权；正式效果与复现门禁未达标 |
| `rca_framework/branches/` | N5a-c | 完全 / 部分 / 低匹配处理器 | `retrieval`、`evidence`、约束库 | 已完成代码接线 | N5a 混合桶、N5b 关键缺失/候选冲突和 N5c 统一批量进入 LLM；无模型时保留原 dry-run 行为 |
| `rca_framework/decision.py` | M9 | 置信度阈值与降级策略 | `BranchCalibration` | 代码已完成 / 待真机标定 | 历史分支和 LLM 分支统一按 Wilson 下界与支持数决定结论/补采/人工；支持 LLM confidence 分桶独立标定与阈值曲线 |
| `rca_framework/report.py` | M10 | 报告生成 | legacy prediction dict | 未开始 | 报告含路径来源、证据链、处置建议 |
| `rca_framework/feedback.py` | M11 | 证据图回灌闭环 | `EvidenceGraph.extend` | 部分完成 | 已有不可变追加原语；缺人工确认输入、脚本、落盘审计与回放 |
| `scripts/run_ablation.py` | M12 | 参数化消融批跑 | 现有分析/标定/评估脚本 | 未开始 | 现有脚本分散，尚无统一 runner、manifest 与结果汇总 |

### 4.3 未提交 lane 代码处置

当前工作树已有 236 行未提交代码，涉及 `rca_framework/anomaly.py`、`rca_framework/data.py`、`rca_framework/pipeline.py`、`rca_framework/types.py`：

- `lane_pairs`
- `lane_directional_loss`
- `lane_loss_report`
- `case_side_mapping`
- `EVIDENCE_STATUSES`
- `observation.lane_loss` 与 summary 影子统计

处置结论：这些改动不随旧阶段 2 冻结，重新归属到 T1 / T3。原因是 N5a 63.04% 的实测结果表明当前 signature 分辨率不足，lane 级特征是最直接的改进候选。

T1 已对这批代码给出结论（2026-08-07）：

- `lane_directional_loss` 被封装为 `lane_direction` 特征家族，触发率不低（211 条中 102 条命中至少一个 signature），但在 1023 个子集的家族消融中，把它加进 v1 组合会同时拉低 train 留一法的 N5a 桶内准确率与全集准确率，因此**不进特征字典 v1**，状态标记为 `candidate`，未入选理由写在 `dictionary.py` 的 `selection_note` 里。
- `tests/test_feature_dictionary.py::test_extractor_does_not_touch_legacy_anomaly_ids` 已证明抽取器不改写 legacy `anomaly_id`，且两个 token 空间完全不相交。
- lane signature 在全量 211 条上的触发分布已实测，见第 9.3 节。

剩余缺口：`lane_pairs` / `lane_loss_report` 自身的边界条件单测（lane 数不匹配、单侧缺失）仍未补，留在 T3。

## 5. 门禁清单

### 5.1 legacy 回归锚点

每次触碰 legacy 路径前后必须检查：

- [ ] `python -m pytest -q` 全绿，包含基线锁定测试。
- [ ] 默认 legacy `--backend none` 输出 58/85。
- [ ] 逐 case prediction 与 `artifacts/organized_rca_v2_60_40_seed42_baseline/predictions.json` 兼容。
- [ ] 有 GPU 时，DeepSeek-32B vLLM legacy 结果保持 59/85。
- [ ] `RCAPipeline.load` 可读取现有 `artifacts/*/model`。
- [ ] `rules.overlap_audit` 的 total overlap 仍为 0。
- [ ] 阶段 1 起的"只增不改"递归比对仍能发现 legacy 值漂移。

上述门禁是 legacy 回归锚点，不再要求新框架复现 58/85。

### 5.2 新框架实验门禁

每次新框架实验必须检查：

- [ ] `run_manifest.json` 记录证据图版本、特征字典版本、约束库版本、SOP 版本、prompt 模板 hash、Top-N、阈值、训练集规模、随机种子。
- [ ] 报告 N4 分流分布：N5a / N5b / N5c case 数。
- [ ] 报告 N5a signature 纯净度、混合标签 signature 覆盖率、桶内多数投票准确率。
- [ ] 报告 coverage / accuracy、precision_at_coverage、低置信度降级率、人工介入率。
- [ ] 报告 `fiber` 分层指标，不只报 overall accuracy。
- [ ] 纯 LLM 无约束基线、Top-N 消融、置信度阈值消融、训练集规模 50/100/150/200 消融有独立产物。

### 门禁运行记录

| 日期 | 阶段 / 任务 | 命令 | 结果 | 产物 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 2026-08-06 | 初始化 | `python -m pytest -q` | 7 passed | 无 | 复制后基础测试通过 |
| 2026-08-06 | 初始化 | `python -m rca_framework.cli train-evaluate --data-dir datasets/organized_rca_v2_stratified_60_40_seed42 --train-size 126 --output-dir artifacts/copy_verify --backend none` | 58/85，accuracy 68.24%，`rule_overlap=0`，`label_leakage=false` | `artifacts/copy_verify/`（已删除，不入库） | 逐 case prediction 与复制基线完全一致 |
| 2026-08-06 | 0 | `python -m pytest -q` | 15 passed（原 7 + 基线锁定 8） | 无 | 基线锁定测试首次全绿 |
| 2026-08-06 | 0 | `python -m rca_framework.cli train-evaluate ... --output-dir artifacts/gate_stage0 --backend none` | 58/85，accuracy 68.24%，`rule_overlap=0`，`label_leakage=false` | `artifacts/gate_stage0/`（不入库） | `predictions.json`、`evaluation_summary.json`、`run_manifest.json` 与基线字节一致 |
| 2026-08-06 | 0 | 同一命令分别用 `PYTHONHASHSEED=1` 与 `PYTHONHASHSEED=98765` 各跑一次后 `diff -rq` | 两次产物全部字节一致 | 无 | 修正前 `model.json` 的 `idf` 键序每次都不同 |
| 2026-08-06 | 1 | `python -m pytest -q` | 46 passed（原 7 + 基线锁定 13 + 观测层 26） | 无 | 含"只增不改"递归比对与观测数字锁定 |
| 2026-08-06 | 1 | `python -m rca_framework.cli train-evaluate ... --output-dir artifacts/gate_stage1 --backend none` | 58/85，accuracy 68.24%，`decision_status` 仍为 `agreement` 82 / `conflict_resolved_by_kg_rag_llm` 3 | `artifacts/gate_stage1/`（不入库） | 递归比对显示相对基线只新增 7 类键，0 处值改变，0 处键消失 |
| 2026-08-06 | 1 | `PYTHONHASHSEED=3` 与 `PYTHONHASHSEED=54321` 各跑一次后 `diff -rq` | 两次产物全部字节一致 | 无 | 新增观测字段没有引入新的不确定性 |
| 2026-08-07 | T1 调研 | 读取现有 126/85 切分并按画板阈值重算 Top-N 相似度 | N5a 46、N5b 8、N5c 31；N5a Top-1 / 多数投票 29/46=63.04%；oracle 40/46=86.96% | 无 | 只读分析；证明现有 signature 分辨率不足 |
| 2026-08-07 | T1 | `python scripts/analyze_signature_resolution.py --feature-set legacy` | 完全复现基线：40 组 / 7 混合组 83 case / 空 signature 31 / N5a 46-N5b 8-N5c 31 / N5a top1 29 | `artifacts/t1_feature_dictionary_v1/legacy.json` | 度量脚本可信性验证；多数投票口径修正为 28/46=60.87%（见 9.5） |
| 2026-08-07 | T1 | `python scripts/sweep_feature_families.py --output artifacts/t1_feature_dictionary_v1/family_sweep.json` | 1023 个非空子集全部评估；满足全部 4 条 train 侧约束的只有 4 个 | `artifacts/t1_feature_dictionary_v1/family_sweep.json`（1.0 MB） | 选型只用训练集留一法，测试集全程未参与选型 |
| 2026-08-07 | T1 | `python scripts/analyze_signature_resolution.py --feature-set v1` | 混合标签覆盖 7.94%；N5a 21 条、桶内多数投票 16/21=76.19%；全测试集纯匹配 58/85=68.24%；`fiber` F1 首次非 0（0.20） | `artifacts/t1_feature_dictionary_v1/v1.json` | 两条量化验收全部达标 |
| 2026-08-07 | T1 | `python -m pytest -q` | 56 passed（原 46 + T1 锁定 10） | 无 | 新增 `tests/test_feature_dictionary.py` |
| 2026-08-07 | T1 | `python -m rca_framework.cli train-evaluate ... --output-dir artifacts/gate_t1 --backend none` | 58/85，accuracy 68.24%，`decision_status` 仍为 `agreement` 82 / `conflict_resolved_by_kg_rag_llm` 3，`label_leakage=false` | `artifacts/gate_t1/`（不入库） | 逐 case prediction、逐 case `anomaly_id`、`model.json` 与基线完全一致 |
| 2026-08-07 | T2 | 训练集 126 条上统计 bias / 温度 / 电压 / 收发功率 / SNR 分布，用于给约束定标 | bias=0 的 45 个 lane 与 tx 断光的 45 个 lane 完全重合；温度 39.35-60.13 °C 全部在 0-70 内；电压仅 1 例低于 3.135 V | 无 | 只读分析；结果写进约束的 `measured_evidence` 字段 |
| 2026-08-07 | T2 | `python scripts/render_constraint_skill.py --check` | SKILL.md 与约束库一致 | `skills/rca-constraints/SKILL.md` | 防止 prompt 文本与代码常量漂移 |
| 2026-08-07 | T2 | `python -m pytest -q` | 67 passed（T1 的 56 + T2 锁定 11） | 无 | 新增 `tests/test_constraint_library.py` |
| 2026-08-07 | T3 / T4 | `python -m pytest -q` | 97 passed | `artifacts/t4_threshold_calibration.json` | EvidencePack、证据图、检索和阈值标定门禁通过 |
| 2026-08-07 | T5 | `python -m pytest -q` | 117 passed | `artifacts/t5_routing_evaluation.json` | 两套路由、三分支、Wilson 标定和确定性排除门禁通过 |
| 2026-08-07 | T6 真机 | `python scripts/evaluate_routing.py --llm-backend vllm ...` | DeepSeek-R1-Distill-Qwen-32B 已跑通；`coverage-v2` 回答 59/85、答对 34、给结论时 57.63% | `artifacts/t6_llm_evaluation.json/.log` | 工程首测；缺正式 manifest、逐 case trace 与类别指标，不能作为论文正式结果 |
| 2026-08-09 | 进度审计 | `python -m pytest -q` | **162 passed** | 无 | 当前工作树全量测试通过；本文同步修正 T6 真机状态和实验缺口 |
| 2026-08-10 | T7 代码门禁 | `python -m pytest -q` | **169 passed** | 无 | 新增 M9、三分支 LLM 接线、正式实验产物回归测试；legacy 基线锁定全绿 |
| 2026-08-10 | T7 无模型评估 | `python scripts/evaluate_routing.py --llm-backend none --policies coverage-v2 --output-dir artifacts/t7_gate_none` | M9 最终出口 35/85，答对 28，给结论时 80.00%；补采 17、人工 33 | `artifacts/t7_gate_none/` | 正式四件套已生成；这是无 LLM 工程门禁，不代表 prompt v3 真机效果 |
| 2026-08-10 | T7 legacy 回归 | `python -m rca_framework.cli train-evaluate ... --output-dir /tmp/nsdi-agent-gate-t7 --backend none` | 58/85，accuracy 68.24%，fiber recall 0，rule overlap 0 | `/tmp/nsdi-agent-gate-t7/`（临时） | legacy 行为未漂移 |
| 2026-08-10 | T7 prompt v3 正式真机门禁 | `python scripts/evaluate_routing.py --llm-backend vllm ... --output-dir artifacts/t7_formal_deepseek32b_promptv3_seed42` | M9 最终回答 33/85、答对 26、给结论时 78.79%；补采 28、人工 24；fiber 0/6 | `artifacts/t7_formal_deepseek32b_promptv3_seed42/` | 正式 manifest/outcome/trace 齐全；模型与 seed=42 已记录 |
| 2026-08-10 | T10 规则 empirical study 代码门禁 | `python -m pytest -q` | **172 passed** | 无 | 新增固定三臂 prompt、配对统计、正式四件套与回归测试；legacy 锁定全绿 |
| 2026-08-10 | T10 规则 empirical study 真机 | `python scripts/run_rule_empirical_study.py --backend vllm --model-path ...DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --seed 42 --output-dir artifacts/rule_empirical_study_deepseek32b_seed42` | 证据-only 7/85；规则 prompt 8/85；prompt+checker 7/85；三组 fiber 均 0/6 | `artifacts/rule_empirical_study_deepseek32b_seed42/` | 同 85 case、同证据/模型/seed；规则净增 1 个正确 case，McNemar `p=1.0`；模型已退出，GPU 已释放 |

## 6. 基线数字与新框架实测事实

以下 legacy 数字属于旧双轨口径 + 旧 organized 60/40 数据集。合并清洗数据集到位后，必须重新建表，不得把旧数字当作新框架结果。

| 指标 | 数值 | 来源 |
| --- | --- | --- |
| deterministic baseline | 58/85，accuracy 68.24% | `artifacts/organized_rca_v2_60_40_seed42_baseline/evaluation_summary.json` |
| deterministic recall | L1 45.83%，L2 85.45%，fiber 0 | 同上 |
| deterministic decision_status | `agreement` 82，`conflict_resolved_by_kg_rag_llm` 3 | 同上 |
| DeepSeek-32B vLLM | 59/85，accuracy 69.41% | `artifacts/organized_rca_v2_60_40_seed42_deepseek32b_vllm/evaluation_summary.json` |
| DeepSeek-32B valid LLM outputs | 85/85 | 同上 |
| majority-class baseline | 55/85，accuracy 64.71% | 既有分析文档 |
| full-feature RandomForest 5-fold ceiling | 约 70.14%，fiber precision / recall / F1 均为 0 | 既有分析文档 |
| zero-anomaly test cases | 21/85 | 既有缺陷分析 |
| organized_data fiber 有效 case | 14，总训练 8 | 既有分析文档 |

本次重构比对新增的必须跟踪数字。`legacy` 列是旧 `anomaly_id` 口径，
`v1` 列是特征字典 v1 口径，两列同切分、同阈值、同检索内核，差异只来自特征集合。
逐项对照与口径说明见第 9.5 节。

| 指标 | legacy | 特征字典 v1 | 含义 |
| --- | --- | --- | --- |
| N4 分流分布（画板阈值） | N5a 46、N5b 8、N5c 31 | N5a 21、N5b 26、N5c 38 | 旧数据集 85 条测试 case |
| N5a Top-1 沿用历史结论 | 29/46，63.04% | 15/21，71.43% | legacy 低于 L2 多数类 64.71% |
| N5a 并列 case 多数投票 | 28/46，60.87% | 16/21，**76.19%** | T1 验收指标 |
| N5a oracle 上界 | 40/46，86.96% | 16/21，76.19% | v1 下并列集合几乎已单一标签，oracle 与多数投票重合 |
| N5a 桶纯净率 | 34.78% | 95.24% | 并列集合是否只有一种标签 |
| 训练集 signature 组数 | 40 组 / 126 case | 113 组 / 126 case | v1 下 82.54% 是 singleton |
| 混合标签 signature 覆盖 | 7 组覆盖 83 条（65.87%） | 3 组覆盖 10 条（**7.94%**） | T1 验收指标 |
| 空 signature 组 | 31 条，L2:20 / L1:10 / fiber:1 | 2 条，L1:1 / L2:1 | 零异常 case 基本消除 |
| N5b 桶内多数投票 | 3/8，37.50% | 16/26，61.54% | v1 下样本量从 8 升到 26 |
| 全测试集纯匹配多数投票 | 39/85，45.88% | 58/85，68.24% | 不含 KG / 规则 / LLM |
| `fiber` F1（纯匹配） | 0.0000 | 0.2000 | 本项目首次非 0，但仍只有 1/6 recall |

必做实验与正式门禁状态：

| 实验 | 当前状态 | 产物要求 / 当前缺口 |
| --- | --- | --- |
| 纯 LLM 无约束基线 | 单次正式配对已完成 | `artifacts/rule_empirical_study_deepseek32b_seed42/`：19/85 回答、7/85 正确、回答准确率 36.84%；仍缺重复运行 |
| 受约束 LLM | 单次正式配对已完成 | 同一产物：规则 prompt 15/85 回答、8/85 正确、回答准确率 53.33%；prompt+checker 14/85 回答、7/85 正确 |
| Top-N 消融 | 未开始 | 至少 N=1/3/5/10；当前 `evaluate_routing.py` 使用 `top_k=0`（保留全部候选） |
| 置信度阈值消融 | 部分完成 | T4 有相似度曲线、T5 有分支 Wilson 下界；缺接入真实 LLM 后的 coverage / selective risk / precision_at_coverage 曲线 |
| 训练集规模消融 | 受数据阻塞 | 50/100 可在当前 126 训练池做嵌套子集；150/200 必须等待合并数据集或重做切分设计 |
| 约束 / checker / SOP 消融 | 规则/checker 首轮已完成 | 已比较无约束、仅 prompt 约束、prompt+checker；SOP 到位后再加 SOP 组，当前配对差异不显著 |
| 重复性与统计检验 | 未开始 | 固定测试集、多训练子集 seed 或重复分层切分；报告置信区间，fiber 同时报绝对样本数 |

## 7. 待测数字与未解决问题

阶段 1 已经把其中五个问题从"未知"变成了可自动统计的数字，全部来自
`artifacts/*/evaluation_summary.json` 的 `observations` 与逐 case 的 `observation`，
并由 `tests/test_baseline_lock.py` 锁定：

| 问题 | 阶段 1 实测 | 落点 |
| --- | --- | --- |
| 82 条 deterministic `agreement` 中有多少是 `same_source_agreement` | **58 条同源一致，22 条根本没有 case 特异证据，只有 2 条是独立互证** | `evidence.aggregate_evidence` |
| `fiber` 的 28 条规则中有多少来自 `minority_fallback` | **0 条，全部是 `strict`；但 28 条的 `matched_training_cases` 全部恰好为 2，`confidence` 在 0.40-0.667，因此全部落在 `low_support`** | `rules.support_audit` |
| `prior_only == True` 的 case 数 | **22，与设计文档预估一致**；其中 21 条是零异常 case，另 1 条有异常但既无 KG 边也无规则命中 | `graph.query` 的 `prior_only` |
| 五档覆盖分布 | **`covered_pair` 47、`covered_singleton` 5、`covered_exemplar` 10、`partial` 1、`uncovered` 22** | `graph.classify_coverage` |
| 全类别匹配规则的支持度分布 | 逐 case 累计 `strong` 370、`moderate` 187、`low_support` 73；规则集层面 `L1` 3/19/9、`L2` 19/12/9、`fiber` 0/0/28 | `rules.support_tier` |

仍然未知，留给 T1-T12：

| 问题 | 当前状态 | 需要的落点 |
| --- | --- | --- |
| ~~lane 级 signature 触发数~~ | **已解答（T1）**：211 条中 102 条命中，`tx_ok_rx_down` 61 条最多；但触发未换来分类增量，家族标为 `candidate`。分布见 9.3 | 已闭环 |
| A / A' 阈值是否应固定为 100% / 70% | **必须重定（T1 升级）**：v1 特征空间下 `sim = 100%` 只剩 21/85，训练集 82.54% 是 singleton signature | T4 / T5，用 coverage-accuracy 曲线标定，见 9.6 第 1 条 |
| 绝对链路损耗能否作为 fiber 判据 | **已否定（T1）**：两个方向的 lane 均值损耗中位数均为负值（-0.285 / -0.227 dB），物理不可能，说明 lane 编号不对应或收发标定口径不同 | T2 约束库不得写绝对损耗门限，见 9.6 第 4 条 |
| 合并数据集后 `fiber` 有效样本数 | 未知 | 黄泽舜 / 王雅琪交付数据后重算 |
| 专家排障 SOP 是否按期到位 | 未知 | M6，黄泽舜 / 其桐文档输入 |
| N5b 样本量能否支撑消融 | 当前只有 8 条 | 合并数据集后重新统计 |
| 选择性分类在同等覆盖率下是否优于朴素置信度截断 | 未知 | T10，coverage-accuracy 曲线 |

### 7.1 阶段 1 观测结论

三个直接影响后续设计的结论：

1. **双路架构并没有提供两路证据。** 82 条 `agreement` 里只有 2 条来自互不相交的 anomaly。
   legacy 把 `agreement` 解释为"两条独立推理链结论一致"，在 `backend=none` 下这句话不成立：
   两路读的是同一批 anomaly，一致是结构性必然。因此 `agreement` 不能作为提高置信度的理由，
   `fuse_results` 在 `agreement` 时给的 `+0.1` 置信度加成没有证据支撑。
2. **22 条 case 的"候选分布"就是训练集类别先验。** 这 22 条的 `score_composition.prior_floor`
   精确等于 1.0，预测全部为 L2，凭先验命中 15 条。它们是低置信度降级或人工介入的首选人群。
3. **覆盖状态目前不带来判别力。** `covered_pair` 47 条准确率 68.09%，`uncovered` 22 条 68.18%，
   `covered_singleton` 5 条 60.00%。也就是说"KG 见过这个异常组合"并不意味着结论更可靠。
   这正是本次 N5a 63.04% 结果的预演，说明完全匹配分支必须先解决 signature 纯净度，而不能直接沿用历史标签。

## 8. 变更日志

| 日期 | 修改人 | 阶段 / 任务 | 文件 / 模块 | 变更摘要 | 验证结果 |
| --- | --- | --- | --- | --- | --- |
| 2026-08-06 | AI | 初始化 | 仓库复制、`AGENTS.md`、`Progress.md` | 从 `nsdi/` 复制轻量活动树；写入开发说明与进度台账 | 7 passed；58/85；逐 case prediction 一致 |
| 2026-08-06 | AI | 初始化 | `.gitignore`、git 仓库 | `nsdi-agent/` 建为独立 git 仓库并提交初始基线，只跟踪两个回归 artifacts；`/home/chenziang/.gitignore` 不再跟踪本目录 | 807 文件入库，`.git` 5.1M |
| 2026-08-06 | AI | 0 | `runtime.py`、`pipeline.py`、`cli.py`、`__init__.py` | 新增 `RuntimeConfig`；抽出 `CaseContext` / `build_case_context` / `finalize_prediction`；reasoner 缓存改用 `RuntimeConfig` 做键 | 15 passed；58/85；`predictions.json` 与基线字节一致 |
| 2026-08-06 | AI | 0 | `tests/test_baseline_lock.py` | 新增 8 个基线锁定测试：切分、summary、逐 case prediction / confidence / 两路分数 / anomaly_id、模型产物可加载且等价、规则互斥、`idf` 键序确定 | 15 passed |
| 2026-08-06 | AI | 0 | `graph.py` | 固定 `idf` 键序与 `retrieve` 的 IDF 求和顺序，消除集合迭代顺序带来的不可复现 | 不同 `PYTHONHASHSEED` 下产物字节一致；58/85 不变 |
| 2026-08-06 | AI | 1 | `types.py` | 新增 `DECISIONS`、`SUFFICIENCY`、`EVIDENCE_SOURCES`、`EvidenceItem`、`Verdict`，全部只增不改 | 46 passed |
| 2026-08-06 | AI | 1 | `retrieval.py`、`graph.py` | IDF 加权 Jaccard 检索搬到 `retrieval.py` 并支持 `hide_labels`；新增五档 `classify_coverage` 与 `evidence_items` | legacy 检索结果逐行一致；`scores` 不变 |
| 2026-08-06 | AI | 1 | `rules.py` | 新增 `SUPPORT_TIERS` 与 `support_tier`、`evidence_items`、`support_audit` | `rule_overlap` 仍为 0；match 分数不变 |
| 2026-08-06 | AI | 1 | `evidence.py` | 新增 `EvidenceView` 与 `aggregate_evidence`，区分 `independent_agreement` / `same_source_agreement` / `conflict` / `no_evidence` | 同源、独立、冲突、无证据四条路径均有单测 |
| 2026-08-06 | AI | 1 | `pipeline.py` | `CaseContext` 增加 `coverage` 与 `evidence_view`，逐 case 输出新增 `observation`，`evaluate` 的 summary 新增 `observations` 汇总 | legacy 键逐值一致 |
| 2026-08-06 | AI | 1 | `tests/` | 新增 26 个观测层单测；基线锁定测试增加"只增不改"递归比对与阶段 1 观测数字锁定 | 46 passed；58/85 不变 |
| 2026-08-07 | AI | 文档切换 | `AGENTS.md`、`Progress.md` | 按最新会议方案从旧阶段 0-5 切到证据图 Agentic AI / T1-T12；legacy 58/85 降为回归锚点；写入 N4 分流与 N5a 63.04% 风险数字 | 文档更新；未运行测试 |
| 2026-08-07 | AI | T1 | `scripts/analyze_signature_resolution.py` | 新增 signature 分辨率与 N4 分流度量脚本，支持按 feature-set 切换口径 | 完全复现 legacy 基线的 40 组 / 83 混合 case / 46-8-31 分流 |
| 2026-08-07 | AI | T1 | `scripts/sweep_feature_families.py` | 新增家族消融脚本，枚举 1023 个非空子集，同时报告训练集留一法与留出测试集两套指标 | 84 秒跑完；选型只用 train 列 |
| 2026-08-07 | AI | T1 | `rca_framework/features/dictionary.py` | 新增特征字典 v1：10 个家族的维度 / 物理含义 / 单位 / 取值域 / 抽取规则 / token 模板 / 准入状态；4 个入选 v1，6 个标 `candidate` 并写明未入选理由 | `content_hash=1b2e66ed650ce60e`，可 JSON 序列化 |
| 2026-08-07 | AI | T1 | `rca_framework/features/extractor.py` | 新增 10 个家族抽取器与 `FeatureModel`（分位边界只在训练集上拟合并可落盘） | 抽取确定；带标签与不带标签 signature 一致 |
| 2026-08-07 | AI | T1 | `tests/test_feature_dictionary.py` | 新增 10 个 T1 锁定测试：字典冻结、家族语义完整、token 前缀合规、抽取确定性、标签无关性、不改写 legacy `anomaly_id`、验收数字锁定 | 56 passed；58/85 不变；逐 case prediction 与 `anomaly_id` 与基线一致 |
| 2026-08-07 | AI | T2 | `rca_framework/constraints/library.py` | 新增物理约束库 v1：14 条约束，每条带 `kind`（排除 / 禁止推断 / 恒等 / 倾向）、`provenance`（规范 / 实测 / 推导）、实测证据、诊断用法、prompt 文本、审核状态 | `content_hash=ee95eddd7885abdf`；覆盖 T2 要求的 5 类物理量 |
| 2026-08-07 | AI | T2 | `scripts/render_constraint_skill.py`、`skills/rca-constraints/SKILL.md` | SKILL.md 改为从约束库自动渲染，支持 `--check` 做一致性校验 | prompt 文本与代码常量单点定义 |
| 2026-08-07 | AI | T2 | `tests/test_constraint_library.py` | 新增 11 个 T2 锁定测试：库冻结、必需类别覆盖、实测参数必须带数字、审核状态如实标注、prompt 顺序、类别过滤、禁止出现绝对损耗门限、SKILL.md 同步 | 67 passed |
| 2026-08-07 | AI | T3 / T4 | `evidence_pack.py`、`features/extractor.py`、`evidence_graph/`、标定脚本与测试 | 完成标签隔离的 EvidencePack、case-token 证据图、Top-N 差集/冲突结构和 N4 标定 | 97 passed；图版本 `evidence-graph-v1:126:5e10b5b25d559777` |
| 2026-08-07 | AI | T5 | `branches/`、`evidence_graph/router.py`、`scripts/evaluate_routing.py` | 完成两套路由、N5a/b/c 处理器、N6 弃权原语、Wilson 标定；新增 C15 并升级约束库 v2 | 117 passed；确定性排除 8 次触发、0 次排掉真标签 |
| 2026-08-07 | AI | T6 | `constraints/checker.py`、`llm/`、真机脚本与产物 | 完成结构化推理、校验/重写循环并运行 DeepSeek-R1-Distill-Qwen-32B/vLLM 首测 | checker + LLM reasoning 共 45 个测试；首测结果见 9.25 |
| 2026-08-09 | AI | 进度审计 | `Progress.md`、`Validation.md`、架构审计 Canvas | 对齐当前代码/产物，修正 T6 真机状态、模块表、实验缺口和门禁记录 | 全量 162 passed；Canvas TypeScript 无错误 |
| 2026-08-10 | AI | T7 | `decision.py`、`branches/`、`llm/prompts.py`、`evaluate_routing.py`、测试与正式产物 | 接通三分支 LLM 仲裁；新增 LLM 独立标定、Wilson 统一出口、branch-aware prompt v3、manifest/outcome/trace/分层指标/选择性风险曲线 | 169 passed；无模型最终出口 35/85、28 对、precision 80.00%；legacy 58/85 不变 |

## 9. 分步产物与原始数据汇总

本节按 T1-T12 逐步累加。每完成一步，写入该步产生的**原始数据**、产物路径和可复核命令，
不写结论性形容词。数字来自当前工作树中的可重跑脚本；截至 2026-08-09，
T1-T6 主体尚未提交到 git，不能把“本地存在”误写成“已入库”。

### 9.1 T1 产物清单（2026-08-07）

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 代码 | `rca_framework/features/dictionary.py` | 特征字典 v1 声明，10 个家族 |
| 代码 | `rca_framework/features/extractor.py` | 10 个家族抽取器 + `FeatureModel` 分位拟合 |
| 代码 | `scripts/analyze_signature_resolution.py` | 分辨率 / N4 分流度量，支持 `--feature-set` 切换 |
| 代码 | `scripts/sweep_feature_families.py` | 1023 子集家族消融，train-LOO 与 held-out 双列 |
| 测试 | `tests/test_feature_dictionary.py` | 10 个锁定测试 |
| 数据 | `artifacts/t1_feature_dictionary_v1/dictionary_v1.json` | 冻结的 v1 字典，`content_hash=1b2e66ed650ce60e` |
| 数据 | `artifacts/t1_feature_dictionary_v1/dictionary_all_families.json` | 含 candidate 家族的完整声明，`content_hash=006c4c01c766055d` |
| 数据 | `artifacts/t1_feature_dictionary_v1/feature_model_v1.json` | 训练集拟合的分位边界，`fitted_case_count=126` |
| 数据 | `artifacts/t1_feature_dictionary_v1/legacy.json` | legacy 口径逐 case 分流明细 |
| 数据 | `artifacts/t1_feature_dictionary_v1/v1.json` | v1 口径逐 case 分流明细 |
| 数据 | `artifacts/t1_feature_dictionary_v1/family_sweep.json` | 1023 个子集的完整消融结果，1.0 MB |

复核命令：

```bash
python scripts/analyze_signature_resolution.py --feature-set legacy
python scripts/analyze_signature_resolution.py --feature-set v1
python scripts/sweep_feature_families.py
python -m pytest tests/test_feature_dictionary.py -q
```

### 9.2 特征字典 v1 冻结内容

版本 `feature-dictionary-v1`，`content_hash=1b2e66ed650ce60e`，入选 4 个家族。
`FeatureModel` 的分位边界只在 126 条训练 case 上拟合，`tail_quantiles=(0.25, 0.75)`。

| 家族 | token 模板 | 物理含义 | 单位 | 取值域 | 211 条中触发 case 数 | 不同 token 数 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `signal_drop` | `drop:{side}:{metric}:{bucket}` | 该侧该指标有多少条 lane 掉到断光哨兵 | lane 计数 | `single_lane` / `partial_lanes` / `all_lanes` | 122 | 21 |
| `status_fault` | `status:{side}:{status_key}` | 模块自报的失光 / 失锁标志位 | 布尔 | `TxLOS` / `TxLOL` / `RxLOS` / `RxLOL` | 116 | 7 |
| `lane_imbalance` | `imbalance:{side}:{metric}` | 同端口内健康 lane 极差超出训练集上界 | dB | `over_learned_spread` | 67 | 5 |
| `level_tail` | `level:{side}:{statistic}:{bucket}` | 收 / 发光功率与介质侧 SNR 的绝对水平分档 | dBm / dB | `low_tail` / `high_tail` | 201 | 12 |

拟合出的分位边界（`feature_model_v1.json` 原文）：

| 统计量 | L1 下 / 上分位 | L2 下 / 上分位 |
| --- | --- | --- |
| `rxpower_mean` | 0.2625 / 1.78 dBm | 0.228125 / 1.57875 dBm |
| `txpower_mean` | 0.3725 / 1.07875 dBm | 0.456875 / 1.0725 dBm |
| `media_snr_min` | 24.715 / 25.64 dB | 25.06 / 25.96 dB |

v1 signature 长度分布（126 条训练 case）：

| token 数 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case 数 | 2 | 6 | 10 | 21 | 15 | 15 | 25 | 13 | 7 | 4 | 1 | 3 | 3 | 1 |

### 9.3 未入选的 6 个 candidate 家族及其实测触发数

这 6 个家族代码保留在 `FULL_DICTIONARY` 中，只用于消融，不进 v1 signature。
未入选理由逐条写在 `dictionary.py` 的 `selection_note` 字段。

| 家族 | 211 条中触发 case 数 | 不同 token 数 | 未入选的实测依据 |
| --- | ---: | ---: | --- |
| `fence_outlier` | 141 | 10 | 3 倍 IQR 围栏与 `level_tail` 高度重合，未进入任何一个满足 train 侧约束的子集 |
| `lane_direction` | 102 | 8 | 加入 v1 组合后 train-LOO 的 N5a 桶内准确率与全集准确率同时下降 |
| `side_asymmetry` | 114 | 6 | 与 `level_tail` 同源冗余，加入后 N5a 桶塌到 10 条以下 |
| `port_width` | 211 | 8 | 恒定产出，把每个 case 推向唯一 signature，自身判别力接近类别先验 |
| `alarm_kind` | 211 | 3 | 只有 3 个取值，每个取值的标签分布都接近全局先验 |
| `telemetry_gap` | 211 | 2 | 恒定产出且只有 2 个实际取值，进 signature 会稀释相似度 |

`lane_direction` 的 signature 触发分布（全量 211 条，来自 `anomaly.lane_loss_report`）：

| 触发组合 | case 数 | 标签分布 |
| --- | ---: | --- |
| 无 signature | 109 | L2 75 / L1 31 / fiber 3 |
| `tx_ok_rx_down` | 61 | L2 40 / L1 13 / fiber 8 |
| `single_lane_outlier` | 27 | L2 14 / L1 12 / fiber 1 |
| `bidirectional_same_lane` + `tx_down` | 6 | L2 4 / fiber 2 |
| `tx_down` | 5 | L2 3 / L1 2 |
| `bidirectional_same_lane` + `tx_down` + `tx_ok_rx_down` | 3 | L2 2 / L1 1 |

这条记录同时回答了第 7 节遗留问题「lane 级 signature 触发数未知」：
lane 级信号确实会触发（211 条中 102 条），legacy `directional_loss` 从未触发确实是过滤器导致的，
但触发本身没有换来分类增量。

### 9.4 T1 选型规则与候选集

选型规则在看测试集之前固定，且四条约束全部只用训练集（126 条，留一法）：

1. 训练集混合标签 signature 覆盖率 <= 10%（基线 65.87%）。
2. 训练集留一法下 N5a 桶至少 20 条，否则分支结论没有统计意义。
3. 训练集留一法下 N5a 桶内多数投票准确率 > 64.71%（L2 多数类基线）。
4. 训练集留一法下全集多数投票准确率 >= 65.87%（训练集 L2 先验）。

1023 个非空子集中只有 4 个同时满足，按 N5a 桶内准确率排序：

| 排名 | 家族组合 | 混合覆盖 | train-LOO N5a | train-LOO N5a 准确率 | train-LOO 全集 | held-out N5a | held-out N5a 准确率 | held-out 全集 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `signal_drop`+`status_fault`+`lane_imbalance`+`level_tail` | 7.94% | 20 | 80.00% | 66.67% | 21 | 76.19% | 68.24% |
| 2 | `status_fault`+`lane_direction`+`level_tail`+`alarm_kind` | 7.94% | 24 | 75.00% | 66.67% | 19 | 68.42% | 51.76% |
| 3 | `status_fault`+`lane_imbalance`+`lane_direction`+`level_tail`+`alarm_kind` | 7.94% | 22 | 72.73% | 68.25% | 19 | 68.42% | 52.94% |
| 4 | `status_fault`+`lane_imbalance`+`lane_direction`+`level_tail`+`side_asymmetry`+`alarm_kind` | 7.94% | 20 | 70.00% | 67.46% | 9 | 88.89% | 56.47% |

第 1 名同时是 4 个候选里 held-out 表现最好的，train 与 test 两列一致，因此冻结为 v1。

消融里必须记住的反面结论：**按全集准确率排序会选出最粗的特征集合**。
例如 `status_fault`+`lane_imbalance`+`telemetry_gap` 的 train-LOO 全集准确率是 77.78%、
held-out 是 68.24%，但它的混合标签覆盖率高达 75.40%、N5a 桶有 114 条——
它只是在复现 L2 多数类先验，不是在做根因判别。所以 T1 的选型目标不能是全集准确率。

### 9.5 legacy 与 v1 的逐项对照（126 训练 / 85 测试，画板阈值 100% / 70%）

| 指标 | legacy `anomaly_id` | 特征字典 v1 | 变化 |
| --- | ---: | ---: | --- |
| 不同 feature token 数 | 32 | 40 | +8 |
| 训练集不同 signature 组数 | 40 | 113 | +73 |
| 混合标签 signature 组数 | 7 | 3 | -4 |
| 混合标签覆盖 case 数 | 83 / 126（65.87%） | 10 / 126（**7.94%**） | **验收 1 达标** |
| 空 signature case 数 | 31（L2 20 / L1 10 / fiber 1） | 2（L1 1 / L2 1） | -29 |
| N4 分流 N5a / N5b / N5c | 46 / 8 / 31 | 21 / 26 / 38 | N5a 缩小 |
| N5a 桶内多数投票 | 28/46 = 60.87% | 16/21 = **76.19%** | **验收 2 达标** |
| N5a Top-1 | 29/46 = 63.04% | 15/21 = 71.43% | +8.39pt |
| N5a 桶纯净率（并列集合单一标签） | 34.78% | 95.24% | +60.46pt |
| N5b 桶内多数投票 | 3/8 = 37.50% | 16/26 = 61.54% | +24.04pt |
| N5c 桶内多数投票 | 8/31 = 25.81% | 26/38 = 68.42% | +42.61pt |
| 全测试集纯匹配多数投票 | 39/85 = 45.88% | 58/85 = **68.24%** | +22.36pt |
| `L1` precision / recall / F1 | 0.4286 / 0.3750 / 0.4000 | 0.5000 / 0.5833 / 0.5385 | 全面提升 |
| `L2` precision / recall / F1 | 0.7500 / 0.5455 / 0.6316 | 0.8113 / 0.7818 / 0.7963 | 全面提升 |
| `fiber` precision / recall / F1 | 0.0000 / 0.0000 / 0.0000 | 0.2500 / 0.1667 / **0.2000** | 本项目首次非 0 |

口径说明：legacy 的 45.88% 低于第 6 节记录的 63.04%，因为这里统计的是**纯历史匹配**
在全部 85 条上的多数投票，legacy 有 22 条 case 在训练集里找不到任何非零相似度的邻居，
计为未命中；第 6 节的 63.04% 只统计 N5a 这 46 条。另外第 6 节记录的
「N5a 并列 case 多数投票 29/46」在本次统一并列打破规则（按 `ROOT_CAUSES` 顺序取最小）
下重算为 28/46 = 60.87%，Top-1 仍为 29/46 = 63.04%。后续一律以脚本口径为准。

### 9.6 T1 暴露的新问题

1. **画板的 `sim = 100%` 阈值在 v1 特征空间下不再成立。** 分辨率提高的直接代价是完全匹配从
   46 条掉到 21 条，训练集 82.54% 的 case 变成 singleton signature。这不是实现缺陷而是结构性权衡：
   特征越细，「见过一模一样的 case」就越罕见。第 7 节「A / A' 是否固定为 100% / 70%」
   因此从「待验证」升级为「必须重定」，交 T4 / T5 用 coverage-accuracy 曲线标定。
2. **`fiber` 仍然是主要缺口。** v1 把 `fiber` F1 从 0 提到 0.20，但绝对值仍是 1/6 recall。
   6 条 fiber 测试 case 中，1 条落 N5a（相似度 1.0 但命中的是 L2 组）、3 条落 N5b、2 条落 N5c，
   唯一判对的那条落在 N5c。不能据此声称 fiber 问题已解决。
3. **v1 仅剩的 3 个混合标签组已经指向数据本身的极限**：
   - `n=5`，L2 4 / fiber 1，signature 含 `status:L1:RxLOS` + `drop:L1:rxpower:single_lane` + `level:L2:rxpower_mean:low_tail`；
   - `n=3`，L2 2 / L1 1，signature 只有三个 `high_tail`；
   - `n=2`，L1 1 / L2 1，空 signature。
   继续加特征只会把它们打散成 singleton，而不会真正提高判别力。
4. **两端「光损耗」在本数据集上恒为负值。** `L1_to_L2` 方向的 lane 均值损耗中位数为 -0.285 dB、
   `L2_to_L1` 为 -0.227 dB，即对端收到的功率高于本端发出的功率。这在无源链路上物理不可能，
   说明两端 lane 编号不对应或收发标定口径不同。结论是**不能用绝对链路损耗做 fiber 判据**，
   只能用相对量与分档量。这条直接约束 T2 的物理约束库写法。

### 9.7 T2 产物清单（2026-08-07）

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 代码 | `rca_framework/constraints/library.py` | 物理约束库 v1，14 条 |
| 代码 | `scripts/render_constraint_skill.py` | 从约束库渲染 SKILL.md，支持 `--check` |
| 文档 | `skills/rca-constraints/SKILL.md` | prompt 化约束说明，自动生成，禁止手工编辑 |
| 测试 | `tests/test_constraint_library.py` | 11 个锁定测试 |

复核命令：

```bash
python scripts/render_constraint_skill.py --check
python -m pytest tests/test_constraint_library.py -q
python -c "from rca_framework.constraints import render_prompt_block; print(render_prompt_block())"
```

版本 `constraint-library-v1`，`content_hash=ee95eddd7885abdf`。
`measured` 类参数的实测口径是 `organized_rca_v2_stratified_60_40_seed42` 训练集 126 条。

### 9.8 约束库 v1 的 14 条

按 `kind` 分类：`exclusion` 3 条、`caveat` 3 条、`invariant` 3 条、`indicator` 5 条。
按参数来源：`device_spec` 5 条、`measured` 7 条、`derived` 2 条。
按类别：`bias_current` 2、`temperature` 1、`voltage` 1、`tx_power` 2、`rx_power` 1、
`signal_quality` 1、`lane_directional_consistency` 3、`measurement_validity` 3。
T2 验收要求的 5 类物理量全部覆盖。

| ID | 类别 | 类型 | 断言 | 参数与实测依据 |
| --- | --- | --- | --- | --- |
| `C1_bias_zero_means_laser_off` | 电流 | invariant | 偏置电流 0 mA 等价于该 lane 未发光 | 训练集 1008 个 lane 读数中 45 个 bias=0，这 45 个 lane 的 txpower **全部**同时断光，双向无例外 |
| `C2_bias_healthy_band` | 电流 | indicator | 健康 lane 偏置电流 7.2-7.8 mA | 非零电流 p25=7.22/7.29 mA，p99=7.72/7.78 mA |
| `C3_temperature_operating_range` | 温度 | exclusion | 0-70 °C 内排除热致故障 | 252 个读数全落在 39.35-60.13 °C；L2 侧中位数比 L1 高 3.58 °C，是端口形态差异 |
| `C4_voltage_nominal_band` | 电压 | exclusion | 3.3 V ±5%（3.135-3.465 V）内排除供电异常 | 252 个读数中仅 1 例越下界（3.10 V），不足以建统计规则 |
| `C5_tx_power_range` | 发光功率 | invariant | 发送功率要么在 -1.8~2.1 dBm，要么掉到 -39 dBm 哨兵，无中间态 | 963 个健康读数，L1 -1.70~1.91、L2 -1.73~2.08 dBm |
| `C6_tx_down_excludes_medium` | 发光功率 | exclusion | 本端未发光则该方向排除 fiber | 由 C1 + C5 推导，不引入新参数 |
| `C7_rx_power_range` | 收光功率 | invariant | 接收功率存在连续劣化区间 -12.3~3.0 dBm | 929 个健康读数，p1 为 -8.45（L1）/ -4.68（L2）dBm |
| `C8_tx_ok_rx_down_indicates_medium` | 同 lane 方向性 | indicator | 本端发光正常而对端同 lane 无光，指向介质或对端接收 | 211 条中 61 条命中，fiber 占 8/61=13.1%，是全局 6.6% 的约 2 倍；但同时命中 53 条非 fiber |
| `C9_bidirectional_symmetry` | 同 lane 方向性 | indicator | 双向对称异常指向介质，单向异常指向该方向端点 | 同 lane 双向异常仅 9 条（L2 6 / fiber 2 / L1 1），样本不足，依据是物理拓扑而非数据 |
| `C10_all_lanes_vs_single_lane` | 同 lane 方向性 | indicator | 全 lane 异常指向端口级，单 lane 异常指向通道级 | 这是 `signal_drop` 分三档的物理依据，T1 消融显示该分档不可替代 |
| `C11_media_snr_floor` | 信号质量 | indicator | 收光正常但 SNR < 22.5 dB 指向链路质量 | 健康 media_snr 的 p1 为 22.47/22.95 dB；低于 20 dB 的读数只有 4 个 |
| `C12_no_absolute_link_loss` | 量测有效性 | caveat | 禁止用两端功率相减求链路损耗 | L1→L2 均值损耗中位数 -0.285 dB、L2→L1 -0.227 dB，负损耗物理不可能 |
| `C13_serdes_snr_unit_unknown` | 量测有效性 | caveat | `serdes_snr` 非 dB 量纲，只能作有效/失效二值 | 972 个读数健康值 6.6e5-8.2e5，失效时为 1 |
| `C14_host_snr_mostly_missing` | 量测有效性 | caveat | `host_snr` 大面积缺失，缺失不等于正常 | 126 条训练 case 中只有 34 条任一侧有读数 |

### 9.9 T2 的两条设计决定与遗留

1. **prompt 文本与代码常量单点定义。** `skills/rca-constraints/SKILL.md` 由
   `scripts/render_constraint_skill.py` 从 `library.py` 渲染，
   `tests/test_constraint_library.py::test_skill_file_is_generated_from_library` 重新渲染并逐字节比对。
   改了门限但忘了改 prompt，测试会直接失败。这是 AGENTS.md「不复制散落门限」的落地方式。
2. **prompt 注入顺序固定为 exclusion → caveat → invariant → indicator。**
   先给能排除的，再给不许推的，最后才给提高可能性的。
   理由是阶段 1 已经证明这套数据里「倾向性证据」很容易把 LLM 推向多数类；
   先做排除可以在加权之前砍掉不可能的选项。`render_prompt_block()` 强制这个顺序，有测试锁定。
3. **遗留：14 条全部是 `pending_expert_review`。** 渲染进 prompt 时会带「（待专家审核）」后缀，
   这一点也有测试锁定，防止未审约束被当成已确认事实。
   `C12`、`C13` 还需要向厂商确认两端 lane 编号对应关系与 `serdes_snr` 量纲，
   这两项不是代码问题，无法在本仓库内闭环（已登记为 `Validation.md` V7 / V8）。

### 9.10 T3 产物清单（2026-08-07）

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 证据包类型 | `rca_framework/evidence_pack.py` | `EvidencePack`（`evidence-pack-v1`）、`MetricReading`、`build_packs`、`labels_of` |
| 冲突检测 | `rca_framework/features/extractor.py` | `MUTUALLY_EXCLUSIVE_PREFIXES`、`detect_token_conflicts` |
| 单测 | `tests/test_evidence_pack.py` | 13 个，覆盖正常 / 缺失 / 冲突 / lane 级四条路径 |

T3 改的是接口契约，不是算法。改完之后 `analyze_signature_resolution.py` 的全部数字逐位不变
（v1：113 组 / 混合 7.94% / N4 21-26-38；legacy：N4 46-8-31），这是接口重构没有副作用的证明。

三个设计决定：

1. **标签隔离从「约定」升级为「结构」。** `EvidencePack.from_case` 是唯一构造入口，
   它在构造时摘掉 `label`，因此下游对象里根本没有标签字段可读。
   `fit_feature_model` 与 `extract_features` 的入参都换成了 `EvidencePack`，
   「拟合 / 抽取时忘了摘标签」这类泄漏在类型层面就不可能发生，不再依赖调用方自觉。
2. **「没有异常」与「没有数据」在结构上分开。** `telemetry_status` 取
   `no_telemetry` / `partial_telemetry` / `full_telemetry`，与 `tokens` 是否为空正交。
   阶段 1 的结论 2 是这两种情况在 legacy 里被混成同一个空 anomaly 集合；现在
   `CaseFeatures.is_empty` 必须配合 `telemetry_status` 才能解释，测试
   `test_empty_signature_is_not_the_same_as_no_telemetry` 锁定了这一点。
   实测：126 条训练 case 里 2 条空 signature，两条都是 `partial_telemetry`，
   即「采到数了并且一切正常」，而不是「什么都没采到」。
3. **冲突规则放在抽取器而不是字典里。** 互斥分档（`drop:` 与 `level:` 前缀下同一维度只能有一个分档）
   描述的是结构性事实，不引入新门限也不引入新维度，因此不改动特征字典 v1 的内容指纹
   `1b2e66ed650ce60e`，T1 的冻结依然成立。`fence_outlier` 故意不列入互斥：
   同一侧不同 lane 一个偏低一个偏高是合法观测，不是冲突。

实测：211 条真实 case 的特征抽取**零冲突**，说明抽取规则自洽。
冲突这个结构真正发挥作用的地方是 T4 的跨 case 比较，见第 9.13 节。

### 9.11 证据包记录的字段完整性（126 训练 case）

`expected_field_count = 18`（2 侧 × (5 指标 + 4 状态位)）。

| 观测 | 数值 | 影响 |
| --- | --- | --- |
| `L1.host_snr` / `L2.host_snr` 缺失 | 126 条中 92 条两侧全缺 | 已登记 `Validation.md` V9；N6 按「未采集」处理 |
| 空 signature 但有遥测 | 2 条 | 属于 `partial_telemetry`，不等于零证据 |
| 完全无遥测 | 0 条 | 本数据集不存在，但 `no_telemetry` 分支有单测覆盖 |

### 9.12 T4 产物清单（2026-08-07）

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 证据图存储 | `rca_framework/evidence_graph/store.py` | `EvidenceGraph`、`GraphCase`、倒排索引、纯净度报告、IDF、版本与持久化 |
| Top-N 检索 | `rca_framework/evidence_graph/match.py` | `match` / `match_many`、`Candidate`、`MatchResult`、`weighted_jaccard`、`find_conflicts` |
| 阈值标定脚本 | `scripts/calibrate_router_thresholds.py` | coverage-accuracy 曲线，train-LOO 选阈值、held-out 事后核对 |
| 标定产物 | `artifacts/t4_threshold_calibration.json` | 16 个候选阈值 × 2 个切分的完整曲线 |
| 单测 | `tests/test_evidence_graph.py` | 17 个 |

证据图版本：`evidence-graph-v1:126:5e10b5b25d559777`。
版本号由图内容、特征字典指纹、`FeatureModel` 指纹三者共同决定，任何一项变了历史匹配结果就不可比。

| 指标 | 数值 |
| --- | --- |
| case 节点 | 126（L1 35 / L2 83 / fiber 8） |
| token 节点 | 40 |
| signature 分组 | 113，其中混合标签 3 组覆盖 10 条（7.94%） |
| 单例组 | 104（82.54%） |
| 空 signature 组 | 2 |
| 测试集 N4 分布（画板阈值） | N5a 21 / N5b 26 / N5c 38 |
| 测试集多数投票 | 58/85 = 68.24% |

N4 分布与多数投票逐位复现第 9.5 节的 T1 数字，说明证据图路径与 T1 的分析口径一致，
两阶段的产物可以直接对照。legacy 回归锚点 58/85 同时复测通过，全量 97 个测试通过。

与 legacy `retrieval.retrieve` 的三个差别：

1. **索引建在 v1 token 上而不是 `anomaly_id` 上。** 相似度内核（IDF 加权 Jaccard）完全一致。
2. **返回结构把差集拆成三类。** legacy 只返回相似度和重叠 token，
   回答不了 N5b 的「还缺什么证据」和 N6 的「够不够判」。现在拆成
   `missing_evidence`（候选有我没有，即补采清单）、`extra_evidence`（我有候选没有，
   说明比历史更严重或场景不同）、`conflicting_evidence`（同一维度取了互斥分档）。
   并列候选的补采清单取**交集**，只让人去补每个候选都要求的证据。
3. **空 signature 之间的相似度定义为 0 而不是 1。** 否则零证据 case 会互相 100% 命中
   并填满 N5a，而它们恰恰是最该走人工介入的那批（阶段 1 结论 2）。有单测锁定。

### 9.13 证据覆盖率与冲突的实测分布（85 条测试集）

| 指标 | 数值 |
| --- | --- |
| 证据覆盖率均值 | 0.8485 |
| 有补采清单的 case | 34 / 85 |
| 与最佳候选存在冲突证据的 case | 7 / 85 |
| 高相似度（>= 0.7）但覆盖率 < 0.7 的 case | 1（`case_9a6532971c2a`） |

`case_9a6532971c2a` 是这个结构的价值示例：它与历史 case 相似度 0.723，
按画板规则会被判进 N5b 直接复用历史结论，但它的证据覆盖率只有 0.67——
`level:L2:media_snr_min:low_tail` 和 `level:L2:rxpower_mean:low_tail` 两条证据历史上没见过。
只看相似度看不出这一点。

### 9.14 N4 阈值标定结果（这是 T4 最重要的产物）

标定方法：训练集留一法选阈值，留出测试集只做事后核对，不参与选择。
命令：`python scripts/calibrate_router_thresholds.py --output artifacts/t4_threshold_calibration.json`。
训练集类别先验 65.87%，阈上准确率必须显著高于它才有意义。

**结论一：`sim = 1.0` 这条线保留，不下调。**

累计曲线看起来 0.90 比 1.0 更好（train-LOO 准确率 81.82% vs 80.00%，覆盖率还略高），
但拆成非累计分档后，`[0.9, 1.0)` 这个区间在训练集只有 2 条、测试集 0 条。
所谓「0.90 更优」完全建立在 2 个样本上，不足以支撑改动。

**结论二：`sim >= 0.7` 这条线没有数据支持，应当废弃。**

非累计分档（训练集留一法 / 留出测试集）：

| 相似度区间 | train-LOO n | train-LOO 准确率 | held-out n | held-out 准确率 |
| --- | ---: | ---: | ---: | ---: |
| `[1.0, 1.0]` | 20 | 80.00% | 21 | 76.19% |
| `[0.9, 1.0)` | 2 | 100.00% | 0 | — |
| `[0.8, 0.9)` | 28 | 64.29% | 16 | 62.50% |
| `[0.7, 0.8)` | 17 | 70.59% | 10 | 60.00% |
| `[0.5, 0.7)` | 48 | 64.58% | 29 | 68.97% |
| `(0, 0.5)` | 9 | 55.56% | 9 | 66.67% |
| `0`（无任何非零邻居） | 2 | 0.00% | 0 | — |

`sim = 1.0` 之下、`sim > 0` 之上的所有区间，准确率都在 60-70% 之间来回摆，
与 65.87% 的类别先验没有区别，两个切分上的排序也不一致。
也就是说**在 v1 特征空间里，相似度一旦低于 1.0，它的具体数值就不再携带准确率信息**。
拿这样一个量去切 N5b 和 N5c，切出来的边界是噪声。

**结论三：N5b 的入口条件改用「证据全覆盖」。**

`evidence_coverage = 1.0` 的含义是「历史上存在一个 case，包含了我当前的全部证据」，
即我身上没有任何一条历史没见过的现象。它是单向的（相似度是对称的），
语义上正好对应 N5b 想问的问题。

| 分档条件 | train-LOO n | train-LOO 准确率 | held-out n | held-out 准确率 |
| --- | ---: | ---: | ---: | ---: |
| `coverage = 1.0` | 61 | 73.77% | 39 | 79.49% |
| `coverage ∈ [0.8, 1.0)` | 28 | 53.57% | 18 | 55.56% |
| `coverage ∈ [0.5, 0.8)` | 35 | 68.57% | 28 | 60.71% |
| `coverage < 0.5` | 2 | 0.00% | 0 | — |

`coverage = 1.0` 这一档在两个切分上都明显高于先验（+7.9 / +11.3 个百分点），
方向一致，样本量也够（61 / 39 条）。这是唯一一个能复现的分档信号。

**两套规则的头对头对比**（`N5a + N5b` 合并即「不必走 LLM 的高置信出口」）：

| 切分 | 规则 | N5b 准确率 | 高置信出口覆盖率 | 高置信出口准确率 |
| --- | --- | ---: | ---: | ---: |
| train-LOO | 画板 100% / 70% | 32/47 = 68.09% | 53.17% | 48/67 = 71.64% |
| train-LOO | sim=1 / 证据全覆盖 | 29/41 = 70.73% | 48.41% | 45/61 = **73.77%** |
| held-out | 画板 100% / 70% | 16/26 = 61.54% | 55.29% | 32/47 = 68.09% |
| held-out | sim=1 / 证据全覆盖 | 15/18 = 83.33% | 45.88% | 31/39 = **79.49%** |

新规则在两个切分上都提高了高置信出口的准确率，代价是覆盖率降约 5-9 个百分点，
即多约 6-8 条 case 要走 N5c。测试集上的提升（+11.4 个百分点）明显大于训练集上的（+2.1），
样本量只有 39-67 条，因此**提升的幅度不可信，但提升的方向在两个切分上一致**。
建议采纳新规则，同时在合并数据集到位后（`Validation.md` V10）重新验证幅度。

**结论四：`sim = 0` 的 case 必须直接进 N6，不进 N5c。**

训练集 2 条零证据 case 在留一法下准确率 0/2。它们没有任何非零相似度的邻居，
既没有历史可复用，也没有证据可供 LLM 推理，交给 N5c 只会得到一个类别先验的猜测。
新规则把它们直接路由到 `N6_abstain`。

**据此建议的 N4 v2 路由规则**（待 T5 实现，决策项见 `Validation.md` V1）：

| 分支 | 条件 | 处置 |
| --- | --- | --- |
| `N6_abstain` | `tokens` 为空 | 人工介入，不猜 |
| `N5a` | `max_similarity = 1.0` | 复用历史结论 |
| `N5b` | `evidence_coverage = 1.0` 且 `sim < 1.0` | 复用 + 按 `missing_evidence` 补采 |
| `N5c` | 其余 | 约束 + LLM 推理 |

**未采纳的信号。** 并列候选数与自身证据条数在两个切分上排序相反，是噪声。
冲突证据（`has_conflict`）在训练集上准确率 42.86%（3/7）、测试集上 71.43%（5/7），
方向相反且各只有 7 条，**不能作为路由信号**；它仍然保留在检索结构里供报告展示，
但 T5 的路由器不读它。

### 9.15 T3 / T4 的门禁复测

| 检查项 | 结果 |
| --- | --- |
| 全量 pytest | 97 passed（T1 前 57 → T3 后 80 → T4 后 97） |
| legacy 回归锚点 | `train-evaluate` 58/85 = 68.24%，`rule_overlap = 0`，未漂移 |
| 特征字典指纹 | `1b2e66ed650ce60e`，未变 |
| 约束库指纹 | `ee95eddd7885abdf`，未变 |
| v1 signature 分辨率 | 113 组 / 混合 7.94%，与 T1 逐位一致 |
| N4 分布（画板阈值） | 21 / 26 / 38，与 T1 逐位一致 |

### 9.16 T5 产物清单（2026-08-07）

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| M4 路由器 | `rca_framework/evidence_graph/router.py` | `RoutingPolicy`、`BOARD_POLICY`、`COVERAGE_POLICY`、`route` / `route_many`、`routing_summary` |
| 分支公共结构 | `rca_framework/branches/base.py` | `BranchOutcome`、`EvidenceLink`、`BranchCalibration`、`wilson_lower_bound` |
| N5a 处理器 | `rca_framework/branches/exact.py` | 按桶纯净度拆成 `N5a_pure` / `N5a_mixed` |
| N5b 处理器 | `rca_framework/branches/partial.py` | 补采清单 + 关键 / 非关键缺失判定 |
| N5c 处理器 | `rca_framework/branches/general.py` | 约束筛选、确定性排除、`DiagnosisRequest`（T6 的 prompt 载荷） |
| 装配 | `rca_framework/branches/dispatch.py` | `fit_calibration`、`handle` / `handle_many` |
| 端到端评估 | `scripts/evaluate_routing.py` | 两套规则对比、标定兑现检查 |
| 评估产物 | `artifacts/t5_routing_evaluation.json` | 完整结果 |
| 单测 | `tests/test_router.py` | 20 个 |

### 9.17 两套路由规则的端到端结果（85 条留出测试集）

| 规则 | N5a | N5b | N5c | N6 | 给出结论 | 给结论时准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `board-100-70` | 21 | 26 | 38 | 0 | 47（55.29%） | 32/47 = 68.09% |
| `coverage-v2` | 20 | 17 | 46 | 2 | 37（43.53%） | 29/37 = 78.38% |

分支级明细（`coverage-v2`）：

| 分支 | n | 给结论 | 判对 | 给结论时准确率 | 需 LLM | 需人工 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N5a | 20 | 20 | 15 | 75.00% | 1 | 0 |
| N5b | 17 | 17 | 14 | 82.35% | 3 | 0 |
| N5c | 46 | 0 | 0 | — | 46 | 0 |
| N6 | 2 | 0 | 0 | — | 0 | 2 |

**N5c 的 46 条全部不给结论，这是 T6 之前的预期状态，不是回归。**
确定性物理排除只能把候选从 3 个缩到 2 个，缩不到 1 个就不给结论——
不退回类别先验是刻意的，阶段 1 已经证明「零证据也给个答案」正是 legacy 的失败模式。
因此当前全量准确率（34.12%）与 legacy 的 58/85 不可比，要等 T6 接上 LLM 之后才可比。

### 9.18 置信度改为标定量而不是常数

常见做法是给 N5a 写 0.9、N5b 写 0.7、N5c 写 0.5。这些数字与实际正确率没有关系，
写进报告只会误导运维。这里改成：`BranchCalibration.fit` 在**训练集留一法**上
统计每个分组实际判对多少，把这个频率作为置信度，另附 Wilson 95% 置信下界。

用 Wilson 而不是正态近似，是因为样本量小、比例接近 0 或 1 时正态近似会越界
（例如把 2/2 的下界算成 1.0）。M9 的降级策略应当卡下界而不是点估计。

标定表与留出集兑现情况（`coverage-v2`）：

| 标定分组 | 训练集留一法 | 声称置信度 | 95% 下界 | 留出实测 | 差距 | 留出 n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `N5a_pure` | 15/18 | 83.33% | 55.20% | 73.68% | -9.65% | 19 |
| `N5a_mixed` | 4/6 | 66.67% | 30.00% | 100.00% | +33.33% | 1 |
| `N5b_minor_gap` | 26/37 | 70.27% | 54.22% | 87.50% | +17.23% | 16 |
| `N5b_critical_gap` | 3/4 | 75.00% | 30.06% | 0.00% | -75.00% | 1 |

`N5a_pure` 与 `N5b_minor_gap` 这两个有几十条样本的分组，标定值与留出实测差
10-17 个百分点，量级可以接受。样本量个位数的两个分组差距极大，
但它们的 95% 下界本来就只有 30% 左右，说明下界如实反映了「这个数字不可信」。
**报告里必须同时显示置信度与 `calibration_support`，单看点估计没有意义。**

N5a 按桶纯净度拆分是 AGENTS.md 的硬要求，实测证明这个拆分是必要的：
训练集留一法上 `N5a_pure` 83.33%、`N5a_mixed` 66.67%，差 16.7 个百分点。
混合桶的 case 会带 caveat 并被标记为需要 LLM 仲裁。

### 9.19 T5 抓出的约束缺陷：C6 排掉过真实标签

这是本步最重要的产物，来自一条刻意设计的全量校验测试
（`test_deterministic_exclusion_never_excludes_the_true_label`）：
**任何可执行的物理排除，都不允许在全量数据上排掉真实标签。**

初次运行就失败了。`C6_tx_down_excludes_medium`（本端未发光则排除介质根因）
在 211 条上触发 14 次，其中 **2 次排掉了真实的 fiber 标签**。

追查过程与结论：

1. 第一个假设是实现越权——`C6` 的形式化表达带「for that direction」限定，
   而实现把它提升到了 case 级。查下来假设不成立：2 条排错的 case 恰恰属于**双向都断光**的那组。
2. 直接看那 6 条双向断光的 case，发现它们的遥测**完全一致**：
   两侧 4 个 lane 的 `txpower` 与 `rxpower` 全部是 -40.0 dBm，介质侧信噪比也全部触底，
   状态位一律 `TxLOS=Normal`、`TxLOL=Normal`、`RxLOS=Abnormal`、`RxLOL=Abnormal`。
3. `TxLOS=Normal` 与 `txpower=-40` 直接矛盾：模块若真的关断了激光，TxLOS 应当告警。
   合理的解释是链路整体中断后遥测通道本身失效，所有光学读数回落到哨兵默认值。
   **此时哨兵表示「读不到」而不是「没有光」，`C6` 的前提根本不成立。**
4. 这 6 条的标签是 L2 四条、fiber 两条。物理观测完全一致而根因不同，
   它们在特征空间里还能分出 4 个 signature，唯一区别来自 `serdes_snr` 是
   `partial_lanes` 还是 `all_lanes`——而这个字段的量纲我们并不知道（`C13`）。

处置：新增 `C15_blackout_sentinel_is_not_laser_off`（measurement_validity / caveat）
作为 `C6` 的前置条件，约束库升到 `constraint-library-v2`（15 条，`content_hash=abb395e9371abc36`）。
加上前置条件后 `C6` 触发 8 次、排错 0 次。

| 阶段 | C6 触发次数 | 排掉真实标签次数 |
| --- | ---: | ---: |
| 加 C15 之前 | 14 / 211 | 2 |
| 加 C15 之后 | 8 / 211 | 0 |

这件事说明两点，都值得写进论文的讨论部分：

1. **物理约束必须对着数据做全量证伪，不能只靠专家写下来就当成立。**
   `C6` 的物理表述本身没错（光纤不能解释一束从未被发出的光），
   错在它的触发条件依赖一个含义会翻转的哨兵值。
2. **token 数量不等于证据强度。** 这 6 条 case 各产出十几个特征 token，
   在任何基于计数的证据充分性判断里都会被认为「证据充分」，
   但它们全部来自同一条失效的采集通道，实际有效证据是零。

据此给路由器加了 `abstain_on_optical_blackout` 开关，`coverage-v2` 默认打开。
需要注意**实测数据并不支持这个默认值**：这 6 条按历史多数投票能对 5 条（83.33%），
高于 65.40% 的全局先验；打开开关会让留出集的给结论准确率从 79.49% 降到 78.38%。
保留弃权的理由是那 5/6 完全建立在量纲未知的 `serdes_snr` 上，不是准确率理由。
这是一个需要拍板的取舍，已登记为 `Validation.md` V16。

### 9.20 T5 的门禁复测

| 检查项 | 结果 |
| --- | --- |
| 全量 pytest | 117 passed（T4 后 97 → T5 后 117） |
| legacy 回归锚点 | `train-evaluate` 58/85 = 68.24%，未漂移 |
| 特征字典指纹 | `1b2e66ed650ce60e`，未变 |
| 证据图指纹 | `5e10b5b25d559777`，未变 |
| 约束库指纹 | `ee95eddd7885abdf`（v1，14 条）→ `abb395e9371abc36`（v2，15 条），**有意变更**，原因见 9.19 |
| v1 signature 分辨率 | 113 组 / 混合 7.94%，与 T1 一致 |
| N4 分布（画板阈值） | 21 / 26 / 38，与 T1 一致 |
| 确定性排除全量校验 | 8 次触发，0 次排掉真实标签 |

### 9.21 T6 产物清单（2026-08-07）

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| M7 校验器 | `rca_framework/constraints/checker.py` | `check_evidence`、`check_response`、`Violation`、`CheckReport` |
| M8 输出协议 | `rca_framework/llm/protocol.py` | `ReasoningStep`、`DiagnosisResponse`、`parse_response`、guided decoding schema |
| M8 prompt 模板 | `rca_framework/llm/prompts.py` | `build_prompt`，版本 `n5c-constrained-reasoning-v1` |
| M8 后端 | `rca_framework/llm/backend.py` | `NoneBackend` / `VLLMBackend` / `ScriptedBackend` |
| M8 推理循环 | `rca_framework/llm/reason.py` | `ConstrainedReasoner`、`ReasoningTrace`、`Attempt` |
| legacy 兼容 | `rca_framework/llm/legacy.py` | 原 `llm.py` 原样搬入，`from rca_framework.llm import PathLLMReasoner` 仍可用 |
| 校验器审计 | `scripts/audit_constraint_checker.py` | 6 类失效模式 + 合规回答的拦截率 |
| 审计产物 | `artifacts/t6_checker_audit.json` | — |
| 单测 | `tests/test_constraint_checker.py`（19）、`tests/test_llm_reasoning.py`（26） | 共 45 个 |

`rca_framework/llm.py` 改成了包 `rca_framework/llm/`，legacy 实现搬到 `legacy.py`，
由 `__init__.py` 做兼容层。`pipeline.py` 的 `from .llm import PathLLMReasoner` 不受影响，
58/85 锚点复测通过，并有单测 `test_legacy_llm_imports_still_work` 锁定。

### 9.22 让 LLM 的每一步可被校验：输出协议的改动

legacy 的输出结构是 `prediction` 加一段自由文本 `reasoning`。这个结构下
「LLM 每步输出可被约束校验」这条验收无法实现——文本里说了什么无法机械判定。

新协议把推理拆成 `ReasoningStep` 序列，每步必须声明四样东西：

| 字段 | 作用 |
| --- | --- |
| `claim` | 这一步断言了什么 |
| `cited_evidence` | 用到了哪些证据 token，必须在证据包里真实存在 |
| `cited_constraints` | 依据了哪几条物理约束，必须在约束库里存在 |
| `effect` + `target` | 对哪个根因起 `support` / `exclude` / `neutral` 作用 |

`cited_evidence` 是防幻觉最有效的一招：模型可以编造措辞，
但编不出一个不在证据包里的 token 而不被发现。`effect` 结构化之后，
「这一步排除了 fiber」可以直接与约束库的排除条件对照，不需要理解自然语言。

解析器刻意不做容错补全。模型没按 schema 输出时，它的推理过程本来就无法校验，
此时最安全的处置是重写或弃权，而不是猜它想说什么。

### 9.23 M7 校验的四层断言与审计结果

| 违规类型 | 严重程度 | 判据 |
| --- | --- | --- |
| `fabricated_evidence` | fatal | 引用了证据包里不存在的 token |
| `fabricated_constraint` | fatal | 引用了不存在的约束编号 |
| `constraint_violation` | fatal | 结论落在已被确定性排除的根因上；或全链路失效时仍给结论 |
| `forbidden_claim` | fatal | 说了 `caveat` 类约束明令禁止的话（C12 / C13 / C14 / C15） |
| `unsupported_step` | fatal | 某步既没引证据也没引约束，凭空断言 |
| `invalid_measurement` | warning | 证据包本身违反 invariant 类约束 |

`invalid_measurement` 是 warning 而不是 fatal：它说明数据有问题，
应当让推理者知道这条读数不可信，但不该阻断推理。

审计结果（`scripts/audit_constraint_checker.py`，46 条 N5c case × 7 种回答）：

| 失效模式 | 应拦截 | 实际拦截率 |
| --- | --- | ---: |
| `fabricated_evidence` | 是 | 46/46 = 100% |
| `fabricated_constraint` | 是 | 46/46 = 100% |
| `unsupported_step` | 是 | 46/46 = 100% |
| `absolute_loss_claim`（C12） | 是 | 46/46 = 100% |
| `serdes_db_claim`（C13） | 是 | 46/46 = 100% |
| `host_snr_normal_claim`（C14） | 是 | 46/46 = 100% |
| `compliant` | **否** | 0/46 = 0% |

**这个 100% 不能被解读成「校验器是鲁棒的」。** 攻击样本是我们自己按已知失效模式构造的，
与校验器的检测规则同源，属于自己考自己。它的真实价值是回归保护：
以后改动校验器或约束库时，这些拦截不能悄悄失效。真正的鲁棒性要等真实模型跑过才知道。

合规回答 0% 拦截同样重要且容易被忽略：一个把合规回答也拦下来的校验器
会让系统陷入无限重写，比不校验更糟。`FORBIDDEN_CLAIM_PATTERNS` 的正则
因此写得保守，宁可漏检也不误伤，并有 `test_forbidden_patterns_do_not_fire_on_ordinary_text` 守着。

### 9.24 受约束推理循环与真机验证方法

循环是「生成 -> 逐步校验 -> 不合规则带着违规原因重写 -> 仍不合规则弃权」。
三个设计决定：

1. **重写用尽后弃权，而不是接受最后一次输出。** 被判 fatal 的违规里最常见的是
   引用了不存在的证据；一个建立在虚构证据上的结论比没有结论更有害。
   有单测 `test_persistent_violation_ends_in_abstention_not_acceptance` 锁定。
2. **重写反馈必须标注为「上一次回答的问题」并放在证据之前**，否则模型会把它当成新证据。
   有单测锁定这个顺序。
3. **批量重试只重发未通过的 case。** N5c 占一半以上的 case，逐条调用会浪费 vLLM 的吞吐；
   已通过的不该在重写轮里再消耗一次生成。有单测 `test_batch_reasoning_only_retries_the_failing_cases` 锁定。

每一轮的 prompt、原始输出、校验报告都记进 `ReasoningTrace`，
这就是画板要求的逐步推理日志，可直接进报告与论文附录。

prompt 由 `DiagnosisRequest` 与约束库渲染，不逐 case 手写。实测每条 N5c prompt
约 3000 字符，只注入与该 case 相关的约束（本数据集上是 3-12 条，不是全部 15 条）。
`abstain` 是 schema 里的合法取值且在 prompt 里明确说明——
如果只给三个根因，模型一定会三选一，而这正是阶段 1 认定的 legacy 主要失败模式。

**历史待验证项已于 2026-08-07 完成首测。** 当时列出的四个问题已有初步答案：

- 真实模型可以批量产出可解析、可校验的结构化输出，但结构通过不代表物理语义正确。
- 日志显示首轮后有少量 case 被重发；当前聚合产物没有单独统计首轮/重写通过率，正式实验仍需补报。
- `coverage-v2` 的 N5c 给结论 22 条只判对 5 条，整条链路没有超过 legacy 58/85。
- 模型会主动弃权，`coverage-v2` 下 N5c 弃权 24/46。

完整数字、产物和新暴露的问题见第 9.25-9.26 节。

真机命令：

```bash
python scripts/evaluate_routing.py \
  --llm-backend vllm --model-path <模型路径> \
  --max-attempts 2 --output artifacts/t6_llm_evaluation.json
```

当前 `--llm-backend none` 的结果与 T5 完全一致（N5c 全部弃权），仍作为对照基线。

### 9.25 T6 真实模型首测（2026-08-07，补记于 2026-08-09）

旧版进度写“本机无 GPU、尚未真机验证”，但当前仓库已经存在完整 vLLM 日志和结果文件。
日志记录的模型为 `/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B`，
vLLM 0.11.0、tensor parallel 2、`max_model_len=8192`、temperature 0、
`disable_custom_all_reduce=True`，prompt 版本为 `n5c-constrained-reasoning-v2`。

产物：

| 产物 | 作用 |
| --- | --- |
| `artifacts/t6_smoke_promptv1.json/.log` | 初版 prompt 真机冒烟 |
| `artifacts/t6_smoke_promptv2.json/.log` | v2 prompt 8 条 N5c 真机冒烟与逐步 trace |
| `artifacts/t6_llm_evaluation.json/.log` | 85 条留出集、两套路由的端到端首测 |

首测结果：

| 路由 | N5c 总数 | N5c 回答 | N5c 判对 | N5c 回答准确率 | N5c 弃权 | 全链路回答 / 85 | 全链路答对 | 给结论时准确率 | 全量准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `board-100-70` | 38 | 13 | 5 | 38.46% | 25 | 60 | 37 | 61.67% | 43.53% |
| `coverage-v2` | 46 | 22 | 5 | 22.73% | 24 | 59 | 34 | 57.63% | 40.00% |

与 T5 的 `coverage-v2 --llm-backend none` 对比：不接 LLM 时回答 37 条、答对 29 条、
给结论时准确率 78.38%；接 LLM 后新增回答 22 条，只新增答对 5 条，
使给结论时准确率降到 57.63%。因此当前 LLM 的作用是提高覆盖率，但显著提高了选择性风险。
这不是“LLM 已接入所以 T6 成功”的效果结论，而是下一步必须做降级和语义校验的直接证据。

行为层面可以确认两件事：

1. 模型会主动弃权，`coverage-v2` 下 24/46 N5c 没有硬猜。
2. 结构化解析与批量重写链路可运行；日志显示首轮后仍有少量 case 被重发。

### 9.26 T6 首测暴露的实现与实验缺口

1. **M7 仍是结构校验器，不是完整物理推理校验器。** 真实输出中出现约束与证据类型错配，
   例如引用 `C7_rx_power_range` 解释 `txpower`、引用 `C13` 的“量纲未知”去支持设备根因，
   以及把 `media_snr high_tail` 当成故障线索；这些回答仍通过 checker。
   下一步要校验“约束适用的 token 家族、允许的 effect/target、前提是否满足”，
   并把真机 badcase 固化成回归测试。
2. **N5a/N5b 的仲裁没有接线。** `exact.py` 和 `partial.py` 会设置 `needs_llm`，
   但 `dispatch.handle_many` 只批量收集 `decision.branch == "N5c"` 的 case。
   因此画板中 N5b 的“缺关键证据或候选冲突触发 LLM 仲裁”目前只停留在状态标记。
3. **N5c 置信度没有标定。** `general.handle` 忽略模型返回的 confidence，
   使用 T5 无模型阶段拟合的 `N5c_*` 分组；这些组的训练留一法 provisional verdict 均为空，
   所以对外置信度固定为 0。T7 必须明确是校准模型 confidence，还是基于独立验证集重新拟合风险。
4. **正式实验记录不合格。** `t6_llm_evaluation.json` 只有 graph version 和聚合 report，
   没有模型、prompt hash、约束版本、seed、生成参数、逐 case outcome/trace 和类别指标；
   不满足第 5.2 节的新框架门禁。
5. **两套路由不是完整可归因消融。** 脚本会分别调用 LLM，并只保存聚合结果。
   正式比较应缓存每个 case 的模型输出，或固定 prompt 集后复用 trace，避免路由变化与生成差异混在一起。
6. **新框架没有统一入口。** 当前通过分析脚本装配 N1-N6，没有可保存/加载的新 pipeline、
   独立 CLI、报告生成和反馈回灌；距离图中的“在线输入 → 报告/工单 → 证据图更新”仍有 T8-T10 的工程缺口。

### 9.27 T7 统一置信度、降级出口与正式记录（2026-08-10）

T7 已完成代码侧闭环：

1. `branches.dispatch.handle_many` 先执行确定性分支处理，再把所有 `needs_llm=True` 的
   N5a 混合桶、N5b 关键缺失/候选冲突和 N5c case 合并成一批请求。prompt v3 会显式记录
   分支、路由原因和历史候选标签分布，不再把 N5a/N5b 错写成“无历史匹配”。
2. `LLMCalibration` 按分支和模型 confidence 分桶，只用训练集留一法输出统计正确率与
   Wilson 95% 下界。模型自报 confidence 只作分桶依据，不直接当作可靠性；
   没有独立标定支持时对外下界固定为 0。
3. `decision.py` 统一输出 `final` / `request_evidence` / `human_review`。V18 未确认前默认
   要求 Wilson 下界至少 0.50 且标定支持数至少 10；阈值均可通过评估脚本参数覆盖。
4. `evaluate_routing.py --output-dir` 写出 `run_manifest.json`、`summary.json`、
   `outcomes.json`、`traces.json`。manifest 记录数据切分、图/字典/约束/SOP/prompt/决策版本、
   prompt hash、Top-N、路由阈值、模型生成参数与 seed；summary 增加三类指标、
   N5a 纯净度、降级率、人工介入率和 precision-at-coverage 曲线。

无模型 `coverage-v2` 门禁结果：

| 口径 | 回答 / 85 | 答对 | 给结论时准确率 | 补采 | 人工 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T5 原始分支输出 | 37 | 29 | 78.38% | — | — |
| T7 M9 最终出口 | 35 | 28 | 80.00% | 17 | 33 |

M9 降级了 1 条 N5a mixed 和 1 条 N5b critical；两组训练留一法 Wilson 下界分别只有
30.00% 和 30.06%。保留的 35 条来自 N5a pure 与 N5b minor，覆盖率从 43.53% 降到
41.18%，给结论时准确率从 78.38% 升到 80.00%。这是阈值工作点的描述，不是方法提升结论。

后续正式真机门禁已写入 `artifacts/t7_formal_deepseek32b_promptv3_seed42/`：
原始分支回答 52/85、答对 31；M9 最终保留 33/85、答对 26，给结论时准确率 78.79%，
补采 28、人工介入 24。该结果完成了 prompt v3、N5a/N5b 仲裁和正式记录的工程验收，
但 `fiber` 仍为 0/6，也不能据此宣称整体三分类效果超过 legacy。

### 9.28 T10 独立规则 empirical study（2026-08-10）

为单独回答“当前规则是否让大模型判断更准”，新增
`scripts/run_rule_empirical_study.py` 与固定 prompt
`rca_framework/llm/empirical.py`。实验使用 DeepSeek-R1-Distill-Qwen-32B、
固定 126/85 切分、`temperature=0`、`seed=42`，三个实验臂共享同一批 evidence token、
输出 schema 和解码参数：

1. `evidence_only`：只给当前 case 证据，不给规则，不做 checker。
2. `rules_prompt`：加入当前 case 相关物理规则和确定性排除，但不拦截输出。
3. `rules_prompt_checker`：复用第 2 组首轮输出，只对违规 case 重写一次。

历史 case、历史标签、证据图投票和 M9 最终门禁均不进入 prompt；证据图只用于按
N5a/N5b/N5c/N6 分层报告。正式产物在
`artifacts/rule_empirical_study_deepseek32b_seed42/`：

| 实验臂 | 回答 / 85 | 弃权或无可用答案 | 答对 | 给结论时准确率 | 全量准确率 | 规则合规 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 只给证据 | 19 | 66（77.65%） | 7 | 36.84% | 8.24% | 77/85 |
| 加规则 prompt | 15 | 70（82.35%） | 8 | 53.33% | 9.41% | 70/85 |
| 加规则 prompt + checker | 14 | 71（83.53%） | 7 | 50.00% | 8.24% | 77/85 |

分层事实：

- N5c：三组分别为 4/9、5/10、5/11（正确/回答），当前规则没有明显解决未见模式。
- `fiber`：三组均答对 0/6；规则组只预测过 1 条 fiber，且判错。
- 规则 prompt 相对 evidence-only 有 6 条从错/弃权变对，也有 5 条从对变错/弃权，
  正确数净增 1；配对 McNemar 精确检验 `p=1.0`。
- 规则组首轮有 15 条未通过 checker；重写后总合规数从 70 提到 77，但正确数从 8 降到 7。
  checker 在做安全过滤，但没有产生效果收益。

因此，本轮可以支持的结论是：**当前规则会让模型更保守，并提高“已回答样本上的表面精度”，
但没有证据证明它显著提高了全量正确判断数。** 15 条规则中多数是量测有效性、禁止误推、
物理范围或弱指示，真正能把 L1/L2/fiber 唯一分开的强规则不足；当两端同时出现异常时，
模型大量合理弃权。后续应先完成专家审核、补充可判别证据和规则，再复用同一脚本做配对复验，
不应把 53.33% 解读为方法已经有效。

### 9.29 非自进化 SOP+LLM 离线实验工程闭环（2026-08-11）

本轮按用户确认的边界排除 N8 回灌和自进化，只实现“训练知识沉淀 → 测试只读推理 → 深度报告”：

1. 新增 `rca_framework/knowledge.py` 与 `offline-knowledge-bundle-v1`。知识包保存 train-only
   ThresholdModel、FeatureModel、161 条 CaseFeatures 稀疏向量、EvidenceGraph、learned SOP、
   分支标定、LLM 标定和构建元数据；加载时同时校验 bundle hash、字典 hash、case 顺序与数量。
2. 训练阶段对 manifest train 做 leave-one-out 分流。需要 LLM 的 case 运行受约束推理，
   合规链与 M9 结果写入 evidence-graph-v2 的 per-case 诊断子图；训练标签只作为历史 case
   的确认标签。所有 161 条训练 case 都有诊断子图，原始训练 trace 独立保存。
3. `learned-sop-v1` 不再替代正式 LLM：SOP 叶子、路径、支持数和分布作为统计先验进入
   prompt v6；prompt 明确禁止把 SOP 当作 `cited_evidence`。LLM 仍只能引用当前 case token
   和约束 ID，输出继续经过 M7 逐步校验与重写。
4. 新增 `scripts/run_offline_sop_llm_experiment.py`。正式入口要求 manifest 161/107 门禁，
   保存知识包后强制从磁盘重新加载，再处理 test；test 标签只在全部推理完成后用于指标。
5. 正式入口在加载模型前检查 GPU，记录运行前快照；`finally` 中关闭 vLLM、销毁分布式状态、
   清理 CUDA cache，并写运行后快照和显存释放判定。正式实验后仍需进程外再跑一次
   `nvidia-smi` 复核。
6. 新增标准库 HTML 渲染器，输出 `html/index.html` 和每个 policy/case 的独立页面。
   总览按正确、错误、弃权/补采、遥测不足分组；逐 case 展示特征、历史候选、SOP、
   物理证据链、M9 原因、缺失证据、LLM 每轮 prompt/原始输出/违规和诊断图。
7. M7 v3 的 token/effect/target 语义契约正在补齐，并按 l2fixed manifest train split
   重算 C14/C15 的实测口径。收紧后旧测试中“media_snr token 引用 C7 rxpower 约束”
   这类伪合规输出已被正确拦截；相关固定样例需改成使用匹配约束或不引用约束。

门禁结果：`python -m pytest -q` 为 **199 passed**；l2fixed manifest / quality report
`--check` 全绿；约束技能与 library v3 一致；`compileall` 和正式入口 `--help` 通过。

当前状态：知识包、SOP+LLM 编排、正式入口、HTML 渲染、约束契约和代码侧门禁已完成；
GPU 正式实验、资源复核和最终结果见 9.30。

### 9.30 非自进化 SOP+LLM 正式实验结果（2026-08-11）

正式结果目录：
`artifacts/offline_sop_llm_l2fixed_deepseek32b_seed42_promptv6/`。

- 数据边界：manifest train/test = 161/107；知识包先保存后重载；历史向量 161 条，
  EvidenceGraph case/诊断子图均为 161；无 test 回灌。
- 模型与资源：DeepSeek-R1-Distill-Qwen-32B，GPU 6/7，tensor parallel = 2，
  seed = 42，prompt = `rca-constrained-reasoning-v6`。运行状态 completed，
  `backend_close_called=true`、`gpu_memory_released=true`；GPU 6/7 运行后各仅占 4 MiB。
- 路由：N5a/N5b/N5c/N6 = 15/26/64/2。M9 前形成 44 个候选，23 个正确，
  候选准确率 52.27%；N5a 9/13、N5b 13/21、N5c 1/10。
- M9：没有标定分组同时达到 Wilson 下界 0.5 与支持数 10，因此最终自动结论为 0；
  71 条 request_evidence、36 条 human_review。这个结果是安全门禁按设计工作，
  也说明当前系统尚不能无人值守。
- SOP 对照：总体 70/107 = 65.42%，但 L1/L2/fiber 分别为
  17/32、53/67、0/8。总体数值明显受 L2 多数类主导，fiber 路径尚未学会。
- LLM：测试侧 73 条 trace，61 条发生重写，22 条通过 M7，其中 12 条形成结论；
  两轮累计 599 条违规。最高频仍是把 neutral 约束当 support（181）和
  token 家族与约束适用范围错配（175）。prompt v6 相比首轮把通过数从 15 提到 22、
  违规从 769 降到 599，但 N5c 质量没有达标，不能通过放松 M7 获得表面覆盖率。
- HTML：`html/index.html` 含实验深度分析、SOP/分支/LLM 失败拆解和 case 分组；
  `html/cases/` 含 107 个逐 case 深度页面。自动核验为 108 个 HTML 文件、
  107 条 case 链接、0 个缺链、0 个缺少必需审计章节的页面。

本轮可支持的效果结论是：**离线知识沉淀与只读测试工程闭环已经完成，但算法效果未达标。**
主要瓶颈依次是 N5c 受约束推理、fiber 少数类知识、约束引用语义，以及 M9 可用标定样本不足。

### 9.31 迭代 1：候选知识审计与嵌套验证（2026-08-11）

本轮目标是三件事：从训练集与测试集里挖出更多可用知识、把过高的拒绝率降下来、
并保证产出的 SOP 与约束是可解释的而不是拟合当前数据。测试集在全过程只提供
**无标签触发率**，不提供证据图、历史向量或标签，因此下面所有 precision 都是 train-only。

#### 新增的四个离线工具

| 脚本 | 回答的问题 |
| --- | --- |
| `scripts/mine_knowledge_candidates.py` | 哪些 token、token 对、派生探针与标签相关 |
| `scripts/audit_candidate_confounding.py` | 这些候选是新证据，还是已有 token 的代理 |
| `scripts/ablate_feature_families.py` | 每个特征家族在 LOO 下带来泛化还是只带来拟合 |
| `scripts/nested_validate_policy.py` | 把「挑配置」的代价算进去后，策略还有多少收益 |

#### 候选审计：8 个候选否掉 7 个

挖掘阶段在 `with_probe` 口径下有 152 条 token 对的 Wilson 下界超过其预测类别的先验，
看起来收获很大。审计阶段用三项检验逐个过筛（共线性、控制共线者后的剩余增益、
在无断 lane 的「安静 case」上的命中率），结果只剩 1 个候选，
而这唯一的存活者恰好是物理上最站不住的那个：

- `probe:txpower_side_gap:L1_worse`（两端发送功率相减）train n=37、precision 81.1%、
  下界 65.8% 高于 L2 先验 62.1%。但它与 `level:L1:txpower_mean:low_tail` 的 Jaccard 达 0.65，
  控制该 token 后剩余支持只有 7 条，增益消失。
- 追查发送电平本身：`level:L1:txpower_mean:low_tail` 命中 39 条且**无一条含断光哨兵**，
  说明它描述的确实是「正常带内偏低」。而按标签分层的健康 tx 中位数几乎相同
  （标签 L1 的 case 为 +0.860 / +0.863 dBm，标签 L2 为 +0.835 / +0.855 dBm），
  即发送电平与根因基本无关，低尾组的标签偏斜是尾部抽样波动。

这条路径被写成 **C21**：正常带内的发送功率高低不是归因证据，两端相减更不是。
它拦掉的是一条统计上很诱人、物理上讲不通的捷径（本端发送功率偏低却指向对端根因）。

同一轮审计里唯一站得住的正向发现被写成 **C22**：`imbalance:L2:rxpower` 命中 7 条，
其中 6 条根因在对端 L1（85.7%，下界 48.7%，L1 先验 30.4%），且无一条含断光哨兵。
这是全训练集上**唯一**一个下界超过 L1 先验的观测条件。它的物理理由是可迁移的：
同侧 lane 间极差消掉了标定口径与整束光纤的共模损耗，剩下的差异只能来自对端发送阵列，
因此「用同侧相对量做跨端归因」比 C12 已经否掉的「两端绝对电平相减」可靠。
镜像方向（L1 侧不均衡支持 L2）下界 49.0% 低于 L2 先验 62.1%，不成立；
这个不对称主要来自两类先验相差一倍，不是物理不对称。

#### 两端 lane 是否对齐：一个只用序的检验

为找「不依赖绝对电平」的跨端量，新增了探针「两端**最差 lane 的编号**是否相同」。
它只用序，不受标定口径影响。结果它单独并不超过先验，但给出了 C12 的第二个证据，
而且比原来的「损耗中位数为负」直观得多：若两端 lane 编号真的对应，
故障 case 里最差 lane 一致的比例应明显高于随机；实测 rxpower 为 37/155 = 23.9%、
media_snr 为 46/161 = 28.6%，而 4 lane 下随机一致就是 25%，两者都与随机无法区分。
这个检验只需要遥测本身，任何团队都可以用它先判断自己的两端 lane 能不能按号配对，
再决定要不要做跨端功率计算。已写入 C12 并加了复算测试。

#### 一个通过审计但仍不敢上线的候选

沿着 C22 的机制（同侧相对量做跨端归因）继续测「哪一端的 lane 更不齐」和
「两端 SNR 谁更差」，得到本轮唯一一个通过全部审计的候选：
`probe:media_snr_side_gap:L2_worse` → L1，n=28、precision 57.1%、下界 39.1%
高于 L1 先验 30.4%，**4/4 个控制分层都保留增益**，且在「安静 case」上的命中率
（8.2%）低于全体（17.4%），不像形态偏差。按标签分层也很干净：
「L2 侧 SNR 更差」在 L1 根因下命中 32.7%，在 L2 根因下只有 9.0%。
lane 离散度的分层更漂亮——L2 根因时 L1 侧 SNR 极差中位 1.695 dB 对 L2 侧 0.755 dB，
L1 根因时方向反转（1.080 对 0.900），正是「受害端的对面才是肇事端」。

但它没有被写成 support 约束，原因是一次口径检验：字典里已有的 `side_asymmetry`
家族做的就是两端对比，只是用 `media_snr_min` 而不是均值，
而它的 6 个 token **全部不超过先验**（最高下界 0.408，L2 先验 0.621）。
同一个物理量换一个统计量信号就消失，说明这个关联对度量细节敏感，
在只有 161 条训练样本、且本轮一共检验了 40 个探针的情况下，
它更可能是多重比较的产物。已注册 `v2_plus_side_asymmetry` profile 留待
新数据集上复验，但不进 v2 字典、不进约束库。

这也是本轮方法上的一个自我约束：**通过审计不等于可以上线**，
还要看它在同一物理量的其它合理口径下是否同样成立。

#### 特征家族 LOO 消融

训练集 161 条、留一法重学 SOP：

| 消融 | LOO 准确率 | L1 召回 | L2 召回 | fiber 召回 |
| --- | ---: | ---: | ---: | ---: |
| 全特征 | 0.6335 | 0.286 | 0.880 | 0.000 |
| 去 imbalance | 0.6584 | 0.327 | 0.900 | 0.000 |
| 去 status | 0.6460 | 0.122 | 0.980 | 0.000 |
| 去全部 level | 0.6584 | 0.286 | 0.920 | 0.000 |

三个事实：全特征 LOO 准确率的 Wilson 下界是 0.5568，**低于多数类先验 0.6211**，
即「按 SOP 判」与「一律报 L2」在统计上分不开；`fiber` 召回在所有消融下恒为 0，
与 C20 一致；`status` 家族是 L1 召回的唯一主要来源，去掉后 L1 召回崩到 0.122、
L2 召回升到 0.980，模型退化成报多数类。

另外，`imbalance:L2:rxpower` 在 161 次 LOO 中每次都被选为根分裂，但屏蔽它后指标反而变好。
原因是它只覆盖 7 条：信息增益偏爱「小而纯」的稀有 token，根分裂被劫持后
整棵树的深度预算浪费在覆盖极少的分支上。这给出一个分工——
统计层不该让稀有 token 主导树结构，而它作为显式 indicator（C22）交给知识层是合适的。

#### 阈值扫描的陷阱与嵌套验证

先修正了一个会导致错误结论的口径问题：门限内的精度必须与**同一批被保留 case** 上的
多数类基线比。用全量先验作基线时，所有配置都「稳定优于多数类」，
但那测的是「门限挑出了容易的 case」，不是「SOP 比拍多数类强」。改成同子集基线后，
270 个配置里只有细树低门限的少数几个还有正 lift，最好的一个是
`no_imbalance / depth 4 / leaf 5 / gate 0.40`：覆盖 77.0%、精度 73.4%、
同子集多数类 63.7%、lift +9.7pp、人工 23.0%。

这个数字不能用。它是从 270 个配置里取的最大值，偏差与选择过程同源。
`nested_validate_policy.py` 用外层 5 折分层、内层只在训练部分 LOO 选配置来重测：

- 5 折中有 **4 折的内层直接选了「永远报多数类」**，唯一选了 SOP 配置的那折
  在 held-out 上 lift 为 **-15.4pp**。
- 汇总 held-out：覆盖 96.3%、选择性精度 60.7%、同子集多数类 63.2%、
  **lift -2.58pp**。也就是说把配置选择的代价算进去后，这套策略不如直接报 L2。

因此本轮的核心结论是：**当前特征字典 + 浅决策树在这份数据上没有超越多数类的能力，
调阈值或调 SOP 超参不可能带来真实的准确率提升。** MVP 里 M9 全部拒绝并不是门限设过严，
而是标定过程如实发现了「没有可信信号」。

#### 由此暴露的指标问题

在这份数据上，同时最大化「准确率」与「最小化人工干预」的最优解是退化解：
一律报 L2，测试集准确率 62.6%、人工干预 0%，且在嵌套验证下打败所有 SOP 配置。
它对 L1 与 fiber 完全无用。这说明这两个指标单独使用会奖励退化行为，
后续必须与至少一个能区分「有诊断价值」与「猜多数类」的指标一起看，
例如同子集 lift、各类召回的均值，或弃答有效性（转人工的 case 里模型原本会答错的比例）。

知识层的定位也相应调整：不再指望物理约束把三分类准确率抬上去，
而是用它缩小候选并产出明确的补采动作——这正是 C17、C20、C22 已经在做的事。

这三项指标已经落进正式评估链路（`scripts/evaluate_routing.py` 的 `degeneracy_guard`），
写入每次实验的 `summary.json`，并且守护本身有测试
（`tests/test_degeneracy_guard.py`：一律报多数类时 lift 必须为 0、平衡召回必须掉到 1/3 附近，
全部弃答时不许崩）。另有 `scripts/score_decision_quality.py` 可对历史产物做同口径复盘。
用它复盘 MVP 的结果：107 条全部转人工，弃答有效性 0.3458——
与 SOP 全量错误率 0.3458 完全相同，因为弃答集就是全集，说明那次弃答没有任何选择性。

#### 迭代 1 正式实验结果（`artifacts/i1_offline_sop_llm_risk035/`）

配置与 MVP 同源（DeepSeek-R1-Distill-Qwen-32B、GPU 2/3、TP=2、seed 42、prompt v6、
manifest 161/107、约束库 v4——v5 在本轮之后才落库），只改三处策略：
按目标选择性风险 0.35 在 train-LOO 上拟合门限、把 SOP 加进候选级联、
把 `fiber` 声明为不可辨识标签。拟合出的门限是 0.4104
（训练 LOO 覆盖 81.99%、实测风险 32.58%、支持数下限 10）。

| 指标 | MVP | 迭代 1 |
| --- | ---: | ---: |
| 自动结论覆盖率 | 0.00% | 66.36% |
| 给结论时准确率 | — | 70.42% |
| 人工介入 | 36 条（33.64%） | 7 条（6.54%） |
| 补采 | 71 条 | 29 条 |
| **同子集多数类基线** | — | **69.01%** |
| **lift over majority on kept** | — | **+1.41pp** |
| **平衡召回** | 0.00 | **0.2596** |
| **弃答有效性** | 0.3458 | **0.5278** |

前三行是用户提的两个目标：覆盖率从 0 升到 66.36%，人工介入从 33.64% 降到 6.54%，
给结论时准确率 70.42%。后四行说明这个成绩的真实成分：

- 在被保留的同一批 71 条 case 上，一律报 L2 就有 69.01%，系统只多对了 1 条（+1.41pp）。
- 平衡召回 0.2596，**低于随机猜一个类的 1/3**。分类召回是
  L1 6.25%（32 条只对 2 条）、L2 71.64%、fiber 0%。
- 71 条自动结论里 **59 条来自 SOP**（精度 71.19%），只有 12 条来自分支（66.67%）。
  也就是说覆盖率的提升几乎全部来自「把多数类机器接进级联」。

> 补注（迭代 2）：这一节的门限 0.4104 与 66.36% 覆盖率**是在一个反序的置信度上标定出来的**，
> 不要作为工作点引用，见 §9.32。修正置信度后，同样的 0.35 目标风险只能拿到 8.41% 覆盖。

因此这一轮的结论要分开说：**工程目标达成，算法能力没有提升。**
覆盖率与人工干预率的改善是真实的、可交付的，但它来自门限校准与兜底策略，
不是来自更强的判别力；把 70.42% 解读为方法进步是错的。

两处是实质进步：

1. **弃答有效性从 0.3458 提到 0.5278**。转人工或补采的 36 条里有 19 条是 SOP 会答错的，
   弃答第一次表现出选择性（MVP 那次弃答集就是全集，有效性等于全局错误率，等于没有选择）。
   这个方向是对的：人工介入的价值在于「用在会错的地方」，不在于「少用」。
2. **8 条 fiber 全部没有被给出错误结论**，而是按 C20 与 `non_identifiable_labels`
   转成带具体补采项的 `request_evidence`（OTDR 曲线、端面镜检、双向功率标定、换纤复测）。
   MVP 那次系统预测 fiber 10 次只对 1 次，本轮预测 0 次——
   把一个不可识别的类别变成明确的补采分支，比继续低精度猜它更有价值。

#### 门禁

约束库升到 `constraint-library-v5`（22 条，指纹 `b1b00ef1d0f3493a`）。
`tests/test_constraint_library.py` 新增两条测试：一条把 C21/C22 的实测数字
从数据里重算出来核对，另一条补上了文件 docstring 从 T2 起就声称存在、
但实际一直没写的 SKILL.md 同步守护——正因为缺了它，约束库升级时
`skills/rca-constraints/SKILL.md` 还停在 v4 也没有被拦住。
`python -m pytest -q` 为 **203 passed**。

### 9.32 迭代 2：留一法置信度反序，以及迭代 1 覆盖率的真实来源（2026-08-11）

本轮的起点是一个结构性怀疑：L1 先验只有 30.4%、L2 有 62.1%，
一条统一门限会不会天然把 L1 全部挡掉？于是先做了按类别校准门限的能力
（`per_label_lower_bound` + `refine_per_label_bounds`，坐标上升，
要求**每一类的选择性风险各自达标**而不是只看整体）。
结论是这条怀疑只说对了一半，而在验证它的过程中挖出了一个更严重的问题。

#### 置信度与正确性在叶内是反序的

新增 `scripts/probe_per_label_operating_points.py`，把每个预测类别的
风险-覆盖率曲线单独打出来。L2 曲线立刻暴露了不可能的形状：

| 下界 | 作答 | 判对 | 纯度 |
| ---: | ---: | ---: | ---: |
| 0.6320 | 39 | 28 | 71.79% |
| 0.6411 | 22 | 11 | 50.00% |
| 0.6649 | 10 | 0 | **0.00%** |
| 0.7510 | 3 | 0 | **0.00%** |

**置信度最高的一批候选全错。** 逐条打开后机制非常清楚，而且不是 bug 而是恒等式：
SOP 候选的置信度取自「去掉自己重拟合」的叶节点纯度，

    纯度(去掉 case i) = (符合该叶结论的样本数 − [i 符合]) / (叶大小 − 1)

所以同一叶子内，**符合结论的 case 必然拿到比不符合的 case 更低的置信度**。
叶 `root.absent.present.absent`（判 L2）上：真值为 L2 的 28 条下界 0.6320，
真值为 L1 的 0.6666~0.7149，真值为 fiber 的 0.7510。三个叶子无一例外，
叶内 AUC 分别是 0.0714、0.0000、0.2857——几乎完全反序。

后果是致命的：按这个置信度反解门限，得到的门限**专挑反例放行、把正例挡在门外**。
迭代 1 的平衡召回 0.2596（低于随机猜一类的 1/3）与「高置信度组在测试集上反而不准」
（`sop:root.absent.absent.present` 声称 42.86%、实测 31.58%）都由此而来。

修法是 `_out_of_fold_sop_predictions`：改用分层 5 折的折外模型打分。
折外同样严格无自身泄漏（预测某条 case 的模型从未见过它），
但被留出 case 的标签只通过它所在那一折（约 20% 数据）影响叶纯度，
不再是唯一扰动源。修后叶内 AUC 变成 0.4186 / 0.6188 / 0.9000，
合并 AUC 从 0.4491 升到 0.5743，L2 曲线也恢复了单调：
下界 0.5682 时 33 条、纯度 75.76%，顶端不再是 0%。

这一条已用 `tests/test_confidence_ordering.py` 锁住，包括一条**故意断言留一法仍然反序**
的测试——它保护的不是旧行为，而是「为什么不能用留一法反解门限」这条理由始终可复现。

对业界的可迁移含义（与具体数据无关）：**只要置信度是从"去掉被评估样本后重拟合"
的模型里读出来的，它就与被评估样本的标签构造性负相关；样本越少、叶越小，反序越彻底。**
留一法能防住正向泄漏，但会引入反向污染，用它做选择性预测的门限标定是不安全的。
这类缺陷不会让任何断言失败，也不会让指标变难看——恰恰相反，它让指标变好看。

#### 修正后：迭代 1 的 66% 覆盖率不复存在

把折外置信度接进正式链路后重跑（离线、不接 LLM、同一 161/107 划分）：

| 目标风险 | 覆盖率 | 给结论时精度 | 同子集多数类 | lift | 平衡召回 | 人工+补采 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 / 0.35 | 8.41% | 88.89% | 88.89% | **0.00pp** | 0.0398 | 91.59% |
| 0.40 | 66.36% | 70.42% | 69.01% | **+1.41pp** | 0.2596 | 33.64% |
| 0.45 | 93.46% | 59.00% | 64.00% | **−5.00pp** | 0.3534 | 6.54% |

三点结论：

1. 迭代 1 那个「风险 0.35 → 覆盖 66.36%」的工作点，在置信度修正后需要把目标风险
   放到 0.40 才能拿到；0.35 下真实可达的只有 8.41% 覆盖。原先的高覆盖是反序置信度
   把大批候选错误地抬到门限之上的结果。
2. **整条曲线上 lift 都 ≤ +1.41pp。** 8.41% 那个点精度 88.89% 看着漂亮，
   但同一批 9 条上一律报 L2 也是 88.89%，lift 恰好为 0——门限只是挑出了容易的 case。
3. 0.45 处覆盖 93.46%、人工降到 6.54%，代价是精度 59.00% **低于同子集多数类 64.00%**。
   平衡召回确实升到 0.3534（L1 召回 34.4%），说明系统有一点判别力，但不足以抵偿错误。

#### 按类别校准门限：机制正确，但在这份数据上是空操作

拟合结果是 L1 与 L2 的下界相同（0.5625/0.5625），实测只多答 1 条。
原因由探针直接给出：**L1 候选在任何门限下都达不到风险目标**——
作答数 ≥ 10 时 L1 的最高纯度只有 31.43%（折外）/ 37.50%（留一），
而 L1 先验就是 30.43%。L1 候选几乎不携带先验之外的信息。

按类别校准因此如实拒绝为 L1 开门，这正是它该做的事（`labels_missing_target` 会记下这一类）。
把它记在这里是为了避免后来者重复这条路：**统一门限确实会结构性地偏向多数类，
但在本数据上真正的瓶颈不是门限的表达能力，而是候选生成器没有 L1 信号。**
迭代 1 那句「统一门限必然导致退化」需要修正为：门限只是把上游的无信号如实反映出来了。

#### 那么当前证据到底能不能定根因：可辨识上限

三轮负面结果都指向同一个没被回答的问题。`scripts/measure_identifiability_ceiling.py`
用不依赖任何模型的办法量它：签名相同的 case 在这个特征空间里**不可区分**，
任何算法（树、最近邻、LLM、人）看到的输入都一样。

第一个数字是个陷阱：精确签名的准确率上界高达 97.52%。但 147 个签名里有 137 个
只出现一次，「每个签名取多数类」等于逐条背答案；配套的一个部署侧数字说明了它的
无用——**测试集只有 15/107（14.02%）条的证据签名在训练集出现过**。
这也解释了 N5a（精确匹配）为什么只覆盖 15 条：证据签名近乎唯一，精确匹配无法推广。

所以要在能推广的粒度上量。按 Jaccard 相似度统计「证据相似的两条 case 是否同根因」：

| 相似度 ≥ | case 对 | 同根因 | 随机（Σp²） | lift |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 1776 | 56.59% | 48.40% | +8.19pp |
| 0.70 | 355 | 64.79% | 48.40% | +16.39pp |
| 0.90 | 50 | 74.00% | 48.40% | +25.60pp |
| **1.00（证据完全相同）** | **20** | **70.00%** | 48.40% | +21.60pp |

两个方向都要读出来：

1. **证据相似度确实携带真实信号**，且随相似度单调上升（+8.19pp → +25.60pp）。
   这支持证据图检索这条路线的前提，也说明浅决策树的失败有具体原因——
   它把 case 压到 5 个粗叶子上，恰好丢掉了携带信号的细粒度相似结构。
2. **但证据完全相同的 20 对 case 里有 6 对（30%）根因不同。** 这是当前遥测口径下
   一个模型无关的硬上限：观测一致而根因不同，任何算法都无法区分它们。

把它做成预测器（留一法、按相似度加权投票，置信度取邻居一致度，
因此不存在本轮修掉的反序问题）后，曲线与上面完全自洽：

| 相似度 ≥ | 覆盖率 | 精度 | 同子集多数类 | lift | 平衡召回 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 93.79% | 64.90% | 64.24% | +0.66pp | 0.3773 |
| **0.70** | **80.75%** | **69.23%** | 66.92% | **+2.31pp** | **0.4326** |
| 0.80 | 59.01% | 69.47% | 73.68% | −4.21pp | 0.3831 |
| 0.90 | 27.33% | 72.73% | 79.55% | −6.82pp | 0.3968 |

相似度投票在 0.70 处取得本项目至今**最好的平衡召回 0.4326**（门限路线最好是 0.3534，
而且是在精度低于基线的 0.45 处取得的），lift +2.31pp 也高于门限路线的 +1.41pp。
再往上精度虽然继续升到 72.73%，但同子集多数类升得更快（79.55%）——
高相似邻域本身就富集多数类，这是所有「提高门限就提高精度」的假象的共同来源。

三条独立路线的精度平台高度一致：N5a 精确匹配 73.33%、相似度 ≥0.9 投票 72.73%、
折外 SOP 最纯子集 75.76%。它们与「同证据不同根因 30%」相互印证，
共同给出一个可对外陈述的结论：**在当前遥测口径下，L1/L2/fiber 归因的精度上限
约在 70~75%，而一律报多数类是 62.6%——整个方法空间只有约 10pp 的可争取带宽。**
这个数字比任何单点准确率都更有参考价值：它告诉同行该把力气花在补采什么遥测上，
而不是继续调模型。

#### 本轮产出

- `rca_framework/decision.py`：`per_label_lower_bound`、`lower_bound_for`、
  `refine_per_label_bounds`（坐标上升，逐类风险达标），`simulate_gate`
  增加 `by_predicted_label` 与 `balanced_recall`。
- `rca_framework/knowledge.py`：`stratified_folds`、`_out_of_fold_sop_predictions`，
  门限标定与训练侧决策全部切到折外；`_loo_sop_predictions` 保留为反例样本。
- `scripts/probe_per_label_operating_points.py`：逐类风险-覆盖率曲线
  与「置信度排序方向」（组内 AUC，< 0.5 即反序）。
- `scripts/measure_identifiability_ceiling.py`：模型无关的可辨识上限、
  证据邻域一致性、留一法相似度投票曲线。产物 `artifacts/i2_identifiability_v2.json`。
- `tests/test_confidence_ordering.py`（5 条）、`tests/test_class_conditional_bounds.py`（4 条）。
- `python -m pytest -q` 为 **218 passed**。

#### 迭代 2 正式实验（`artifacts/i2_offline_sop_llm_risk040/`）

与迭代 1 同源（DeepSeek-R1-Distill-Qwen-32B、GPU 6/7、TP=2、seed 42、prompt v6、
manifest 161/107），改动三处：折外置信度、目标风险 0.40、开启按类别校准。
约束库为 v5（22 条，指纹 `6db9b1c80f98090d`）。

| 指标 | 迭代 1 | 迭代 2 |
| --- | ---: | ---: |
| 覆盖率 | 66.36% | 67.29% |
| 给结论时精度 | 70.42% | 69.44% |
| 同子集多数类 | 69.01% | 68.06% |
| lift | +1.41pp | **+1.39pp** |
| 平衡召回 | 0.2596 | **0.2596** |
| 弃答有效性 | 0.5278 | 0.5429 |
| 人工介入 | 6.54% | 6.54% |

**结果与迭代 1 实质相同。** 置信度修好了，但在 0.40 的风险预算下工作点落回同一处，
lift 仍然只有 +1.39pp，平衡召回一模一样。按类别校准给出的下界是
L1=0.4104、L2=0.0000——它把 L2 的门完全打开了。

这里得到本轮第二条可推广的设计规则：**风险预算必须紧于多数类的错误率，
否则「一律报多数类」本身就是可行解。** 本数据多数类准确率 62.6%，错误率 37.4%；
把目标选择性风险设成 0.40 > 0.374，等于在约束里默许了退化解，
逐类校准于是诚实地把 L2 的门限降到 0。要让门限真正起作用，目标风险必须 < 37.4%，
而那时（0.35）真实覆盖只有 8.41%。这两件事不能同时要。

#### LLM 侧：知识被够到了，但 79% 的回答被约束校验判废

73 条进入 LLM 的 case 中，只有 7 条一次通过校验，重写后再救回 8 条，
**58 条（79.5%）两次都不合规而弃权**（重写把 253 条 fatal 降到 198 条，
但其中 21 条 case 重写后违规反而更多）。违规分布指向三类系统性错误：

| 次数 | 违规 | 说明 |
| ---: | --- | --- |
| 113 | 引用的证据 token 家族与约束适用范围不匹配 | 主要是 C11（61）、C16（46）、C22（24） |
| 17 | 把 `effect=neutral` 的护栏当 `support` 用 | C5/C7/C11/C13 |
| 8 | `target` 方向错 | C16 要求 target=L2，模型写 L1 |

最后一类虽然次数最少，却是最要紧的：C16 说的是「某侧出现**接收**症状时，
指向的是**对端的发送链路**」，而模型写的是「L1 的 RxLOL/RxLOS 异常 → 支持 L1 接收链路异常」，
**把接收侧症状归给了接收侧自己**——这正是光链路定界里最典型的归因方向错误。
C16 就在 prompt 里，模型也确实去引用它了，但用反了方向。

第二类同样有物理意义：模型逐个 token 找一条约束来「支持」它，于是同一次回答里
出现「L2 接收功率偏低 → 支持 L2 异常」和「L2 发送功率偏高 → 支持 L2 正常」
两个互相矛盾的步骤，都挂在 C5/C7 这两条**中性护栏**上。C21 已经写明
「正常带内的电平高低不是归因证据」，但约束库没有给模型一个合法的位置去
安放「这一侧看起来正常」这类观察，模型只能挪用护栏。

这直接回答了「LLM 在有 knowledge 的情况下能不能分析测试集」：
**能够到知识（C16/C22 被高频引用），但用不对方向；当前失败不是因为知识不足，
而是因为知识没有把归因方向写成模型必须遵守的形式。**

#### 由此确定的迭代 3 方向

按优先级：

1. **把归因方向写进不可绕过的位置**。现在方向只存在于 C16 的文字描述与校验器的
   target 契约里（违规才发现）。应当在 prompt 顶部作为独立规则声明「接收侧症状
   指向对端发送链路」，并让校验器的方向类违规在重写提示里单独强调。
2. **给「该侧正常」一个合法出口**。扩写 C21 覆盖 rxpower/media_snr 的带内电平，
   使模型可以把这些观察显式标为 neutral 而不是挪用护栏做 support。
3. **候选生成器换成相似度加权邻居投票**（置信度 = 邻居一致度，天然无反序），
   门限设在相似度 0.70 附近。这是唯一测出来同时优于门限路线的路径
   （lift +2.31pp、平衡召回 0.4326）。
4. 目标选择性风险必须设在 37.4% 以下，并且始终与同子集多数类基线一起报，
   否则等于默许退化解。预期精度上限 70~75%，不要指望更多。
