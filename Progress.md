# nsdi-agent 重构进度

本文是 `nsdi-agent/` 的重构进度台账。后续 AI 写代码前必须先阅读 `AGENTS.md` 和本文，确认当前阶段、已完成改造、不可破坏基线和下一步任务。

## 1. 使用约定

状态图例：

- `未开始`：尚未动代码或文档。
- `进行中`：已有部分改动，但未通过该项验收。
- `已完成`：实现完成且通过对应验收。
- `已冻结`：作为兼容锚点保留，不应改动。
- `已放弃`：经确认不再执行。

更新规则：

1. 每次代码改动前，先看“当前阶段指针”和“回归门禁清单”。
2. 每完成一个模块或阶段，更新对应表格的状态、结果和日期。
3. 每次运行测试或实验，把命令、结果、产物路径写入“门禁运行记录”。
4. 每次引入行为变化，必须说明是否影响 legacy 基线。
5. 不要把 `/home/chenziang/nsdi/` 的状态写成当前进度；这里记录的是 `/home/chenziang/nsdi-agent/`。

## 2. 当前阶段指针

当前阶段：`阶段 1 - 观测字段扩展`

当前状态：`未开始`

阶段 0 已于 2026-08-06 完成，基线锁定已自动化，可以进入阶段 1。

下一步：

1. `types.py` 增加 `EvidenceItem`、`Verdict` 等协议类型，只增不改。
2. 新建 `evidence.py`，把 `EvidenceItem` 聚合为独立互证 / 同源一致 / 冲突 / 无证据四类。
3. `graph.py` 增加 `prior_only`、`score_composition`、`classify_coverage`，并把 IDF 检索拆到 `retrieval.py`。
4. `rules.py` 增加 `support_tier` 与 `evidence_items`，用于回答 fiber 的 28 条规则里有多少来自 `minority_fallback`。
5. 每步都必须让 `tests/test_baseline_lock.py` 保持全绿，即 legacy 分数逐 case 不变。

## 3. 阶段进度表

| 阶段 | 内容 | 是否允许改变 legacy 结果 | 依赖 | 状态 | 验收 |
| --- | --- | --- | --- | --- | --- |
| 0 | `RuntimeConfig` + `build_case_context` 去重 + 基线锁定测试 | 否，必须逐 case 一致 | 无 | 已完成 | pytest 15 passed；58/85；`predictions.json` 与基线字节一致 |
| 1 | `types` 扩展、`EvidenceItem.origin_anomalies`、`evidence.aggregate_evidence`、`graph.classify_coverage` / `prior_only`、`rules.support_tier` | 否，只增加观测字段 | 阶段 0 | 未开始 | legacy 分数逐 case 一致；新增字段可统计 |
| 2 | `anomaly.evidence_status`、`lane_pairs`、`lane_directional_loss` | 可能改变，只能先影子模式运行 | 阶段 1 | 未开始 | 旧 `anomaly_id` 集合不变；lane signature 单独报触发数 |
| 3 | `rca_framework/agent/` + `agent-diagnose`，`backend=none` 的确定性 Agent | 否，legacy 默认不变 | 阶段 1 | 未开始 | Agent 控制流跑通；legacy 仍默认；trace 可写 |
| 4 | `llm/` 子包拆分 + Agent prompt + `abstain` 出口 + 选择性评估 | 是，产生新评估口径 | 阶段 2、3 | 未开始 | coverage / precision_at_coverage 可报；legacy 可回归 |
| 5 | `skills/` + playbook + trace 回放 | 是，引入 Skill 版本 | 阶段 4 | 未开始 | skill 版本写入 manifest；playbook 命中率与误导率可统计 |

## 4. 逐模块进度表

### 4.1 现有模块

| 模块 | 初始行数 | 当前角色 | Agent 化目标 | 所属阶段 | 状态 | 验收标准 |
| --- | ---: | --- | --- | --- | --- | --- |
| `rca_framework/types.py` | 84 | `ROOT_CAUSES`、`Anomaly`、`CaseEvidence`、分数归一化 | 增加 `DECISIONS`、`EvidenceItem`、`Verdict` 等协议类型 | 1 | 未开始 | 旧类型行为不变；新类型有单测 |
| `rca_framework/data.py` | 332 | 数据清单、脱敏、L1/L2 归一化、数据集加载 | 补 `case_side_mapping` 与 `evidence_manifest` 元数据 | 1 | 未开始 | 现有数据集无需重生成；旧 loader 签名不变 |
| `rca_framework/anomaly.py` | 262 | 阈值拟合、异常提取、方向性损耗 | 增加 `evidence_status`、lane pair、lane directional signature | 2 | 未开始 | 旧 `anomaly_id` 逐 case 不变；新 signature 影子统计 |
| `rca_framework/graph.py` | 334 | KG 学习、路径评分、feature rules、RAG 检索 | 增加 `prior_only`、`score_composition`、`classify_coverage`，拆检索 | 1 | 未开始 | 旧 `scores` 数值逐 case 一致；`prior_only` 可统计 |
| `rca_framework/graph.py` 确定性修正 | 336 | 同上 | 固定 `idf` 键序与检索求和顺序 | 0 | 已完成 | 不同 `PYTHONHASHSEED` 下 artifacts 字节一致 |
| `rca_framework/rules.py` | 182 | 互斥符号规则学习与匹配 | 增加 `support_tier` 与 `evidence_items` | 1 | 未开始 | `rule_overlap` 仍为 0；旧 match 分数一致 |
| `rca_framework/llm.py` | 218 | prompt、schema、解析、后端、LLM 路打分 | 拆为 `llm/` 子包，LLM 只输出结构化推理和动作 | 4 | 未开始 | legacy import 与 `reason_many` 兼容；59/85 可回归 |
| `rca_framework/fusion.py` | 100 | legacy 两路融合与冲突仲裁 | 冻结 `fuse_results`，新增证据聚合另放 `evidence.py` | 1 | 已冻结 | `fuse_results` 不改；legacy 58/85 不漂移 |
| `rca_framework/pipeline.py` | 233 | fit / infer / evaluate / save / load、reasoner 缓存 | 拆出 `KnowledgeBundle`、`RuntimeConfig`、`RCASession`、`Evaluator` | 0 | 已完成 | `RCAPipeline.load` 兼容；legacy 输出逐 case 一致 |
| `rca_framework/cli.py` | 169 | `prepare` / `train-evaluate` / `infer` 入口 | 新增 Agent 入口与 `--policy`，默认 `legacy` | 3 | 未开始 | 旧参数、默认值和命令行为不变 |
| `rca_framework/__init__.py` | 5 | 包初始化 | 必要时只做兼容导出 | 4 | 未开始 | 旧 import 不破 |

阶段 0 完成后 `pipeline.py` 为 235 行，新增 `CaseContext`、`build_case_context`、`finalize_prediction` 三个装配单元，`infer` 与 `evaluate` 不再各自复制一份装配代码。`cli.py` 改为通过 `runtime_from_args` 传 `RuntimeConfig`，命令行参数、默认值与 `run_manifest.json` 内容均未变。

### 4.2 待建模块

| 目标路径 | 目标职责 | 所属阶段 | 状态 | 验收标准 |
| --- | --- | --- | --- | --- |
| `rca_framework/runtime.py` | `RuntimeConfig`：推理期运行参数对象，不进入 `model.json` | 0 | 已完成 | 旧 kwargs 调用等价；未知参数报错 |
| `tests/test_baseline_lock.py` | deterministic 基线锁定，逐 case prediction / 分数 / anomaly_id / 模型产物 | 0 | 已完成 | 8 个测试，缺数据集时自动 skip |
| `rca_framework/evidence.py` | 聚合 `EvidenceItem`，区分独立互证、同源一致、冲突、无证据 | 1 | 未开始 | `same_source_agreement` 可统计 |
| `rca_framework/retrieval.py` | 从 `graph.py` 拆出 IDF 加权 Jaccard 检索，支持 `hide_labels` | 1 | 未开始 | legacy 检索结果兼容；`hide_labels=True` 不泄漏标签 |
| `rca_framework/agent/protocol.py` | AgentAction、ToolCall、ToolResult、Verdict 控制流协议 | 3 | 未开始 | 输入输出可 JSON 序列化 |
| `rca_framework/agent/tools.py` | 工具注册表与 9 个无状态工具包装 | 3 | 未开始 | 每个工具有 schema 契约测试 |
| `rca_framework/agent/sufficiency.py` | 证据充分性门限 | 3 | 未开始 | insufficient / weak / sufficient 路径可单测 |
| `rca_framework/agent/policy.py` | decide / request_evidence / abstain，含 legacy 兼容策略 | 3 | 未开始 | `abstain` 路径可达；legacy policy 不改变结果 |
| `rca_framework/agent/loop.py` | Plan -> Call -> Check -> Decide 控制循环 | 3 | 未开始 | 最大步数与重复调用检测生效 |
| `rca_framework/agent/trace.py` | JSONL trace 写入与回放 | 3 | 未开始 | trace 可回放，字段稳定 |
| `rca_framework/agent/playbook.py` | `rca-playbook` signature 匹配与回退 | 5 | 未开始 | 命中率、误导率可统计 |
| `rca_framework/llm/__init__.py` | 保持旧 import 兼容 | 4 | 未开始 | `from rca_framework.llm import PathLLMReasoner` 仍可用 |
| `rca_framework/llm/backend.py` | vLLM / transformers / none 后端加载与生成 | 4 | 未开始 | 加载期参数与推理期参数分离 |
| `rca_framework/llm/prompts.py` | legacy prompt、layered prompt、agent prompt | 4 | 未开始 | `build_path_prompt` 一字不改 |
| `rca_framework/llm/protocol.py` | legacy schema、layered schema、Agent action / verdict schema | 4 | 未开始 | legacy `LLM_OUTPUT_SCHEMA` 一字不改 |
| `skills/rca-domain/SKILL.md` | 标签定义、物理判据、指标单位、弱证据清单 | 5 | 未开始 | 版本号可读取并写入 manifest |
| `skills/rca-workflow/SKILL.md` | 诊断流程、弃权准则、缺失证据到补采动作映射 | 5 | 未开始 | 门限引用代码常量，不复制散落 |
| `skills/rca-playbook/SKILL.md` | 历史故障 signature 索引与匹配约定 | 5 | 未开始 | `cases/*.md` 可被 playbook 读取 |

## 5. 回归门禁清单

每次阶段完成前必须检查：

- [x] `python -m pytest -q` 全绿，包含基线锁定测试。
- [x] `--policy legacy --backend none` 或默认 legacy `--backend none` 输出 58/85。
- [x] 逐 case prediction 与 `artifacts/organized_rca_v2_60_40_seed42_baseline/predictions.json` 完全一致。
- [ ] 有 GPU 时，DeepSeek-32B vLLM legacy 结果保持 59/85。
- [x] `RCAPipeline.load` 可读取现有 `artifacts/*/model`。
- [x] `rules.overlap_audit` 的 total overlap 仍为 0。
- [ ] `run_manifest.json` 在引入 policy / skill 后记录 `policy`、`skill_versions`、`trace_path`、`coverage_policy`。

前 6 项中除 GPU 项外都已由 `tests/test_baseline_lock.py` 自动覆盖，直接 `python -m pytest -q` 即可复核，不必手工跑 CLI 再肉眼比对。勾选状态代表阶段 0 收尾时的复核结果，进入新阶段后应重新执行。

### 门禁运行记录

| 日期 | 阶段 | 命令 | 结果 | 产物 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 2026-08-06 | 初始化 | `python -m pytest -q` | 7 passed | 无 | 复制后基础测试通过 |
| 2026-08-06 | 初始化 | `python -m rca_framework.cli train-evaluate --data-dir datasets/organized_rca_v2_stratified_60_40_seed42 --train-size 126 --output-dir artifacts/copy_verify --backend none` | 58/85，accuracy 68.24%，`rule_overlap=0`，`label_leakage=false` | `artifacts/copy_verify/`（已删除，不入库） | 逐 case prediction 与复制基线完全一致 |
| 2026-08-06 | 0 | `python -m pytest -q` | 15 passed（原 7 + 基线锁定 8） | 无 | 基线锁定测试首次全绿 |
| 2026-08-06 | 0 | `python -m rca_framework.cli train-evaluate ... --output-dir artifacts/gate_stage0 --backend none` | 58/85，accuracy 68.24%，`rule_overlap=0`，`label_leakage=false` | `artifacts/gate_stage0/`（不入库） | `predictions.json`、`evaluation_summary.json`、`run_manifest.json` 与基线字节一致 |
| 2026-08-06 | 0 | 同一命令分别用 `PYTHONHASHSEED=1` 与 `PYTHONHASHSEED=98765` 各跑一次后 `diff -rq` | 两次产物全部字节一致 | 无 | 修正前 `model.json` 的 `idf` 键序每次都不同 |

## 6. 不可变基线数字

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
| organized_data fiber 有效 case | 14，总训练 8 | 既有缺陷分析 |

## 7. 待测数字与未解决问题

这些数字目前还没有在代码中被自动统计，是 Agent 化第一阶段的重要观测目标：

| 问题 | 当前状态 | 需要的落点 |
| --- | --- | --- |
| 82 条 deterministic `agreement` 中有多少是 `same_source_agreement` | 未知 | `rca_framework/evidence.py` + trace 统计 |
| `fiber` 的 28 条规则中有多少来自 `minority_fallback` | 未知 | `rules.support_tier` / `rules.evidence_items` |
| lane 级 `tx_ok_rx_down`、`tx_down`、`bidirectional_same_lane` 等 signature 触发数 | 当前旧 `directional_loss` 为 0 | `anomaly.lane_directional_loss` 影子统计 |
| `prior_only == True` 的 case 数 | 预期约 22 | `graph.query` 增加说明字段后复核 |
| `covered_pair` / `covered_singleton` / `covered_exemplar` / `partial` / `uncovered` 分布 | 未知 | `graph.classify_coverage` |
| 选择性分类在同等覆盖率下是否优于朴素置信度截断 | 未知 | `Evaluator` coverage-accuracy 曲线 |

## 8. 变更日志

| 日期 | 修改人 | 阶段 | 文件 / 模块 | 变更摘要 | 验证结果 |
| --- | --- | --- | --- | --- | --- |
| 2026-08-06 | AI | 初始化 | 仓库复制、`AGENTS.md`、`Progress.md` | 从 `nsdi/` 复制轻量活动树；写入开发说明与进度台账 | 7 passed；58/85；逐 case prediction 一致 |
| 2026-08-06 | AI | 初始化 | `.gitignore`、git 仓库 | `nsdi-agent/` 建为独立 git 仓库并提交初始基线，只跟踪两个回归 artifacts；`/home/chenziang/.gitignore` 不再跟踪本目录 | 807 文件入库，`.git` 5.1M |
| 2026-08-06 | AI | 0 | `runtime.py`、`pipeline.py`、`cli.py`、`__init__.py` | 新增 `RuntimeConfig`（推理期参数对象化，默认值不变，旧 kwargs 仍可用）；抽出 `CaseContext` / `build_case_context` / `finalize_prediction`，`infer` 与 `evaluate` 共用同一套装配；reasoner 缓存改用 `RuntimeConfig` 做键 | 15 passed；58/85；`predictions.json` 与基线字节一致 |
| 2026-08-06 | AI | 0 | `tests/test_baseline_lock.py` | 新增 8 个基线锁定测试：切分、summary、逐 case prediction / confidence / 两路分数 / anomaly_id、模型产物可加载且等价、规则互斥、`idf` 键序确定 | 15 passed |
| 2026-08-06 | AI | 0 | `graph.py` | 固定 `idf` 键序与 `retrieve` 的 IDF 求和顺序，消除集合迭代顺序带来的不可复现；`scores` 与 legacy 逐 case 一致，`model.json` 的 `idf` 键序由随机变为排序，schema 与数值不变 | 不同 `PYTHONHASHSEED` 下产物字节一致；58/85 不变 |
