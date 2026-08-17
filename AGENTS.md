# nsdi-agent 开发说明

本文件是 `nsdi-agent/` 的项目级开发约束。任何后续 AI 或人工开发在修改代码前，必须先阅读本文件、`Progress.md` 和 `Validation.md`。

三份文档的分工：本文件写约束，`Progress.md` 写已完成的事和已测出的数字，`Validation.md` 写还需要人拍板、需要外部输入或需要新数据才能验证的事。任何待确认事项都不允许只留在对话或注释里，必须进 `Validation.md`，并且必须写明未确认前代码采用什么默认取值。

## 1. 项目定位

`nsdi-agent/` 是 `/home/chenziang/nsdi/` 的 Agentic AI 重构树，用于把当前光链路 RCA v2 从固定双轨流水线改造成以证据图历史匹配为主干、按相似度分流处理、可回灌自迭代的诊断系统。

当前正式标签仍然只有三类：

- `L1`：400G 端口或其设备侧根因。
- `L2`：200G 端口或其设备侧根因。
- `fiber`：L1 与 L2 之间的光纤 / 链路介质根因。

本项目不再沿用"神经层 + 符号层"双轨分类器叙事。现有证据表明，继续堆叠 prompt 或融合权重来提高 100% 覆盖率下的强制三分类 accuracy，难以突破可用特征空间的上限：organized 60/40 DeepSeek-32B vLLM 为 59/85（69.41%），全特征 RandomForest 5 折天花板约 70.14%，且 `fiber` precision / recall / F1 仍为 0。

新框架的论文口径与工程目标是：

1. 以证据图历史匹配作为主干，先判断当前 case 是否命中已确认历史模式。
2. 按 N4 相似度路由分为完全匹配、部分匹配和低匹配三条处理路径。
3. 对低匹配 / 证据冲突 case，引入光模块物理约束与专家排障 SOP，约束 LLM 逐步推理。
4. 在低置信度或证据不足时输出通用排查建议或人工介入，而不是多数类硬猜。
5. 把人工确认后的 case 回灌到证据图和特征字典版本中，使已见模式覆盖率随 case 演化提升。

论文动机应表述为：现有网络故障研究没有针对光模块物理属性做专项优化。核心价值应表述为：冷启动阶段复用专家经验，运行后随历史 case 增长自迭代。

### 1.0 个人整体思路硬约束 / Loop 门禁

`docs/个人整体思路.md` 是当前 `nsdi-agent/` 的主架构约束。任何 Loop 实验、正式实验或主链路改造都必须先对齐这条流程。允许把全覆盖三分类作为 N6 置信度阈值标定的观测口径，但不得只用强制三分类 accuracy 或 lift 替代架构目标。

- Loop 启动前必须重读 `docs/个人整体思路.md`，并说明本轮只优化以下目标之一：证据图形态、证据链路 / 归因路径匹配、物理约束下的关键证据判定、专家 SOP 约束下的 LLM 推理校验。
- 主链路固定为：证据图历史匹配 → N5a 完全匹配复用历史证据链并交给 LLM 给出独立候选与置信度 → N5b 部分匹配用物理约束判断缺失证据是否关键并交由 LLM 仲裁 → N5c 冷启动注入专家 SOP 约束 LLM 推理 → 低置信度降级。
- `expert.py` 的方向表、`learned-sop-v1`、数值决策树和多数类先验都不能替代证据图匹配主干。它们只能作为物理语义、对照基线、报告字段或低匹配分支的统计先验。
- M9 正式默认 `candidate_order=("branch",)`；不得默认加入 `expert` 或 `sop` 作为自动终裁候选。需要比较 expert / SOP 时必须作为消融或对照实验显式开启。
- 物理约束库只能保存真实物理关系或量测契约。训练集区间、命中率、Wilson 下界和标签分布必须留在统计 / 决策树层，不能写成物理约束。
- 本阶段不推进 N8 自动回灌或自迭代。测试 badcase 不能反向修改知识包、SOP、约束、阈值、向量或证据图；人工确认回灌只保留接口和文档语义。

### 1.0.1 实验归档与 Bad Case 门禁

每一次 Loop 实验都必须围绕测试 bad case 做假设、改动和验证，并在 `experiments/<YYYYMMDD>_<short-name>/` 下沉淀完整报告。没有 HTML 报告、没有主流程图差异说明、没有逐 case bad case 分析的运行，不算有效实验。

每次实验必须满足：

- 报告首页必须放当前主流程图，并标注本轮只改了哪一处；允许的改动类别只有：证据图约束 / schema、阈值或路由、大模型 prompt、代码 bug fix。
- 报告必须记录当前证据图是什么、物理约束是什么、专家 SOP / learned SOP 是否变化、prompt 版本、阈值、M9 candidate order、N8 是否冻结。
- 正确 case 要按分支和关键步骤归纳：例如历史经验相近、N5a 复用成功、N5b 关键证据判断正确、N5c SOP 步骤有效。
- Bad case 是报告重点。每个 bad case 必须写明失败发生在哪一步、错因是证据图缺边/约束不对/阈值不对/prompt 不对/代码 bug/数据不可辨识/疑似标签问题，并给出下一步动作。
- 如果怀疑测试 label 有问题，必须写入 `label_suspects.json`，后续实验持续保留，等待人工进一步关注。
- 如果当前训练集、专家规则、SOP 和测试可见证据都无法支持提升，该 case 必须写入 `irreducible_cases.json`。后续实验保留它，但不要继续围绕它刷指标。
- 实验报告必须包含逐 case 分析；正确 case 可统计归纳，bad case 必须逐条解释。

### 1.1 RCA v2 / l2fixed 当前主线

截至 2026-08-10，新增一条基于 `datasets/rca_v2_l2fixed` 的 v2 重构主线：

- `datasets/rca_v2_l2fixed/_metadata/manifest.json` 是新主线的唯一 split 契约；
  使用 seed=42、60/40 分层切分，train 161 / test 107。
- legacy organized 数据集、`python -m rca_framework.cli` 和 58/85 只作为回归锚点保留；
  不允许把 l2fixed 数字和 organized 126/85 数字直接混表比较。
- 新框架实验入口仍是 `scripts/evaluate_routing.py`，l2fixed v2 实验必须显式带
  `--manifest-split --feature-profile v2`，需要数据归纳 SOP 时再加 `--learned-sop`。
- `learned-sop-v1` 是训练集标签归纳的浅层决策树，不是专家 SOP。报告里必须展示叶节点支持数、
  标签分布和 Wilson 下界；低支持叶不得进入最终结论。
- `skills/` 已拆为 `rca-domain`、`rca-constraints`、`rca-sop`、`rca-evidence-graph`、
  `rca-workflow`。修改约束、SOP 或图 schema 后必须运行 `python scripts/render_rca_skills.py`。

## 2. 与 `nsdi/` 的差别

`/home/chenziang/nsdi/` 是只读参照树，用于保存当前 v2 基线、历史探索和实验记录。`/home/chenziang/nsdi-agent/` 是唯一活动开发树，后续证据图 Agentic AI 代码都应在这里完成。

开发时遵守以下边界：

- 不在 `nsdi/` 中修改代码、脚本、数据或 artifacts。
- 不让 `nsdi-agent/` 的脚本默认写入 `nsdi/`。如需读取历史实验或归档，只能显式只读引用。
- 不把 `nsdi/before/` 的旧探索代码混入 v2 主路径，除非明确作为对照基线引用。
- 不把 `nsdi/artifacts/layered_injection_20260805/` 当作活动实现维护；它只保留实验结果与回放价值。

本仓库是按轻量复制策略初始化的，并非 `nsdi/` 的全量镜像。

已复制：

- `rca_framework/`
- `docs/`
- `datasets/`
- `organized_data/`
- `tests/`
- `scripts/`
- `archive/`
- `pytest.ini`
- `README.md`
- `70.json`
- `artifacts/organized_rca_v2_60_40_seed42_baseline/`
- `artifacts/organized_rca_v2_60_40_seed42_deepseek32b_vllm/`

未复制：

- `before/`：旧探索代码、旧报告、旧 outputs 和 saved_methods，约 25M。
- 其余历史 artifacts，包括 `artifacts/layered_injection_20260805/`，约 40M。
- 缓存目录，如 `__pycache__/`、`.pytest_cache/`。

如后续确需查看未复制内容，从 `/home/chenziang/nsdi/` 只读引用，不要复制回活动树形成两份漂移实现。

## 3. Agentic AI 叙事

新框架把原来的 KG + 双轨分类器降级为知识来源和 legacy 回归锚点，主链路围绕证据图历史匹配、相似度分流、约束推理和结果回灌展开。

| 步骤 | 环节 | 输入 | 产出 |
| --- | --- | --- | --- |
| N1 | 新告警 / case 触发与证据标准化 | 告警、端口、链路上下文、多源遥测 | `evidence pack` |
| N2 | 可解释特征抽取 | `evidence pack` | 带物理含义的稀疏 case vector |
| N3 | 证据图历史匹配 | case vector、历史 case 证据图 | Top-N 候选、相似度、覆盖率、缺失 / 冲突证据 |
| N4 | 分流路由 | Top-N 相似度分布 | N5a / N5b / N5c 路径 |
| N5a | 完全匹配 | `sim = 100%` 的历史 case | 历史证据链复用候选 |
| N5b | 部分匹配（已见模式） | `70% <= sim < 100%` 的候选与缺失证据 | 补证据、LLM 仲裁或降置信结论 |
| N5c | 低匹配（未见模式） | `sim < 70%` 的证据包、约束库、SOP | 约束内推理、补采清单或降级建议 |
| N6 | 置信度决策 | N5 输出 | 最终根因、低置信度降级或人工介入 |
| N7 | 报告生成 | 根因、证据链、路径来源 | 排障报告 / 工单 |
| N8 | 证据图回灌 | 人工确认结果 | 证据图版本、特征字典版本、审计记录 |

N4 分流阈值按最新画板执行：

- `sim = 100%`：N5a 完全匹配，沿用命中历史 case 的证据链，但必须先校验该 signature 是否标签纯净。
- `70% <= sim < 100%`：N5b 部分匹配，缺非关键证据时补齐，缺关键证据或候选冲突时触发 LLM 仲裁。
- `sim < 70%`：N5c 低匹配，走物理约束库 + 专家 SOP 的通用排障 prompt，并逐步做合规性校验。

```mermaid
flowchart TB
    n1["N1 证据构建与标准化"] --> n2["N2 可解释特征抽取"]
    n2 --> n3["N3 证据图历史匹配<br/>Jaccard Top-N"]
    n3 --> n4{"N4 分流路由"}
    n4 -->|"sim = 100%"| n5a["N5a 完全匹配"]
    n4 -->|"70% <= sim < 100%"| n5b["N5b 部分匹配 已见模式"]
    n4 -->|"sim < 70%"| n5c["N5c 低匹配 未见模式"]
    constraints["物理约束库 M5"] --> n5b
    constraints --> n5c
    sop["专家排障 SOP M6"] --> n5c
    n5a --> n6{"N6 置信度阈值"}
    n5b --> n6
    n5c --> n6
    n6 -->|"确认结果"| n7["N7 报告生成"]
    n6 -->|"低置信度"| degrade["降级策略<br/>通用建议或人工介入"]
    n7 --> n8["N8 证据图回灌"]
    degrade --> n8
    n8 --> graph[("证据图 历史 case")]
    graph -.->|"历史 case 索引"| n3
```

### 3.1 目标目录结构

最终目标不是一次性重写所有代码，而是在保留 legacy 回归锚点的前提下逐步引入以下结构：

```text
rca_framework/
  features/
    dictionary.py      # M1 特征字典 v1：维度 / 物理含义 / 单位 / 取值域 / 版本号
    extractor.py       # M1 证据 -> 可解释稀疏特征向量
  evidence_graph/
    store.py           # M2 证据图构建与索引
    match.py           # M3 Top-N 检索打分 + 缺失 / 冲突证据清单
    router.py          # M4 分流路由，100% / 70% 阈值可配置
  constraints/
    library.py         # M5 物理约束库，结构化 + prompt 化
    checker.py         # M7 可执行断言校验器
  branches/
    exact.py           # N5a 完全匹配处理器
    partial.py         # N5b 部分匹配处理器
    general.py         # N5c 低匹配通用排障处理器
  llm/
    __init__.py
    backend.py         # M8 vLLM / transformers / none 后端
    prompts.py         # M8 固定 prompt 模板
    protocol.py        # M8 输出 schema
  decision.py          # M9 置信度与降级策略
  report.py            # M10 报告生成
  feedback.py          # M11 证据图回灌闭环
scripts/
  run_ablation.py      # M12 消融 / 基线批跑
skills/
  rca-constraints/SKILL.md  # M5 prompt 化约束
  rca-sop/SKILL.md          # M6 专家排障 SOP
```

### 3.2 证据图回灌闭环

反馈学习不再描述为"把样本回灌训练集"或"重训决策树"这一条路，而是描述为 N8 证据图回灌闭环：

1. 每条人工确认 case 必须抽取必要证据、缺失证据、冲突证据、排除条件、处置动作，并写入证据图版本。
2. 完全匹配命中但误导时，优先补充排除条件或细化特征字典，不直接删除历史 case。
3. 低匹配模式中反复使用但未显式编码的物理判断，经专家审核后沉淀为 `constraints` 或 SOP。
4. 特征字典、证据图、约束库、SOP、prompt 模板都必须有版本号；实验结果必须在 `run_manifest.json` 里记录这些版本，否则不同版本结果不可比较。

## 4. 初始仓库事实

复制初始化时，活动代码位于 `rca_framework/`，合计 1919 行：

| 文件 | 行数 | 初始职责 |
| --- | ---: | --- |
| `rca_framework/data.py` | 332 | 数据清单、脱敏、L1/L2 归一化、数据集加载 |
| `rca_framework/anomaly.py` | 262 | 阈值拟合、异常提取、方向性损耗 |
| `rca_framework/graph.py` | 334 | label-centered anomaly KG、路径评分、feature rules、RAG 检索 |
| `rca_framework/rules.py` | 182 | 互斥符号规则学习与匹配 |
| `rca_framework/llm.py` | 218 | KG/RAG prompt、schema、解析、LLM 后端、LLM 路打分 |
| `rca_framework/fusion.py` | 100 | 两路加权融合、冲突仲裁、证据整理 |
| `rca_framework/pipeline.py` | 233 | fit / infer / evaluate / save / load、reasoner 缓存 |
| `rca_framework/cli.py` | 169 | `prepare` / `train-evaluate` / `infer` 入口 |
| `rca_framework/types.py` | 84 | 基础类型、`ROOT_CAUSES`、分数归一化与排序 |
| `rca_framework/__init__.py` | 5 | 包初始化 |

测试初始状态：

- `tests/test_data_pipeline.py`：2 个测试。
- `tests/test_graph_rules.py`：2 个测试。
- `tests/test_pipeline_and_fusion.py`：3 个测试。
- 合计 7 个测试，168 行。

可靠基线：

| 基线 | 结果 | 备注 |
| --- | --- | --- |
| organized 60/40 deterministic baseline | 58/85，accuracy 68.24% | `backend=none`，`fiber` recall 0 |
| organized 60/40 DeepSeek-32B vLLM | 59/85，accuracy 69.41% | 85 条有效 LLM 输出，`fiber` recall 0 |
| 同一测试集多数类基线 | 55/85，accuracy 64.71% | L2 majority |
| 全特征 RandomForest 5 折天花板 | 约 70.14% | `fiber` precision / recall / F1 均为 0 |

已知边界：

- 21/85 测试 case 提取零异常并回退到 L2 先验。
- `directional_loss` 与 `bidirectional_loss` 在现有 artifacts 中没有触发。
- `bias`、`Temperature`、`Voltage`、`alarm_name`、`vendor` 没有表现出稳定可用的类别分离。
- `fiber` 在 `organized_data` 中只有 14 条有效 case，60/40 切分下训练集只有 8 条。
- 当前系统不应被描述为解决了 fiber RCA；它只是略高于 L2 多数类基线。

阶段 1 观测后新增的三条可复核事实（数字见 `Progress.md` 第 7 节，由测试锁定）：

- 82 条 `agreement` 中只有 2 条是独立互证，58 条同源一致，22 条没有 case 特异证据。
  不要再把 `decision_status == "agreement"` 当作两路互相确认的依据。
- 22 条 case 的 `score_composition.prior_floor` 精确等于 1.0，它们的"候选分布"就是训练集类别先验。
- `fiber` 的 28 条符号规则每条只有 2 个训练 case 支持，全部为 `low_support`；它们不是
  `minority_fallback` 产生的，而是达标规则本身就只有这个支持度。

现有代码中可复用到新框架的资产：

- `rca_framework/anomaly.py` 的 `extract_evidence` 产出 `anomaly_id`，已经是 M1 的可解释稀疏特征雏形，但分辨率不足。
- `rca_framework/retrieval.py` 的 `retrieve` 已经是 M2 / M3 所需的 IDF 加权 Jaccard Top-N 内核。
- `rca_framework/graph.py` 的 `CoverageReport.max_retrieval_similarity` 可直接用于 N4 分流统计与阈值标定。
- 未提交的 lane 级改动（`lane_pairs`、`lane_directional_loss`、`lane_loss_report`、`EVIDENCE_STATUSES`、影子统计）应重新归属到 T1 / T3，而不是按旧阶段 2 冻结，因为它是提升 signature 分辨率的直接候选。

必须写入后续验收的新实测事实：

- 现有 126/85 切分按画板阈值分流：N5a 46 条、N5b 8 条、N5c 31 条。
- N5a "直接沿用历史结论"只有 29/46，accuracy 63.04%，低于 legacy 68.24% 和 L2 多数类 64.71%；并列 case 多数投票同样为 63.04%，oracle 上界为 86.96%。
- 训练集 126 条 case 只有 40 个不同 signature，7 个混合标签 signature 组覆盖 83 条训练 case；空 signature 组 31 条，标签分布为 L2:20 / L1:10 / fiber:1。
- N5b 只有 8 条，最近邻标签命中率 37.5%，暂不足以支撑消融结论。
- N5c 中 22 条 `sim = 0.0` 且零异常，是 N6 降级或人工介入的首选人群。

结论：不能照搬现有 `anomaly_id` 集合作为最终特征字典 v1。T1 的核心验收不是"写出特征表"，而是量化证明 signature 分辨率提高。

## 5. 开发铁律

### 5.1 必须冻结或版本化

除非用户明确要求破坏兼容并重建全部 artifacts，否则不要修改：

- `ROOT_CAUSES` 的元素和顺序。
- 脱敏算法与 L1/L2 归一化规则。
- legacy `model.json` 的现有 schema。
- legacy CLI 默认路径的语义。

以下内容只在 legacy 路径中冻结，新框架可以新建替代实现，但不得悄悄改变旧入口行为：

- `anomaly_id` 现有字符串格式。
- `fusion.fuse_results` 的 legacy 行为。
- `build_path_prompt` 与 `LLM_OUTPUT_SCHEMA` 的 legacy 协议。
- `scripts/run_main_experiment.sh` 的回归命令语义，除非是目录迁移所需路径修正。

以下内容必须版本化：

- 特征字典 v1：维度、类型、物理含义、单位、取值域、抽取规则。
- 证据图版本：导入 case 集合、边构建规则、相似度权重。
- 物理约束库与专家 SOP。
- prompt 模板：变量、顺序、输出协议。
- 消融实验配置：Top-N、阈值、训练集规模、随机种子。

### 5.2 双口径门禁

legacy 基线不再是新框架的硬门禁，但仍是回归锚点。任何改动只要影响 legacy 路径，都必须满足：

```bash
python -m pytest -q
```

`tests/test_baseline_lock.py` 必须一直全绿，锁住旧 CLI 路径不漂移。若运行 legacy 评估，仍应保持：

- `case_count == 85`
- `correct == 58`
- `accuracy == 0.6823529411764706`
- `fiber` recall 为 0
- 逐 case prediction 与 `artifacts/organized_rca_v2_60_40_seed42_baseline/predictions.json` 兼容
- `label_leakage == false`

新框架走独立入口与独立评估口径，不要求复现 58/85。新框架实验必须满足：

- `run_manifest.json` 记录证据图版本、特征字典版本、约束库版本、SOP 版本、prompt 模板 hash、Top-N、阈值、训练集规模、随机种子。
- 报告 coverage / accuracy、precision_at_coverage、低置信度降级率、人工介入率、`fiber` 分层指标。
- N5a 必须报告 signature 纯净度、混合标签 signature 覆盖率、桶内多数投票准确率，不允许只报完全匹配数量。
- N5b 当前只有 8 条样本，除非合并数据集后样本量显著增加，否则不要据此下稳定消融结论。

l2fixed v2 实验还必须满足：

- `python scripts/prepare_l2fixed_stratified.py --check` 全绿，manifest 不漂移。
- `run_manifest.json` 记录 `split_manifest_hash`、`feature-dictionary-v2`、`constraint-library-v3`、
  `learned-sop-v1` hash、证据图版本和决策策略。
- 报告数据质量摘要：缺 `alarm_ip_interface`、缺 `Lane number`、L1/L2 lane 宽度异常、
  `host_snr` 存在率和 optical blackout 数。
- learned SOP 只能从 train split 学习；test split 只做最终评估，不允许反向改树、约束或特征。
- 只有人工确认的 case 可以通过 `feedback.py` 回灌 evidence-graph-v2，自动推理结果只能进入 artifact。
- 非自进化正式实验统一使用 `scripts/run_offline_sop_llm_experiment.py`：先保存并重新加载
  `offline-knowledge-bundle-v1`，再运行 test；不得在同一进程里绕过持久化边界直接拿临时训练对象评估。
- 正式 GPU 实验必须记录运行前后 `nvidia-smi` 快照并在 `finally` 中关闭 vLLM；
  运行结束后还要从进程外复查显存，不得仅凭脚本正常退出宣称资源已释放。
- HTML 报告必须包含总览与逐 case 页面；逐 case 至少展示 feature token、历史候选、
  learned SOP、物理证据链、M9 原因、缺失证据、LLM 每轮输出/违规和诊断子图。

### 5.3 不做清单

- 不做多 Agent 编排；只做一个主控流程加一组可审计模块。
- 不重命名 `L1`、`L2`、`fiber`。
- 不重新生成数据集作为默认动作；合并清洗数据集必须显式记录来源与版本。
- 不删除 `fusion.fuse_results`，它是 legacy 58/85 与 59/85 的回归锚点。
- 不把同源一致解释为独立互证。
- 不承诺靠 prompt 或 LLM 单独解决 `fiber`。
- 不把可解释特征落成黑盒 embedding。
- 不逐 case 手写 prompt；prompt 模板必须固定变量、顺序和输出协议。
- 不在没有特征字典版本号、证据图版本号和 prompt hash 的情况下报告实验结果。
- 不把 `before/` 或归档快照变成第二套活动实现。

## 6. 推荐开发顺序

严格遵循 `Progress.md` 的 T1-T12 指针。当前正确顺序是：

| 编号 | 任务 | 交付物 | 建议窗口 |
| --- | --- | --- | --- |
| T1 | 冻结特征字典 v1 | 特征字典表 + 版本号 + signature 分辨率验收 | Day 1-2 |
| T2 | 整理通用物理约束规则 | 约束规则清单 + prompt 化模板 | Day 1-2 |
| T3 | 实现证据到特征向量抽取 | 抽取器代码 + 单测 | Day 3-5 |
| T4 | 构建证据图与 Jaccard 检索 | 图存储 + 检索 API + Top-N 结果结构 | Day 3-5 |
| T5 | 实现分流路由与三分支处理器 | N4 + N5a / N5b / N5c | Day 6-9 |
| T6 | 接入 LLM 推理与约束校验器 | prompt 模板 + 校验断言 + 推理日志 | Day 6-9 |
| T7 | 置信度与降级策略 | 阈值配置 + 通用建议 / 人工介入路径 | Day 10-12 |
| T8 | 报告生成器改造 | 结论 + 证据链 + 匹配路径来源 | Day 10-12 |
| T9 | 证据图回灌闭环 | 回灌脚本 + 版本记录 | Day 10-12 |
| T10 | 实验 / 消融框架 | 参数化脚本 + 结果汇总模板 | 实验期并行 |
| T11 | 系统架构图 / 过程说明图 | 可用于论文的框图 | Day 1-5 并行 |
| T12 | 论文初稿 | 方法 / 实验 / 讨论章节草稿 | 第 3-4 周 |

串并行要求：

1. Day 1-2：T1 特征字典冻结、T2 约束规则整理、T11 架构图并行。
2. Day 3-5：T3 特征抽取接 T4 证据图 + 检索；同步等待黄泽舜 / 王雅琪的合并数据集。
3. Day 6-9：T5 三分支路由接 T6 LLM + 约束校验；专家 SOP 应在 Day 8 前进入 M6，否则用兜底 SOP。
4. Day 10-12：T7-T9 收尾，跑一版端到端 dry-run。
5. Day 13-14：与夏思博对齐 badcase，冻结 v1 框架，转入实验期。

分工交接：

- 夏思博：审核 M5 约束库，提供 badcase 清单，确认实验拆分。
- 黄泽舜 / 王雅琪：完成两个数据集合并，提供统一数据集给 T3 / T4，并查找专家排障 SOP。
- 谢其桐：挑战赛项目结束后接 T10 消融 / 基线执行。
- Codex / 自动化：优先承担批跑、结果汇总、表格生成，关键节点人工校验。

每次完成代码或实验变更后，更新 `Progress.md` 的任务状态、门禁结果和变更日志。

当前 v2 后续顺序以 `Progress.md` 0.4 为准：先稳定 l2fixed 数据契约、v2 特征、learned SOP、
evidence-graph-v2 报告/回灌，再做 Top-N、阈值、SOP 深度和多 seed 消融。旧 Day 1-14 排期仅作历史规划参考。
