# KG 分层注入实验代码修改说明

> **状态：本文描述的修改已于 2026-08-06 从活动代码树回退。**
> `rca_framework/llm.py`、`pipeline.py`、`cli.py` 已恢复到改造前版本，
> `tests/test_kg_injection.py` 与两个消融脚本已移出活动树。
> 本文继续作为该版实现的完整技术记录保留。
>
> - 回退后的代码基线与逐模块 Agent 化改造策略：`docs/AGENT_REFACTOR_MODULE_STRATEGY_CN.md`
> - 被回退版本的完整代码与 diff：`archive/rca_framework_snapshot_20260806_layered_injection/`
> - 实验产物未删除，仍在 `artifacts/layered_injection_20260805/`

## 1. 文档目的

本文记录为了执行
`docs/KG_INJECTION_ABLATION_DEEPSEEK32B_REPORT_CN.md`
中的四组 DeepSeek-32B 消融实验，对 RCA v2 代码所做的修改。

修改目标不是重写 RCA 主链路，而是在保持数据处理、异常提取、KG、符号规则和融合器
不变的前提下，引入两个可独立控制的实验变量：

1. KG 信息是否按覆盖状态分层注入 LLM。
2. LLM 路打分是否再次混入 KG 分数。

同时增加证据充分性输出、四组消融运行器、结果汇总工具和回归测试。

## 2. 修改范围

### 2.1 修改的核心文件

| 文件 | 修改内容 |
| --- | --- |
| `rca_framework/llm.py` | KG 覆盖分档、分层 prompt、新输出 schema、独立 LLM 打分、运行时模式切换 |
| `rca_framework/pipeline.py` | reasoner 复用、推理参数透传、覆盖状态和充分性统计 |
| `rca_framework/cli.py` | 新增三项命令行参数，并写入运行 manifest |

### 2.2 新增的实验文件

| 文件 | 用途 |
| --- | --- |
| `scripts/run_injection_ablation.py` | 单次加载 32B，连续执行四组消融 |
| `scripts/summarize_injection_ablation.py` | 汇总逐 case 结果、覆盖分档、充分性和组间变化 |
| `scripts/fetch_model_modelscope.py` | 从 ModelScope 补全并校验本地 32B 模型快照 |
| `tests/test_kg_injection.py` | 覆盖分档、prompt 屏蔽、打分公式和兼容模式测试 |

### 2.3 归档与报告

| 路径 | 用途 |
| --- | --- |
| `archive/rca_framework_snapshot_20260805_pre_layered_injection/` | 改造前完整代码与测试快照 |
| `docs/KG_INJECTION_ABLATION_DEEPSEEK32B_REPORT_CN.md` | 实验结果和结论 |
| `artifacts/layered_injection_20260805/` | 确定性基线、四组预测、汇总和运行日志 |

### 2.4 未修改的推理模块

以下模块没有为了本次实验修改：

- `rca_framework/data.py`
- `rca_framework/anomaly.py`
- `rca_framework/graph.py`
- `rca_framework/rules.py`
- `rca_framework/fusion.py`
- `rca_framework/types.py`

因此，训练数据、阈值学习、异常定义、KG 边和 feature rule 学习、符号规则以及最终
融合公式保持不变。实验变量只位于 KG 结果进入 LLM 的方式，以及 LLM 路分数的构造方式。

## 3. 改造前后的调用链

### 3.1 改造前

```text
CaseEvidence
  → graph.query
  → build_path_prompt（所有 case 全量注入 KG）
  → LLM
  → 0.35 × KG scores + 0.65 × LLM confidence
  → fuse_results（再与符号规则路融合）
```

改造前有两个需要验证的问题：

- 当 KG 对 case 没有路径或组合规则时，prompt 仍注入归一化候选分数；此时分数主要是
  训练集类别先验。
- LLM 输出后先混入一次 KG 分数，随后整个 KG+RAG+LLM 路又进入最终融合，KG 影响可能
  被重复计入。

### 3.2 改造后

```text
CaseEvidence
  → graph.query
  → classify_kg_coverage
       ├─ covered
       ├─ partial
       └─ uncovered
  → full 或 layered prompt
  → LLM + evidence_sufficiency
  → legacy 或 llm_only 打分
  → fuse_results
  → 按覆盖状态/充分性统计
```

旧链路没有被删除，而是保留为 `full + legacy`，用于精确复现历史基线。

## 4. `rca_framework/llm.py` 修改

### 4.1 增加两组实验模式

新增常量：

```python
INJECTION_MODES = ("full", "layered")
SCORE_MODES = ("legacy", "llm_only")
```

含义：

- `full`：继续调用原来的 `build_path_prompt()`，对所有 case 全量注入 KG。
- `layered`：调用新的 `build_layered_prompt()`，按 KG 覆盖状态选择注入字段。
- `legacy`：保留原有 0.35 KG 分数回灌。
- `llm_only`：LLM 路分数不再引用 `graph_result["scores"]`。

`build_path_prompt()` 本身未改写，确保 `full` 模式的 prompt 可以复现旧行为。

### 4.2 增加 KG 覆盖状态判定

新增：

```python
classify_kg_coverage(case, graph_result)
```

当前实现只依据 KG 查询结果的结构判定，不增加新的调参阈值：

| 状态 | 判定条件 | 解释 |
| --- | --- | --- |
| `covered` | `matched_rule_count > 0` | 至少命中一条 KG feature rule |
| `partial` | 未命中规则，但 `path_count > 0` | 只命中原子异常路径 |
| `uncovered` | 无规则且无路径 | KG 没有 case 特异结构证据 |

判定结果还记录：

- `anomaly_count`
- `path_count`
- `matched_rule_count`
- `max_retrieval_similarity`

这些字段随每条 LLM 结果写入 `kg_coverage`，用于报告按覆盖状态分组。

### 4.3 增加分层 prompt

新增：

```python
build_layered_prompt(case, graph_result)
```

不同覆盖状态下的注入策略：

| Prompt 字段 | covered | partial | uncovered |
| --- | --- | --- | --- |
| 根因定义 | 注入 | 注入 | 注入 |
| 目标 case 摘要和异常 | 注入 | 注入 | 注入 |
| `candidate_path_scores` | 注入 | 屏蔽 | 屏蔽 |
| `candidate_feature_profile_scores` | 注入 | 屏蔽 | 屏蔽 |
| `matched_kg_feature_rules` | 注入 | 屏蔽 | 屏蔽 |
| `root_cause_paths` | 注入 | 仅保留原子路径 | 屏蔽 |
| `retrieved_training_cases` | 注入 | 注入并附低相似度提示 | 注入并附低相似度提示 |
| `withheld_kg_fields` | 无 | 说明屏蔽原因 | 说明屏蔽原因 |

对 `partial/uncovered` 还增加两条约束：

- 缺失字段只表示未采集，不能当作正常。
- 不得根据历史类别频率倾向多数类。

`uncovered` prompt 明确要求把 `evidence_sufficiency` 设为 `insufficient`，并在
`missing_information` 中列出需要补采的测量。

### 4.4 增加证据充分性 schema

保留原 `LLM_OUTPUT_SCHEMA` 供 `full` 模式使用；新增：

```python
LAYERED_OUTPUT_SCHEMA
```

新 schema 增加必填字段：

```json
{
  "evidence_sufficiency": "sufficient | insufficient"
}
```

仍然保留三分类必选的 `prediction`，目的是让本次实验与历史 accuracy 直接可比。
也就是说，本次修改只让模型表达“证据不足”，还没有把输出协议改成真正的
`L1/L2/fiber/abstain` 四态协议。

`parse_llm_json()` 同步解析该字段：

- 合法值：`sufficient`、`insufficient`
- 旧 schema 或缺失字段：记为 `unreported`

因此旧模式和旧模型输出仍能被解析。

### 4.5 增加独立 LLM 打分

原 `legacy` 公式保持不变：

```text
score = normalize(0.35 × graph_scores
                  + 0.65 × confidence × one_hot(prediction))
```

新增 `llm_only` 公式：

```text
uniform = 1 / 3
other_score = (1 - confidence) × uniform
predicted_score = confidence + other_score
```

等价于在均匀分布与 prediction 的 one-hot 投票之间插值。这样设计有两个目的：

1. 分数不再读取 KG 候选分数。
2. 只要 `confidence > 0`，LLM 选择的类别始终是分数最高类别；`confidence=0`
   时退化为均匀分布。

该实现替代了最初尝试的“预测类直接取 confidence，剩余概率平分”方案。后者在
`confidence < 1/3` 时会导致其他类别分数高于 LLM 自己的 prediction，测试阶段发现后
已修正。

### 4.6 充分性对融合置信度的可选缩放

新增参数：

```python
insufficient_confidence_scale: float = 1.0
```

当 LLM 返回 `insufficient` 时：

```text
fusion_confidence = reported_confidence × insufficient_confidence_scale
```

并同时保存：

- `reported_confidence`：LLM 原始置信度
- `confidence`：进入融合器的置信度

本次实验固定为 `1.0`，所以只记录充分性，不改变融合权重。这避免把 prompt 改动和
人为置信度惩罚混在同一个实验变量中。

### 4.7 Reasoner 运行时重配置

`PathLLMReasoner` 新增：

```python
configure(...)
output_schema
build_prompt(...)
```

`configure()` 只允许修改不需要重新加载模型的参数：

- `max_new_tokens`
- `guided_json`
- `injection_mode`
- `score_mode`
- `insufficient_confidence_scale`

模型路径、tensor parallel、dtype 等加载期配置不能通过 `configure()` 修改，防止错误
复用已加载模型。

vLLM guided decoding 会根据当前模式动态选择：

- `full` → `LLM_OUTPUT_SCHEMA`
- `layered` → `LAYERED_OUTPUT_SCHEMA`

### 4.8 输出结果增加审计字段

LLM 路结果新增：

- `reported_confidence`
- `evidence_sufficiency`
- `kg_coverage`
- `injection_mode`
- `score_mode`

fallback 路也写入相同审计字段。`uncovered` 的确定性 fallback 会标记
`evidence_sufficiency="insufficient"`。

## 5. `rca_framework/pipeline.py` 修改

### 5.1 拆分模型加载配置与推理配置

新增：

```python
RCAPipeline._get_reasoner(settings)
```

改造前 `_reasoners` 的缓存键包含 `max_new_tokens`、guided JSON 等所有参数，只要任一
参数变化就会创建新 reasoner，并可能重新加载 32B 模型。

改造后缓存键只包含模型加载期参数：

- backend
- model path
- tensor parallel size
- GPU memory utilization
- max model length
- dtype
- enforce eager
- disable custom all-reduce

prompt 模式、打分模式、max tokens 和充分性缩放通过 `configure()` 切换。

这是四组消融能够共享一次 32B 模型加载的关键改动。

### 5.2 `infer()` 增加参数

新增可选参数：

```python
injection_mode="layered"
score_mode="llm_only"
insufficient_confidence_scale=1.0
```

参数通过 `_get_reasoner()` 传入 LLM 路。目标标签删除、异常提取、KG 查询、规则匹配和
融合顺序未变。

### 5.3 `evaluate()` 增加分组统计

评估 summary 新增：

```json
{
  "kg_injection": {
    "injection_mode": "...",
    "score_mode": "...",
    "insufficient_confidence_scale": 1.0
  },
  "kg_coverage_regime": {
    "covered": {"cases": 0, "correct": 0, "accuracy": 0.0},
    "partial": {"cases": 0, "correct": 0, "accuracy": 0.0},
    "uncovered": {"cases": 0, "correct": 0, "accuracy": 0.0}
  },
  "evidence_sufficiency": {
    "sufficient": {"cases": 0, "correct": 0, "accuracy": 0.0},
    "insufficient": {"cases": 0, "correct": 0, "accuracy": 0.0}
  }
}
```

原有 accuracy、recall、confusion matrix、decision status 和 label leakage 字段保留。

## 6. `rca_framework/cli.py` 修改

`train-evaluate` 和 `infer` 新增：

```text
--kg-injection full|layered
--llm-score-mode legacy|llm_only
--insufficient-confidence-scale FLOAT
```

当前代码默认值：

```text
--kg-injection layered
--llm-score-mode llm_only
--insufficient-confidence-scale 1.0
```

CLI 会把配置传给 pipeline，并在 `run_manifest.json` 中新增：

```json
{
  "kg_injection": {
    "injection_mode": "layered",
    "score_mode": "llm_only",
    "insufficient_confidence_scale": 1.0
  }
}
```

复现改造前行为应显式使用：

```bash
--kg-injection full \
--llm-score-mode legacy
```

## 7. 四组消融运行器

新增 `scripts/run_injection_ablation.py`。

固定执行顺序：

```text
full__legacy
full__llm_only
layered__legacy
layered__llm_only
```

运行器行为：

1. 加载固定数据集并按前 126 条训练、后 85 条测试。
2. 只训练一次 threshold、KG 和符号规则。
3. 保存一次模型产物。
4. 四组依次调用同一个 `RCAPipeline.evaluate()`。
5. 由于 `_get_reasoner()` 的缓存键只含加载期配置，四组共享同一个 vLLM 实例。
6. 每组保存：
   - `evaluation_summary.json`
   - `predictions.json`
7. 总目录保存：
   - `run_manifest.json`
   - `ablation_comparison.json`

为避免覆盖已有实验，输出目录非空时直接报错。

## 8. 结果汇总工具

新增 `scripts/summarize_injection_ablation.py`，读取四组 `predictions.json` 并生成
`analysis.json`。

汇总内容包括：

- 最终 prediction 分布
- LLM 路 prediction 分布与单路 accuracy
- 符号规则路 prediction 分布
- covered/partial/uncovered 的 case 数、accuracy 和真实标签分布
- sufficient/insufficient 的 case 数和 accuracy
- 零异常 case 数
- 平均 prompt 字符数
- 组间最终 prediction、LLM prediction 和 raw output 变化
- 由改动修正、恶化或仍错误的具体 case

该工具只做离线分析，不进入在线推理路径。

## 9. 模型快照补全工具

实验开始时，本地
`DeepSeek-R1-Distill-Qwen-32B`
目录只有 8 个权重分片中的第 1 个，且该分片不完整，同时缺少 tokenizer 和 index。
HuggingFace 在当前主机不可达，因此新增 `scripts/fetch_model_modelscope.py`。

工具特性：

- 从 ModelScope API 获取远端文件 manifest。
- 最多并发下载 4 个文件。
- 使用 `.part` 临时文件，成功后原子替换目标文件。
- 按远端 manifest 校验每个文件的字节数。
- 失败后退避重试，默认最多 5 次。
- 已有且大小正确的文件自动跳过，支持中断后重跑。
- 全部完成后再次做完整性检查。

该脚本只解决实验环境准备，不是 RCA 运行时依赖。

## 10. 测试修改

新增 `tests/test_kg_injection.py`，覆盖：

1. KG 覆盖状态只由路径和匹配规则结构决定。
2. `partial/uncovered` prompt 确实屏蔽聚合 KG 分数。
3. `partial` 保留原子路径统计。
4. 分层 prompt 不直接泄露 `graph_result["prediction"]`。
5. `full` prompt 保持旧字段结构。
6. `llm_only` 分数不依赖 KG 分数。
7. LLM prediction 在正置信度下始终是分数 argmax。
8. 零置信度退化为均匀分布。
9. 充分性字段可解析，旧输出回退为 `unreported`。
10. `configure()` 拒绝修改模型加载期配置。

最终完整测试结果：

```text
16 passed
```

确定性 baseline 与改造前逐 case 对比：

```text
旧：58/85
新：58/85
confusion matrix：完全一致
decision status：完全一致
逐 case prediction 差异：0
```

`full + legacy` 的真实 32B 结果为 59/85，与历史报告一致，说明兼容模式能够复现旧
LLM 行为。

## 11. 兼容性与回滚

### 11.1 运行时兼容

- `build_path_prompt()` 和旧 schema 保留。
- `parse_llm_json()` 可以解析没有充分性字段的旧输出。
- 新增构造参数均追加在 `PathLLMReasoner` 参数末尾，旧位置参数调用不受影响。
- `PipelineConfig` 和保存的 `model.json` schema 未增加新必填训练字段。
- 新配置属于推理运行时参数，记录在 run manifest，而不是训练模型本体。

### 11.2 逻辑回滚

无需替换文件即可回到旧行为：

```bash
python -m rca_framework.cli train-evaluate \
  ... \
  --kg-injection full \
  --llm-score-mode legacy
```

### 11.3 代码回滚

改造前完整代码位于：

```text
archive/rca_framework_snapshot_20260805_pre_layered_injection/
```

快照中保存了：

- `rca_framework/`
- `tests/`
- 原 `scripts/run_main_experiment.sh`
- 文件 SHA-256 记录和行为说明

## 12. 实验环境相关但未固化到业务代码的配置

当前主机的 NCCL 2.27.3 在 GPU 6、7 默认 P2P/SHM 通道上会初始化死锁。独立
`all_reduce` 测试确认后，正式实验通过以下环境变量运行：

```bash
VLLM_USE_V1=1
VLLM_ENABLE_V1_MULTIPROCESSING=0
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=6,7
```

这些参数没有硬编码进 `rca_framework` 或消融脚本，因为它们属于当前服务器的运行环境
规避项，在其他服务器上未必需要。复现实验时需要显式设置。

## 13. 已知限制与注意事项

### 13.1 当前默认值与实验结论存在策略差异

当前 CLI 和 Python API 默认使用 `layered + llm_only`，但实验中该配置为 58/85，
低于 `full + legacy` 的 59/85。实验报告建议暂时把旧模式保留为准确率基线，将分层
模式作为影子诊断和充分性实验。

在生产策略确定前，调用方应显式传参，不要依赖默认值。

### 13.2 `covered` 的命名强于当前判定语义

当前只要命中任意 KG feature rule 就记为 `covered`。但 feature rule 同时包含：

- 单特征 `characteristic`
- 双特征 `characteristic_pair`

因此 `covered` 可能只表示“命中一个已知判别特征”，不一定表示完整异常组合在训练集中
出现过。若要严格区分“已见组合”，后续应单独检查 pair rule、相似 case 或定义更严格的
coverage policy。

### 13.3 `uncovered` 仍会收到带标签的检索案例

分层 prompt 屏蔽了聚合 KG 分数，但仍注入 `retrieved_training_cases`。即使最高
similarity 为 0，检索结果仍带 `root_cause` 字段，只是 prompt 增加了“不得直接套用”
提示。

因此，当前实现没有完全消除类别先验的间接影响。更严格的实验可以在低相似度时：

- 完全不注入检索案例；或
- 保留相似度和重叠异常，但隐藏案例标签。

### 13.4 “分层注入”消融同时改变了输出 schema

`full` 使用旧 schema，`layered` 使用带 `evidence_sufficiency` 的新 schema，并增加
了约束和覆盖说明。因此 `full` 与 `layered` 的差异不只是屏蔽字段，还包括任务提示和
输出格式变化。

若要做更严格的因果消融，应拆成：

1. 只改变注入字段，schema 不变。
2. 只增加充分性字段，注入字段不变。
3. 两者同时改变。

### 13.5 充分性尚未成为真正门控

`insufficient_confidence_scale` 默认 1.0，系统仍强制输出 L1/L2/fiber，并继续执行
原融合逻辑。当前代码没有：

- `abstain`
- `request_evidence`
- 选择性分类出口
- 补证据后重入

所以这次修改只为后续 Agent 门控提供信号，没有完成完整的三态诊断出口。

### 13.6 缩放参数缺少范围校验

CLI 当前接受任意浮点数作为 `--insufficient-confidence-scale`。建议后续限制在
`[0, 1]`，避免负数或大于 1 的值进入融合器。

### 13.7 仍未校验 LLM 返回的 `path_ids`

schema 只限制 `path_ids` 为字符串数组，没有在解析后验证它们一定属于目标 case 的
异常或候选路径。这是改造前就存在的限制，本次实验没有扩大修改范围。

## 14. 建议的后续代码调整

基于本次实验，下一步建议按以下顺序修改：

1. 将 `covered` 判定细化为 singleton、pair、high-similarity exemplar 三类覆盖。
2. 对 `partial/uncovered` 隐藏低相似度检索案例的标签。
3. 把 `score_mode` 改为按 coverage 条件选择：
   - covered：允许 KG 校准；
   - partial/uncovered：保持 LLM 独立。
4. 给 `insufficient_confidence_scale` 增加 `[0,1]` 校验。
5. 新增真正的 `abstain/request_evidence` 输出和 coverage-accuracy 评估。
6. 校验并过滤 LLM 返回的 `path_ids`。

在完成这些修改之前，应继续保留 `full + legacy` 回归测试，防止架构演进过程中丢失
可比较的历史基线。
