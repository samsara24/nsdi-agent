# 用一条真实 case 讲懂 RCA 

这份文档带着一条真实 case 完整走一遍：

```text
case 原始指标
  -> 提取异常
  -> 在训练好的 KG 中找路径
  -> RAG 检索相似训练 case
  -> 可选 LLM 推理
  -> 独立符号规则推理
  -> 两路融合
  -> L1 / L2 / fiber
```

看完后应该能完成四件事：

1. 跑一条 case；
2. 单独测试 KG+RAG 或 KG+RCA；
3. 看懂结果为什么这样预测；
4. 知道修改异常、KG、RAG、规则和融合时分别改哪里。

## 1. 先把业务概念说清楚

系统最终只判断三个位置：

```text
400G 设备/端口 ----- 中间光纤 ----- 200G 设备/端口
      L1                fiber              L2
```

- `L1` 永远是 400G 一端；
- `L2` 永远是 200G 一端；
- `fiber` 是中间链路介质。

原始数据使用 local/remote，但 local/remote 只是观察方向，不是固定物理角色。数据准备时会把
400G 一端整体改成 L1，把 200G 一端整体改成 L2。后面的模型只认 L1/L2。

## 2. 本文使用哪条 case

使用：

```text
datasets/rca_v2/case_000268.json
```

它的脱敏 ID 是 `case_2ee657ca7489`，参考标签是 L1。标签只在最后检查预测对错时使用，
推理前代码会主动删除它。

先查看这条 case：

```bash
cd /home/shibinpeng/luoyu/huangzeshun/nsdi27

jq '{
  case_id: .case_id,
  label: .label,
  alarm_name: .alarm_name,
  endpoints: .link_side_ip_interface_map,
  rxpower: .rxpower,
  media_snr: .media_snr,
  serdes_snr: .serdes_snr,
  RxLOS: .RxLOS,
  RxLOL: .RxLOL
}' datasets/rca_v2/case_000268.json
```

这条 case 最值得关注的现象是：

- L2 的 `RxLOS=Abnormal`；
- L2 的 `RxLOL=Abnormal`；
- L2 的 rxpower 有 1 个 lane 掉到哨兵值；
- L2 的 media_snr 有 1 个 lane 掉到哨兵值；
- L2 的 serdes_snr 有 1 个 lane 掉到哨兵值；
- L1 的 host_snr 缺失。

先不要直接判断根因。模型第一步是把数值翻译成统一的“异常名词”。

## 3. 第一步：case 怎么变成异常节点

运行：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel evidence \
  --output /tmp/rca_evidence.json
```

只看异常：

```bash
jq '.evidence | {
  anomalies: [.anomalies[] | {
    anomaly_id,
    noun,
    severity,
    evidence
  }],
  missing_fields
}' /tmp/rca_evidence.json
```

实际提取出 5 个异常节点：

| 异常 ID | 人能理解的意思 | 严重度 |
|---|---|---:|
| `status_fault:L2:RxLOL` | L2 的 RxLOL 状态异常 | 1.0 |
| `status_fault:L2:RxLOS` | L2 的 RxLOS 状态异常 | 1.0 |
| `signal_drop:L2:media_snr` | L2 介质侧 SNR 有 1/8 lane 掉底 | 0.125 |
| `signal_drop:L2:rxpower` | L2 接收光功率有 1/8 lane 掉底 | 0.125 |
| `signal_drop:L2:serdes_snr` | L2 SerDes SNR 有 1/4 lane 掉底 | 0.25 |

缺失字段为 `L1.host_snr`。

### 为什么数值要先变成异常节点

如果直接用原始数值建 KG：

- 每条 case 数值不同，图节点会非常碎；
- 很难解释“哪个现象支持哪个根因”。

统一成异常 ID 后，不同 case 可以共享同一个节点：

```text
case A --出现--> signal_drop:L2:rxpower
case B --出现--> signal_drop:L2:rxpower
case C --出现--> signal_drop:L2:rxpower
```

训练时就能统计这个异常在 L1、L2、fiber 中分别出现多少次。

### 对应关键代码

入口在 `rca_framework/anomaly.py`：

```python
evidence = extract_evidence(target, pipeline.thresholds)
```

`extract_evidence()` 主要判断：

1. 有没有掉底值，例如 rxpower 小于等于 -39；
2. 是否超出训练集学到的 IQR 稳健范围；
3. LOS/LOL、lane 不均衡、方向损耗和双向耦合是否异常。

阈值只由训练集拟合：

```python
pipeline.thresholds = fit_thresholds(training_cases)
```

目标 case 不参与阈值拟合。

## 4. 第二步：训练数据怎么构建 KG

KG 将训练 case 的关系聚合成：

```text
根因节点 --某种异常关系--> 异常节点
```

例如：

```text
root_cause:L1
  --HAS_STATUS_FAULT-->
anomaly:status_fault:L2:RxLOL
```

这里的边不是人工物理定律，而是训练数据统计出来的关联。

### KG 的训练过程

```bash
python -m rca_framework.cli train-evaluate \
  --data-dir datasets/rca_v2 \
  --train-size 200 \
  --output-dir artifacts/rca_demo_new \
  --backend none
```

`artifacts/rca_demo_new` 必须是新目录，代码拒绝覆盖已有结果。

内部执行：

```python
cases = load_cases(data_dir)
pipeline = RCAPipeline().fit(cases[:200])
```

`RCAPipeline.fit()` 继续执行：

```python
self.thresholds = fit_thresholds(labeled)
views = [extract_evidence(case, self.thresholds) for case in labeled]
self.graph.fit(views)
self.rules.fit(views)
```

`AnomalyKnowledgeGraph.fit()` 为每条边统计：

- `count`：某异常和某根因共同出现多少次；
- `root_cause_frequency`：该根因中出现此异常的比例；
- `precision`：出现此异常的 case 中有多少属于该根因；
- `lift`：该异常对这个根因的提升程度；
- `weight`：综合统计得到的边权重。

边权重大致是：

```text
类内频率 × precision × log(1 + lift) × log(1 + count)
```

### 查看真实 KG 边

查看 `status_fault:L2:RxLOL` 连向哪些根因：

```bash
jq '
  [.edges[]
   | select(.anomaly_id == "status_fault:L2:RxLOL")
   | {
       root_cause,
       count,
       root_cause_frequency,
       precision,
       lift,
       weight
     }]
' artifacts/rca_v2_baseline/model/knowledge_graph.json
```

同一个异常可以分别连向 L1、L2、fiber，因为它可能在三类训练 case 中都出现过。
推理比较的是全部匹配边累加后的总分。

## 5. 第三步：新 case 怎么在 KG 中推理

单独运行 KG+RAG 通道：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel kg-rag \
  --output /tmp/rca_kg_rag.json
```

查看分数和前三条路径：

```bash
jq '{
  prediction: .kg_rag_llm.prediction,
  confidence: .kg_rag_llm.confidence,
  scores: .kg_rag_llm.scores,
  top_paths: [.kg_rag_llm.graph_paths[:3][] | {
    anomaly_id,
    root_cause,
    score,
    edge_statistics
  }]
}' /tmp/rca_kg_rag.json
```

实际分数：

```text
L1     0.6442
L2     0.0838
fiber  0.2720
```

所以 KG 通道预测 L1。

### 一条路径是怎么算出来的

这条 case 最强路径是：

```text
query:case_2ee657ca7489
  -> EXHIBITS
anomaly:status_fault:L2:RxLOL
  -> INDICATES
root_cause:L1
```

该边的训练统计：

```text
training_count       19
precision            0.50
lift                 1.7544
root_cause_frequency 0.3333
```

case 中这个状态异常的 severity 是 1.0，因此路径分约为 0.5059。

同一个 RxLOL 异常也会连向 L2 或 fiber，但那些边权较低。系统把当前 case 的全部异常、
全部候选根因路径加起来，再归一化成三类分数。

对应代码在 `rca_framework/graph.py` 的 `query()`：

```python
for anomaly_id, item in anomaly_map.items():
    for edge in self.edge_index.get(anomaly_id, []):
        path_score = edge.weight * severity
        raw_scores[edge.root_cause] += path_score
```

severity 会限制在 0.25 到 3.0 之间，避免极端值完全控制结果。

## 6. 第四步：RAG 怎么找相似 case

```bash
jq '[
  .kg_rag_llm.retrieved_cases[] | {
    case_id,
    root_cause,
    similarity,
    overlap_anomalies
  }
]' /tmp/rca_kg_rag.json
```

第一条相似训练 case：

```text
case_id:     case_926693d885a8
root_cause:  L1
similarity:  1.0
```

它和目标 case 共享全部 5 个异常 ID，所以相似度为 1。

### RAG 相似度是什么
当前 RAG 使用：

```text
IDF 加权的异常集合 Jaccard 相似度
```

对应 `rca_framework/graph.py` 的 `retrieve()`：

```python
overlap = query_ids & candidate_ids
union = query_ids | candidate_ids
similarity = sum(idf[x] for x in overlap) / sum(idf[x] for x in union)
```

少见异常的 IDF 更高，所以少见异常重合更重要。

### 这个例子暴露的 RAG 局限

目标 case 有 1/8 lane 掉底，最相似训练 case 是 1/4 lane 掉底，但相似度仍可为 1。

原因是当前异常 ID 只有：

```text
signal_drop:L2:rxpower
```

没有把“1/8”或“1/4”编码进异常 ID。严重程度用于 KG 路径分，却没有进入 RAG 集合相似度。

如果要让 RAG 区分 lane 比例，可以：

1. 给异常增加 severity 分桶；
2. 修改 `retrieve()`，加入 severity 接近度；
3. 改成结构化向量或 embedding 检索。

## 7. 第五步：LLM 在 KG+RAG 后面做什么

方法一路的 LLM 输入包括：

- 目标 case 的异常；
- KG 三类候选分数；
- KG 路径；
- KG 特征规则；
- RAG 相似训练 case；
- 缺失字段。

Prompt 在 `rca_framework/llm.py` 的 `build_path_prompt()` 构建。

使用 `--backend none` 时不加载大模型，直接采用 KG 路径预测：

```text
reasoning_mode = deterministic_path_fallback
```

启用真实 LLM：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel kg-rag \
  --backend vllm \
  --model-path /absolute/path/to/model \
  --output /tmp/rca_kg_rag_llm.json
```

LLM 必须返回：

```json
{
  "prediction": "L1",
  "confidence": 0.8,
  "path_ids": ["status_fault:L2:RxLOL"],
  "reasoning": "解释",
  "missing_information": ["L1.host_snr"]
}
```

合法 LLM 结果与 KG 分数组合：

```text
35% KG 分数 + 65% LLM 选择及置信度
```

JSON 无法解析时自动回退到 KG。判断是否真的用了 LLM：

```bash
jq '.kg_rag_llm | {
  reasoning_mode,
  prediction,
  raw_output,
  reasoning
}' /tmp/rca_kg_rag_llm.json
```

## 8. 第六步：独立测试 KG+RCA 符号规则

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel kg-rca \
  --output /tmp/rca_kg_rca.json
```

查看：

```bash
jq '{
  prediction: .kg_rca.prediction,
  confidence: .kg_rca.confidence,
  scores: .kg_rca.scores,
  matched_rule_count: .kg_rca.matched_rule_count,
  top_L1_rules: .kg_rca.matched_rules.L1[:3]
}' /tmp/rca_kg_rca.json
```

实际结果：

```text
prediction:         L1
confidence:         0.9985
matched_rule_count: 10
```

最强规则之一：

```text
RULE_L1_0023

all_of:
  signal_drop:L2:media_snr
  status_fault:L2:RxLOL
```

目标 case 同时包含两个前件，所以规则命中。

`rca_framework/rules.py` 的 `SymbolicRuleEngine.fit()` 会：

1. 枚举训练 case 中的单异常；
2. 枚举两两异常组合；
3. 计算 confidence、lift、支持度和排他 margin；
4. 把同一个前件只分给判别力最强的一类；
5. 保证三类规则前件不重叠。

查看重叠审计：

```bash
jq '.overlap_audit' \
  artifacts/rca_v2_baseline/model/symbolic_rules.json
```

正常必须是 `total_overlap_count = 0`。

## 9. 第七步：两路怎么融合

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel full \
  --output /tmp/rca_full.json
```

查看：

```bash
jq '{
  reference_label: .reference_label_for_evaluation_only,
  kg_rag_prediction: .kg_rag_llm.prediction,
  kg_rca_prediction: .kg_rca.prediction,
  final_prediction: .fusion.prediction,
  final_confidence: .fusion.confidence,
  decision_status: .fusion.decision_status,
  fused_scores: .fusion.fused_scores,
  missing: .fusion.information_completion.missing_or_requested_fields
}' /tmp/rca_full.json
```

实际结果：

```text
参考标签：   L1
KG+RAG：    L1
KG+RCA：    L1
最终预测：   L1
最终置信度： 0.8959
状态：       agreement
缺失：       L1.host_snr
```

融合规则在 `rca_framework/fusion.py`：

```text
两路一致
  -> 采用共同标签

两路不同，某一路置信度高至少 0.20
  -> 采用明显更强的一路

两路不同且没有明显强者
  -> KG/RAG/LLM 占 0.55
  -> 符号规则占 0.45

融合第一名和第二名差小于 0.10
  -> 给暂定标签
  -> 标记 manual_review_recommended
```

生产 CLI 的等价命令：

```bash
python -m rca_framework.cli infer \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --output /tmp/rca_production_result.json
```

调试脚本用于看中间过程；生产使用 `rca_framework.cli infer`。

## 10. 为什么“L2 出现异常”却预测 L1

系统没有写死：

```text
L2 出异常 => L2 根因
```

它学到的是：

```text
在前 200 条训练数据中，
某组 L2 侧可观测异常
与 L1 标签有较强统计关联。
```

例如 `L2 RxLOL 异常 -> L1` 的训练边出现 19 次，precision 为 0.5，lift 为 1.7544。
符号规则也把多个 L2 异常组合分给 L1。

可能有两种解释：

1. L1 发送侧问题确实会在 L2 接收侧被观测；
2. 数据标签、采集方向或样本分布造成伪相关。

代码无法证明是哪一种。分析时必须结合物理含义和 bad case，不能只说“模型置信度高”。

## 11. 单通道测试方法

### 只测试异常提取

适用于新增指标、修改掉底阈值、修改 LOS/LOL：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel evidence
```

检查异常是否漏提/误提、severity 和缺失字段。

### 只测试 KG+RAG

适用于修改边权、路径打分或相似度：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel kg-rag \
  --output /tmp/kg_rag_only.json
```

检查三类分数、top path、top-k 相似 case 和 `reasoning_mode`。

### 只测试 KG+RCA

适用于修改规则选择或匹配：

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel kg-rca \
  --output /tmp/kg_rca_only.json
```

检查命中前件、规则归属、规则强度和零重叠。

### 测试完整融合

```bash
python scripts/debug_rca_case.py \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --channel full \
  --output /tmp/full.json
```

检查两路结果、冲突策略、证据是否重复和人工复核状态。

最后运行：

```bash
pytest -q
```

当前预期是 `7 passed`。

## 12. 修改结构时改哪里

### 新增一种原始指标

例如加入 `ber`：

1. 在 `rca_framework/anomaly.py` 的 `METRIC_ALIASES` 加别名；
2. 在 `METRIC_NOUNS` 加中文含义；
3. 在 `DOWN_THRESHOLDS` 定义掉底规则；
4. 确认 `extract_evidence()` 适用；
5. 增加测试；
6. 重新训练，旧模型不会自动出现新指标。


### 修改 KG

改：

```text
rca_framework/graph.py
  AnomalyKnowledgeGraph.fit()
  _fit_feature_rules()
  query()
```

修改后检查节点数、边数、每类边、top path 和 fiber 是否仍有有效边。

### 修改 RAG

改：

```text
rca_framework/graph.py
  retrieve()
```

可以加入 severity、异常组合、embedding、时间或链路上下文。修改后要人工查看 top-5，
不要只看总 accuracy。

### 修改 LLM

同步修改：

```text
rca_framework/llm.py
  LLM_OUTPUT_SCHEMA
  build_path_prompt()
  parse_llm_json()
```

三处不同步会导致模型不知道怎么输出，或输出字段被 parser 丢弃。

### 修改符号规则

改：

```text
rca_framework/rules.py
  SymbolicRuleEngine.fit()
  match()
```

参数在 `PipelineConfig`：

```text
min_rule_count
min_rule_confidence
min_rule_lift
min_rule_margin
max_rules_per_class
```

修改后必须保证 `overlap_audit.total_overlap_count == 0`。

### 修改融合

配置在 `rca_framework/pipeline.py`：

```text
graph_weight                 0.55
symbolic_weight              0.45
conflict_dominance_gap       0.20
manual_review_margin         0.10
```

逻辑在 `rca_framework/fusion.py`。至少测试两路一致、两种单路占优、加权冲突和人工复核五种情况。
不要用后 68 条测试标签直接挑权重。

## 13. 怎么分析批量结果

训练评估后主要看：

```text
artifacts/<run>/
  evaluation_summary.json
  predictions.json
  run_manifest.json
  model/
```

### 先确认实验配置

```bash
jq '{
  data_dir,
  train_size,
  test_size,
  backend,
  llm_runtime
}' artifacts/rca_v2_deepseek32b_vllm/run_manifest.json
```

### 看总体和分类指标

```bash
jq '{
  case_count,
  correct,
  accuracy,
  recall,
  confusion_matrix,
  decision_status,
  llm_reasoning_mode
}' artifacts/rca_v2_deepseek32b_vllm/evaluation_summary.json
```

不要只看 accuracy，必须单独看 fiber recall。

### 找错例

```bash
jq '[
  .[]
  | select(.correct == false)
  | {
      case_id,
      actual_label,
      prediction,
      confidence,
      decision_status,
      method1: .KG_RAG_LLM.prediction,
      method2: .KG_RCA.prediction,
      anomalies: [.extracted_anomalies[].anomaly_id]
    }
]' artifacts/rca_v2_deepseek32b_vllm/predictions.json
```

把错例分成：

- 两路都错：检查异常语义或训练关联；
- KG+RAG 对、规则错：检查规则；
- 规则对、KG+RAG 错：检查路径和检索；
- 有一路对但融合错：检查置信度和融合；
- 缺失字段多：先查数据质量。

### 分析 fiber

```bash
jq '[
  .[]
  | select(.actual_label == "fiber")
  | {
      case_id,
      prediction,
      method1_scores: .KG_RAG_LLM.scores,
      method2_scores: .KG_RCA.scores,
      decision_status,
      anomalies: [.extracted_anomalies[].anomaly_id]
    }
]' artifacts/rca_v2_deepseek32b_vllm/predictions.json
```

依次检查：

1. 是否提取出 fiber 特有异常；
2. KG 是否有对应 fiber 边；
3. RAG top-k 是否有 fiber case；
4. 是否命中 fiber 规则；
5. 若规则预测 fiber，是否被融合改掉；
6. 数据字段和标签是否足够可信。

## 14. 现场讲解的最短版本

可以直接这样讲：

> 我们看 case_000268。它在 L2 侧出现 RxLOS、RxLOL 和三个掉底异常。程序先把原始数值转成
> 5 个统一异常节点。KG 用前 200 条训练 case 构建，每条“根因到异常”的边保存出现次数、
> precision、lift 和权重。新 case 沿这些边给三类累积分数，得到 L1 0.644。RAG 再按异常集合
> 找相似训练 case，最相似的一条标签也是 L1。没开 LLM 时直接采用图结果；开 LLM 时把异常、
> 路径和相似 case 交给模型。另一条独立规则通道命中 10 条 L1 规则，也判 L1。两路一致，
> 所以最终是 L1、agreement。L2 看到异常却判 L1 是训练统计关联，不是写死规则，需要领域验证。

现场依次运行：

```bash
python scripts/debug_rca_case.py --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json --channel evidence

python scripts/debug_rca_case.py --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json --channel kg-rag

python scripts/debug_rca_case.py --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json --channel kg-rca

python scripts/debug_rca_case.py --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json --channel full
```

最后说明当前边界：

- 工程链路和真实 LLM 均已跑通；
- 无 LLM 基线为 38/68；
- DeepSeek 32B 为 37/68；
- 两者 fiber recall 都是 0；
- 下一步重点是检查 fiber 数据、异常语义、RAG 和融合，而不是只换更大的 LLM。
