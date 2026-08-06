# NSDI 光链路 RCA 缺陷深度分析

本文只分析缺陷，不修改 `rca_framework/` 现有代码。分析对象是当前 RCA v2 实现、`organized_data` 数据集、以及已有 `artifacts/organized_rca_v2_60_40_seed42_deepseek32b_vllm/` 实验产物。

## 结论摘要

当前系统准确率低并不是单一代码 bug 导致的。更准确的判断是：

1. 代码中确实存在让关键物理证据失效的问题，尤其是 `directional_loss` 对断链哨兵值的过滤和均值聚合。
2. 但即使把这些问题修正，现有字段对 `fiber` 的判别力仍然很弱。实测全特征 RandomForest 5 折交叉验证 accuracy 约 70.14%，`fiber` 的 precision / recall / F1 仍为 0。
3. 当前 DeepSeek-32B vLLM 结果为 59/85，accuracy 69.41%，而多数类基线全判 `L2` 已有 55/85，accuracy 64.71%。系统真正超过先验的部分只有 4 条 case。
4. 因此后续优化重点不应是继续调融合权重或换模型，而应转向证据充分性判断、主动索取证据、人工反馈沉淀和允许弃权。

## 事实基线

### 数据规模与类别分布

`organized_data` 中满足 RCA v2 物理定义的一端 400G、一端 200G case 共 211 条：

| 类别 | 样本数 |
|---|---:|
| `L1` | 59 |
| `L2` | 138 |
| `fiber` | 14 |

当前固定 60/40 分层切分中，测试集为 85 条：

| 指标 | 数值 |
|---|---:|
| 多数类基线，全判 `L2` | 55/85 = 64.71% |
| 当前 DeepSeek-32B vLLM | 59/85 = 69.41% |
| 净收益 | +4 条 case |
| `fiber` recall | 0/6 = 0% |

### 信息上界实测

用当前代码未使用的字段也纳入特征，包括 `bias`、`Temperature`、`Voltage`、逐 lane 方向配对损耗、断 lane 计数等，构造 76 维全特征并运行 RandomForest-300 的 5 折交叉验证：

| 模型 | 5 折 accuracy |
|---|---:|
| 多数类 Dummy | 65.40% |
| RandomForest-300 | 70.14% |

RandomForest 的逐类结果：

```text
              precision    recall  f1-score   support

          L1      0.571     0.407     0.475        59
          L2      0.734     0.899     0.808       138
       fiber      0.000     0.000     0.000        14
```

这说明现有字段的信息量不足以稳定识别 `fiber`。如果只在当前字段上做算法替换，预期收益非常有限。

## A 类：物理判据失效

### A1. `directional_loss()` 过滤断链哨兵值，导致光损耗证据消失

代码定位：`rca_framework/anomaly.py`

```python
def directional_loss(case: Dict[str, Any], source: str, target: str) -> Optional[float]:
    tx = metric_values(case, "txpower", source, healthy_only=True)
    rx = metric_values(case, "rxpower", target, healthy_only=True)
    if not tx or not rx:
        return None
    return abs(mean(tx) - mean(rx))
```

触发机制：

- `metric_values(..., healthy_only=True)` 会过滤 `<= DOWN_THRESHOLDS[metric]` 的值。
- 对 `rxpower` 来说，断光哨兵值是 `-40`，阈值是 `-39`。
- 因此最重要的断光证据会在计算方向损耗前被删除。
- 如果某个方向的 RX lane 全部断光，`rx` 为空，`directional_loss()` 直接返回 `None`。
- 如果只有部分 lane 断光，断光 lane 被删除，只剩健康 lane 参与均值，损耗会被严重低估。

具体样本：`organized_data/fiber_26/1019.json`

该样本中 `local` 是 200G，即规范化后的 `L2`；`remote` 是 400G，即规范化后的 `L1`。关键字段：

```json
"txpower": {
  "local": {
    "0": 0.62,
    "1": 0.62,
    "2": 0.62,
    "3": 0.62
  }
},
"rxpower": {
  "remote": {
    "0": 0.84,
    "1": 0.66,
    "2": 0.68,
    "3": -40
  }
}
```

按物理含义，`L2` 发送端正常，`L1` 接收端同 lane 断光，lane3 的方向损耗约为 `0.62 - (-40) = 40.62 dB`。但当前代码会先删除 `-40`，然后用剩余三条 RX lane 算均值：

```text
tx_mean = 0.62
rx_mean = mean(0.84, 0.66, 0.68) = 0.7267
directional_loss = abs(0.62 - 0.7267) = 0.1067
```

而模型中学到的 `L2_to_L1` 损耗阈值为 3.42dB，因此该 `fiber` case 不会触发方向损耗异常。

实测影响：

- 在当前已有 85 条测试预测中，`directional_loss:*` 和 `bidirectional_loss:*` 的触发次数为 0。
- 这两类异常本应是 `fiber` 的核心物理证据，却在现有 artifact 中完全没有进入 KG 路径和规则匹配。
- `fiber` 测试集 6 条全部被判为 `L1` 或 `L2`。

成因溯源：

该函数把 `directional_loss` 当成“健康信号的均值差异”来计算，但链路故障的核心证据往往正是断光哨兵值。`healthy_only=True` 对阈值拟合可能有意义，对故障检测本身则会删除最强证据。这是训练分布统计和故障物理判据混在同一个 helper 中造成的抽象错误。

修复代价与收益上限：

- 修复代价：中等。需要把方向损耗从均值差改成逐 lane 配对，并区分“健康 baseline 统计”和“故障证据提取”两个语义。
- 收益上限：有限。实测“发端 lane 正常 + 对端同 lane 收光断”这一签名的命中情况为 `L1` 14/59、`L2` 42/138、`fiber` 8/14。换算后，该签名的 `fiber` 命中率为 57.14%，但 precision 只有约 12.5%。它有信号，但不足以单独把 `fiber` 分出来。

### A2. 均值聚合抹平 lane 级故障

代码定位仍是 `rca_framework/anomaly.py` 的 `directional_loss()`。

触发机制：

- 光链路故障常见形态是单 lane 或少数 lane 异常。
- 当前函数用 `mean(tx) - mean(rx)` 聚合所有 lane。
- 单 lane 40dB 损耗会被 3 条正常 lane 稀释。
- 如果先叠加 A1 的 `healthy_only=True`，断 lane 甚至不会进入均值。

实测影响：

在 211 条有效样本上，逐 lane 配对的最大方向损耗分布如下：

| 指标 | `L1` | `L2` | `fiber` |
|---|---:|---:|---:|
| `L2_to_L1` maxloss > 20dB | 10% | 27% | 36% |
| `L1_to_L2` maxloss > 20dB | 14% | 4% | 21% |

方向配对确实有一定物理信号，但 `L1`/`L2` 中也大量存在同类模式。说明均值聚合是缺陷，但逐 lane 修正不能直接解决三分类。

成因溯源：

当前异常抽象是“case 级异常节点”，没有显式保留 lane 作为一等实体。结果是 lane 对齐关系在进入 KG 前已经被压缩掉。对于光链路 RCA，这属于过早聚合。

修复代价与收益上限：

- 修复代价：中等偏高。需要扩展 `Anomaly` 的 evidence 表达，至少保存 lane id、方向、source/target、tx/rx 值和损耗。
- 收益上限：受数据本身限制。逐 lane 方向证据对 `fiber` 的 lift 约 1.9，但不是强判据。

## B 类：异常检测阈值体系失准

### B1. 在全故障样本上拟合“离群”阈值

代码定位：`rca_framework/anomaly.py`

```python
def fit_thresholds(cases: Sequence[Dict[str, Any]]) -> ThresholdModel:
    values: Dict[str, List[float]] = defaultdict(list)
    spreads: Dict[str, List[float]] = defaultdict(list)
    losses: Dict[str, List[float]] = defaultdict(list)
    for case in cases:
        ...
                healthy = metric_values(case, metric, side, healthy_only=True)
                values[key].extend(healthy)
```

触发机制：

- 训练集不是健康样本集合，而是已经发生故障的 `L1` / `L2` / `fiber` 样本集合。
- `robust_fence()` 在故障分布上用 IQR 拟合“正常范围”。
- 如果某类故障在训练集中很常见，它的异常值会被吸收到“正常范围”里。
- 3.0 倍 IQR 又进一步放宽边界。

实测阈值示例：

```text
L1:rxpower fence = [-3.93, 6.01]
L2:rxpower fence = [-3.50, 5.41]
L1:media_snr fence = [23.18, 28.045]
L2:media_snr fence = [23.345, 28.00]
L1_to_L2 loss_upper = 3.10875
L2_to_L1 loss_upper = 3.42
```

后果链条：

1. 阈值过宽。
2. 许多故障样本不触发任何 outlier。
3. 测试集中 21/85（24.71%）提取到 0 个异常。
4. 零异常 case 的图路径为空、规则匹配为空。
5. 系统退化为按先验和默认 tie-breaker 判 `L2`。

实测影响：

```text
zero_anomaly_cases = 21/85
zero_anomaly_actual = {'L1': 7, 'L2': 14}
zero_anomaly_prediction = {'L2': 21}
zero_anomaly_correct = 14
```

这 14 条正确的 `L2` 不是方法识别出来的，而是多数类先验碰巧命中。

成因溯源：

`fit_thresholds()` 的设计意图是避免测试标签泄漏，使用训练集自适应阈值。但它把“无标签健康基线估计”和“有标签故障样本统计”混为一谈。RCA 任务需要的是健康基线、设备规格阈值或历史时序基线，而不是从故障集合中反推正常范围。

修复代价与收益上限：

- 修复代价：高。需要新增健康窗口、历史时序、设备规格或人工阈值库。仅调小 IQR 倍数会增加误报，不保证提升 RCA。
- 收益上限：取决于新数据源。若仍只使用当前单点故障快照，最多只能改善“零异常”比例，但不能解决 `fiber` 和设备侧故障模式高度重叠的问题。

## C 类：打分与规则学习的统计缺陷

### C1. `severity` 跨异常类型量纲不可比

代码定位：`rca_framework/graph.py`

```python
path_score = edge.weight * max(0.25, min(3.0, item.severity))
raw_scores[edge.root_cause] += path_score
```

触发机制：

不同异常类型的 `severity` 来源不同：

- `status_fault` 固定为 1.0。
- `signal_drop` 是 down lane 比例，范围约 0 到 1。
- `low_outlier` 是 `(low - min) / (abs(low) + 1)`，通常是很小的小数。
- `lane_imbalance` 是 `spread / spread_limit`，可能大于 1。
- `directional_loss` 是 `value / limit`，但当前几乎不触发。

这些量没有统一校准，却被直接乘到 KG 边权重上。结果是分数不仅反映“特征与类别的统计关系”，还混入了不同异常构造公式的尺度差异。

实测影响：

当前最高频、最强势的证据主要是 `status_fault:*`、`signal_drop:*` 和 `low_outlier:*:serdes_snr`。其中 `status_fault` 因为 severity 恒为 1.0，天然比小数型 outlier 更稳定；`lane_imbalance` 又可能因为倍率较大获得额外优势。该缺陷会放大规则偶然性，影响 L1/L2 分界稳定性。

成因溯源：

代码把“异常严重程度”和“类别判别力”混在同一个乘法项里。对于 RCA，严重程度大不等于更指向某个根因；例如大面积断 lane 既可能是设备侧，也可能是链路侧。

修复代价与收益上限：

- 修复代价：中等。需要把 severity 映射为同一概率尺度，或只在同类型异常内部比较 severity。
- 收益上限：主要改善置信度校准和 L1/L2 分界，难以解决 `fiber` recall。

### C2. 符号规则打分对不平衡类别有结构性偏置

代码定位：`rca_framework/rules.py`

```python
confidence = hit / total_count
lift = confidence / self.priors[label] if self.priors[label] else 0.0
discriminative = confidence * math.log1p(max(0.0, lift))
```

触发机制：

- `confidence` 是某 antecedent 命中某类的条件概率。
- 在 `L2` 占比 65% 的数据中，许多普通异常天然更容易获得高 `confidence`。
- `fiber` 先验极小，因此 lift 可能很高，但 `hit` 只有 2 时，规则非常脆弱。
- `confidence * log1p(lift)` 同时奖励多数类的高 confidence 和少数类的小样本高 lift，容易形成两类问题：多数类支配总体决策，少数类规则看似强但不可泛化。

实测影响：

在当前 60/40 split 中：

```text
fiber 全量 = 14
fiber 训练 = 8
fiber 测试 = 6
```

模型中的 `fiber` 规则大量建立在 `n=2` 上：

```text
['coupled_fault:L1_to_L2:tx_rx', 'low_outlier:L1:serdes_snr']
confidence = 0.67
lift = 10.50
matched_training_cases = 2
```

这类规则在训练集里看起来 lift 很高，但测试集 6 条 `fiber` 全部未被最终预测为 `fiber`。

成因溯源：

规则学习试图用统计特征自动发现领域规则，但 `fiber` 样本数过少，统计学习没有足够支撑。对少数类，应该引入专家规则、物理判据和证据充足性门控，而不是依赖两两组合的训练集频次。

修复代价与收益上限：

- 修复代价：中等到高。需要重写规则选择目标，加入置信区间、最小正例数、反事实覆盖和人工规则优先级。
- 收益上限：如果没有新增 `fiber` 证据，自动规则学习仍受 14 条样本限制。

### C3. `serdes_snr` 量纲与语义不一致

代码定位：`rca_framework/anomaly.py`

```python
METRIC_ALIASES = {
    ...
    "serdes_snr": ("serdes_snr", "serdesSNR", "SerdesSNR"),
}
DOWN_THRESHOLDS = {
    ...
    "serdes_snr": 0.0,
}
```

触发机制：

样本中的 `serdes_snr` 取值常见为数十万，例如 `823041`、`799445`、`122146`。这不是普通意义上的 SNR dB 量纲，却被作为和 `media_snr`、`host_snr` 同类的 SNR 处理。低值 outlier 被解释为“SerDes 信噪比偏低”，但真实物理含义不明确。

实测影响：

`low_outlier:L1:serdes_snr` 是当前系统中最重要的 L2 证据之一：

```text
测试集频次：24
L2 测试集中出现：17
训练模型中 L2 KG rule：matched_training_cases = 37
```

也就是说，一个量纲可疑的字段偶然成为最强判别特征之一。这类缺陷危险在于：它能提高当前 split 的准确率，却削弱跨设备、跨版本和跨采集系统的泛化可信度。

成因溯源：

字段命名被直接当作物理语义，没有做单位核验和采集来源核验。RCA 系统里，指标单位与采集口径必须是知识层的一部分。

修复代价与收益上限：

- 修复代价：中等。需要确认 `serdes_snr` 的真实单位、缩放方式、设备来源和健康范围。
- 收益上限：短期可能降低当前 split 的 accuracy，因为它现在提供了伪相关；长期有助于提高可信度。

## D 类：架构空转

### D1. LLM 被候选分数锚定，基本复制图推理结果

代码定位：`rca_framework/llm.py`

```python
"candidate_path_scores": graph_result["scores"],
"root_cause_paths": graph_result["paths"],
"candidate_feature_profile_scores": graph_result.get("feature_profile_scores", {}),
"matched_kg_feature_rules": graph_result.get("matched_feature_rules", {}),
```

触发机制：

Prompt 中直接给出 KG 的候选分数、路径和已匹配规则。LLM 的任务表面上是“推理”，实际输入已经强烈暗示应选择哪个 label。在没有外部新证据的情况下，LLM 很难推翻 KG。

实测影响：

DeepSeek-32B vLLM 对 85 条全部产生合法 JSON，但最终只相对确定性基线净增 1 条正确结果：

```text
确定性基线：58/85 = 68.24%
DeepSeek-32B：59/85 = 69.41%
真实 LLM 输出：85/85
fiber recall：0 -> 0
```

已有报告显示最终预测变化仅 3 条：2 条 L2 从错误 L1 修正为 L2，1 条 L2 从正确 L2 改错为 L1。

成因溯源：

LLM 被放在“解释 KG 输出”的位置，而不是“提出证据缺口、挑战候选路径、请求额外信息”的位置。这与框架图中“协同推理与信息校准”的目标不一致。

修复代价与收益上限：

- 修复代价：中等。需要重写 prompt 与输出 schema，把 LLM 的职责从直接三分类改为审查证据充分性、生成冲突解释和提出补证据请求。
- 收益上限：若不引入新证据，LLM 仍不会突破 70% 左右的信息上界。

### D2. 融合模块多数时间空转

代码定位：`rca_framework/fusion.py`

```python
if first_pred == second_pred:
    prediction = first_pred
    status = "agreement"
```

触发机制：

KG/RAG/LLM 与符号规则都从同一套 `extract_evidence()` 异常集合出发，训练数据也相同。两路方法看似独立，实质共享输入和大量统计假设，因此很容易一致。一旦一致，后续冲突消解、置信度差、加权证据等逻辑都不会起作用。

实测影响：

在 DeepSeek-32B vLLM 85 条测试中：

```text
agreement = 71
conflict_resolved_by_symbolic_rules = 11
conflict_resolved_by_kg_rag_llm = 2
conflict_resolved_by_weighted_evidence = 1
manual_review_recommended = 0
```

确定性基线中 agreement 更高，为 82/85。说明系统的“协同”更多是同源证据的重复确认，不是真正的异构推理。

成因溯源：

两路方法没有形成互补证据源。KG 规则、RAG 检索、符号规则都建立在同一异常 id 集合上，差异只是打分方式。

修复代价与收益上限：

- 修复代价：中等。需要让至少一路引入不同证据，例如历史工单、时序趋势、OTDR、FEC 误码、拓扑路径。
- 收益上限：如果继续共享同一异常集合，融合层调参收益不超过 1 个百分点。

### D3. `missing_information` 没有闭环

代码定位：`rca_framework/fusion.py`

```python
missing = list(dict.fromkeys(case.missing_fields + method1.get("missing_information", [])))
...
"missing_or_requested_fields": missing
```

触发机制：

系统会把缺失字段写入结果，但没有任何后续动作使用这些字段。缺失信息不会触发重新采集、人工复核、弃权或二次推理。

实测影响：

当 21/85 case 没有异常时，系统仍强制输出 `L1`/`L2`/`fiber` 三分类结果，而不是把“证据不足”升级为主决策状态。这使 accuracy 指标掩盖了证据不足问题。

成因溯源：

当前 pipeline 是一次性分类器，而不是诊断流程。框架图中的“信息校准”和“人工反馈”没有进入控制流。

修复代价与收益上限：

- 修复代价：中等。需要在 `infer` 层引入证据充分性判断和 abstention。
- 收益上限：单纯 accuracy 可能下降，但可信度和可用性会提升。正确指标应改为覆盖率-精度曲线。

## E 类：不建议继续投入的方向

### E1. 接入 `bias`、温度、电压、告警名、厂商

实测结果：

| 字段 | `L1` 中位数 | `L2` 中位数 | `fiber` 中位数 |
|---|---:|---:|---:|
| `L1_bias_max` | 7.29 | 7.35 | 7.385 |
| `L2_bias_max` | 7.35 | 7.395 | 7.33 |
| `L1_temp` | 47.72 | 46.26 | 47.31 |
| `L2_temp` | 49.77 | 50.93 | 50.69 |
| `L1_volt` | 3.25 | 3.25 | 3.255 |
| `L2_volt` | 3.29 | 3.32 | 3.30 |

`alarm_name` 也几乎无判别力：211 条中 162 条都是 `DCN-AI超平面-接口降lane-网络侧`，其中 `L1=46`、`L2=104`、`fiber=12`，类别比例接近总体比例。

结论：这些字段可以作为解释背景，但不应作为下一轮准确率优化重点。

### E2. 继续换更强分类模型

全特征 RandomForest 已经把当前代码未用字段全部纳入，accuracy 也只有 70.14%，且 `fiber` 仍为 0。继续换 XGBoost、神经网络或更大 LLM，除非引入新证据，否则大概率只是在 65% 到 70% 之间震荡。

### E3. 继续调融合权重和规则数

当前错误主要来自证据不可分，而不是最后一步融合。融合权重、规则上限、`manual_review_margin` 等超参只能改变少量冲突样本，不会改变 21 条零异常和 `fiber` 无强判据的事实。

## 缺陷优先级建议

| 优先级 | 缺陷 | 是否值得修 | 原因 |
|---|---|---|---|
| P0 | `directional_loss` 删除断链哨兵值 | 值得，但不应期待大幅涨点 | 它是明确 bug，会让物理证据失效；但单独修复 precision 不高 |
| P0 | 均值聚合抹平 lane 故障 | 值得，但需要配合证据充分性 | 逐 lane 是正确物理建模方式 |
| P1 | 阈值体系缺少健康基线 | 值得，但需要新数据源 | 没有健康 baseline 时无法可靠定义 outlier |
| P1 | LLM 只复制 KG | 值得重构职责 | 应改为审查证据与生成补证据请求 |
| P2 | 融合权重调参 | 不建议优先 | 信息源同质，收益很小 |
| P2 | 接入 bias/温度/电压 | 不建议优先 | 实测判别力弱 |

## 对后续 Agent 化方案的影响

本缺陷分析直接推导出 Agent 化探索方案的设计边界：

1. Agent 不应只是把 KG、规则、LLM 包成更复杂的三分类器。
2. Agent 必须有 `assess_sufficiency`：判断当前证据是否足够支持三分类。
3. Agent 必须有 `request_evidence`：在证据不足时输出结构化补采清单。
4. 历史故障沉淀 Skill 的价值高于继续从 8 条 `fiber` 训练样本中挖统计规则。
5. 评估指标应从单一 accuracy 改为覆盖率-精度曲线和证据请求质量。

简言之，当前系统的问题不是“不够聪明”，而是“在证据不足时仍被迫给出一个三分类答案”。下一阶段应把 RCA 从分类器改造成诊断流程。

## 复现入口

本报告中的主要诊断数字可由新增脚本复现：

```bash
cd /home/chenziang/nsdi
python scripts/diagnose_dataset.py
python scripts/summarize_runs.py
```

若只需要快速输出而不运行 RandomForest 交叉验证：

```bash
python scripts/diagnose_dataset.py --skip-supervised
```
