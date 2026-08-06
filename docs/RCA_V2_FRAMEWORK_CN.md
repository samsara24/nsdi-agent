# RCA v2 框架说明

## 1. 目标

框架面向 400G–200G 光链路故障定界，最终只输出三类根因：

- `L1`：400G 端口或其设备侧故障；
- `L2`：200G 端口或其设备侧故障；
- `fiber`：L1 与 L2 之间的光纤或链路介质故障。

整体采用两条相互独立但共享异常语义的推理链：

```text
                              ┌─> 异常知识图谱 -> 路径检索/RAG -> LLM 路径推理 ─┐
脱敏 case -> L1/L2 归一化 -> 异常行为提取                                      ├─> 融合决策
                              └─> 三套互斥符号规则 -> 规则匹配与评分 ───────────┘
```

第一条链强调图路径、相似案例和 LLM 解释能力；第二条链强调可验证的符号规则。二者一致时相互补全，不一致时进入显式的冲突解决流程。

## 2. 数据层

### 2.1 原始数据保护

原始 `data/` 只作为输入，流水线拒绝覆盖已有输出目录。每个原始文件的文件名、大小和 SHA-256 被写入归档清单，因此即使当前目录没有可用 Git 历史，也可以验证原始数据是否变化。

### 2.2 L1/L2 统一定义

系统不再使用相对的 local/remote 作为模型标签，而使用固定物理含义：

```text
400G endpoint -> L1
200G endpoint -> L2
link medium   -> fiber
```

如果原 case 为 local=200G、remote=400G，系统会交换整个端点作用域，而不是只修改标签。交换内容包括所有 lane 指标、LOS/LOL 状态、端口信息、温度、电压、厂商和序列号等。原 local 根因随 200G 端点变为 L2，原 remote 根因随 400G 端点变为 L1。

### 2.3 脱敏策略

IP、真实接口、序列号、厂商、区域、拓扑位置、任务/设备标识和时间原点均被删除、抽象或使用带密钥 HMAC 生成稳定假名。最终端口只保留物理角色：

```text
L1_ENDPOINT--400G_PORT
L2_ENDPOINT--200G_PORT
```

当前 366 条原始数据中，268 条满足一端 400G、一端 200G；另外 98 条为 400G–400G，不强行映射，保留在原数据和 skipped 清单中。

## 3. 统一异常语义层

所有阈值只在训练集拟合。框架使用稳健分位数/IQR 学习不同端点、不同指标的正常范围，然后把新 case 转换为类型化异常名词，例如：

- 信号中断 `SignalDrop`；
- 信号偏低或偏高 `LowSignal` / `HighSignal`；
- lane 不均衡 `LaneImbalance`；
- LOS/LOL 状态异常 `DeviceStatusFault`；
- 单向光损耗 `DirectionalLoss`；
- TX/RX 耦合异常 `CoupledTxRxFault`；
- 双向光损耗 `BidirectionalLoss`。

缺失数据不作为异常边写入知识图谱，而是单独记录为 evidence coverage，并在最终结果中给出需要补采的信息。

## 4. 方法一：KG + RAG + LLM

### 4.1 图结构

知识图谱以三个根因标签为中心：

```text
root_cause:L1
root_cause:L2
root_cause:fiber
```

图中只有异常行为节点能够与根因形成边，正常状态不会进入图。边使用不同关系名描述异常语义，例如：

```text
HAS_SIGNAL_DROP
HAS_LANE_IMBALANCE
HAS_STATUS_FAULT
HAS_DIRECTIONAL_LOSS
HAS_COUPLED_TX_RX_FAULT
HAS_BIDIRECTIONAL_LOSS
```

每条边保存训练样本数、`P(异常|根因)`、`P(根因|异常)`、lift 和综合权重。

### 4.2 新 case 路径推理

新 case 的异常被投影为以下路径：

```text
query case -> EXHIBITS -> anomaly -> INDICATES -> root cause
```

系统汇总所有匹配路径，为 L1/L2/fiber 分别计算候选分数。同时使用 IDF 加权的异常 Jaccard 相似度，从训练集中检索相似 case。检索过程只读取训练 case 标签，不读取目标 case 标签。

### 4.3 LLM 的作用

LLM 接收目标异常、候选路径、边统计和训练集检索案例，并输出预测、置信度、使用的路径和缺失信息。目标真实标签在构造 prompt 前已经删除。

框架支持 `vllm` 和 `transformers` 后端。当前冻结基线使用 `backend=none`，即使用完全可复现的确定性路径打分，同时保存完整 prompt；因此当前基线用于证明框架连通性，并不代表真实 LLM 的最终效果。

## 5. 方法二：KG + RCA 符号规则

符号方法使用相同的异常节点定义，但不复用方法一的路径分数。训练阶段枚举：

- 单一异常规则；
- 两个异常同时出现的合取规则。

每个规则前件根据 confidence、lift、support 和相对其他类别的 exclusivity margin，唯一分配给 L1、L2 或 fiber。一个前件最多属于一个规则集，因此三套规则在结构上保证不重合。

新 case 到来时，只有规则的全部 `all_of` 前件都出现才算匹配。匹配分由规则强度、匹配数量和异常覆盖率共同决定，并输出匹配规则明细。

当前前 200 条训练数据生成：

```text
L1 rules:    40
L2 rules:    40
fiber rules: 35
规则交集:     0
```

## 6. 双路结论融合

### 6.1 两路一致

如果 KG+RAG+LLM 与 KG+RCA 输出相同标签：

1. 保留一致标签；
2. 合并图路径和符号规则；
3. 使用相似案例补充上下文；
4. 输出缺失指标和建议补采信息；
5. 决策状态标记为 `agreement`。

### 6.2 两路不一致

冲突处理分为三步：

1. 如果某一路校准置信度比另一路高至少 0.20，采用明显更强的一路；
2. 否则对三类候选分数做加权融合，方法一权重为 0.55，方法二权重为 0.45；
3. 如果融合后第一、第二候选分差仍小于 0.10，保留暂定的 L1/L2/fiber 结果，同时标记 `manual_review_recommended`。

冲突结果同时保存支持证据和反对证据，避免只给最终标签却隐藏另一条推理链。

## 7. 输出结构

单 case 推理结果包含：

```text
prediction                 最终 L1/L2/fiber 标签
confidence                 最终置信度
decision_status            一致、冲突解决或建议人工复核
KG_RAG_LLM                 路径、检索案例、LLM/确定性推理结果
KG_RCA                     三类规则分数及命中规则
extracted_anomalies        新 case 的异常名词节点
supporting_evidence        支持最终结论的路径和规则
conflicting_evidence       与最终结论冲突的证据
information_completion     数据覆盖率、缺失字段和建议补采项
```

## 8. 当前状态

框架的代码、数据流水线、模型序列化、单 case 推理、固定切分评估和冲突融合已经连通。自动化测试覆盖：

- 200G/400G 反向 case 的整侧交换；
- 原始数据不被覆盖；
- 图中只存在异常边；
- 三套规则零重叠；
- 修改目标标签不能改变推理结果；
- 低分差冲突必须触发人工复核建议。

当前不加载 LLM 的后 68 条连通性基线准确率为 55.88%，fiber recall 为 0。下一阶段应在不读取测试标签的前提下，重点研究 fiber 少数类、时间切分、规则阈值、类别平衡以及真实 LLM 的路径决策能力。
