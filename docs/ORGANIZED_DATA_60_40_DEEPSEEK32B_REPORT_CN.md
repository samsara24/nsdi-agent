# organized_data 分层 6:4 完整测试报告

## 结论

本次按类别分别进行 6:4 分层划分，固定随机种子为 42。`organized_data` 中实际有
231 条 JSON，其中 211 条满足 RCA v2 的“一端 400G、一端 200G”物理定义并进入
实验；20 条 400G–400G case 按框架既有规则跳过。

使用本地 `DeepSeek-R1-Distill-Qwen-32B`、vLLM 和两张 RTX A6000 完成了全部
85 条测试 case 的真实 LLM 端到端评估。85/85 均产生有效
`llm_path_reasoning`，最终正确 59 条，accuracy 为 **69.41%**。确定性全链路
基线为 58/85（68.24%），真实 LLM 净提升 1 条，即 1.18 个百分点。

当前主要短板仍是 fiber：6 条 fiber 测试 case 全部被判为 L1 或 L2，recall 为 0。
因此 69.41% 主要来自占比较大的 L2 类，不能理解为三类性能都已达标。

## 数据准备与切分

| 类别 | 有效总数 | 训练集 | 测试集 | 实际训练占比 |
|---|---:|---:|---:|---:|
| L1（400G 侧） | 59 | 35 | 24 | 59.32% |
| L2（200G 侧） | 138 | 83 | 55 | 60.14% |
| fiber | 14 | 8 | 6 | 57.14% |
| 合计 | 211 | 126 | 85 | 59.72% |

由于每类样本数不是 5 的倍数，整数切分采用 `round(类别数 × 0.6)`。训练 case
写在数据集前 126 个位置，测试 case 写在后 85 个位置，以兼容现有
`train-evaluate --train-size 126` 入口。切分清单中保存了全部匿名 case ID、
源文件顺序和 train/test 标记。

数据质量检查：

- 输入端点模式：200G–400G 为 211 条，400G–400G 为 20 条。
- 输出标签分布：L1=59、L2=138、fiber=14。
- 训练与测试 case ID 交集为 0；训练 126 个和测试 85 个 ID 均唯一。
- 数据集 manifest 与运行 manifest 中的训练/测试 ID 顺序完全一致。
- 输出中 IPv4、邮箱、手机号、原始接口和 local/remote 旧侧标记残留均为 0。
- 源数据清单 SHA-256 与数据集记录一致。

## 完整评估结果

| 指标 | 确定性基线 | DeepSeek 32B |
|---|---:|---:|
| Accuracy | 58/85（68.24%） | **59/85（69.41%）** |
| L1 recall | 11/24（45.83%） | 11/24（45.83%） |
| L2 recall | 47/55（85.45%） | **48/55（87.27%）** |
| fiber recall | 0/6（0%） | 0/6（0%） |
| 有效真实 LLM 输出 | 0/85 | 85/85 |

DeepSeek 32B 最终逐类指标：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| L1 | 50.00% | 45.83% | 47.83% | 24 |
| L2 | 76.19% | 87.27% | 81.36% | 55 |
| fiber | 0% | 0% | 0% | 6 |
| Macro average | 42.06% | 44.37% | 43.06% | 85 |

DeepSeek 32B 最终混淆矩阵（行是真实标签，列是预测标签）：

| 实际 \ 预测 | L1 | L2 | fiber |
|---|---:|---:|---:|
| L1 | 11 | 13 | 0 |
| L2 | 7 | 48 | 0 |
| fiber | 4 | 2 | 0 |

最终预测分布为 L1=22、L2=63、fiber=0。KG+RAG+LLM 一路的预测分布为
L1=10、L2=75、fiber=0；KG+RCA 符号规则一路为 L1=21、L2=63、fiber=1。
融合后唯一的 fiber 候选没有保留下来。

决策状态分布：

- 两路一致：71 条。
- 冲突由符号规则解决：11 条。
- 冲突由 KG+RAG+LLM 解决：2 条。
- 冲突由加权证据解决：1 条。

与确定性基线相比，共有 3 条最终预测变化：2 条 L2 从错误的 L1 修正为 L2，
1 条 L2 从正确的 L2 变为 L1，净增加 1 条正确结果。最终置信度范围为
0.4661–1.0000，均值为 0.8192。

## 运行配置与完整性

```text
model: DeepSeek-R1-Distill-Qwen-32B
backend: vLLM 0.6.6.post1
dtype: bfloat16
GPU: 2 x NVIDIA RTX A6000 48GB（GPU 4,5）
tensor_parallel_size: 2
gpu_memory_utilization: 0.85
max_model_len: 8192
max_new_tokens: 512
guided_json: true
enforce_eager: true
disable_custom_all_reduce: true
split: stratified 126 train / 85 test, seed 42
```

模型得到 35 个图节点和 66 条边；符号规则数量为 L1=31、L2=40、fiber=28，
规则前件重叠数为 0。评估记录确认测试标签在异常提取、图查询、检索、prompt
和规则匹配前已移除，`label_leakage=false`。项目自动化测试结果为 7 passed。

运行结束时 vLLM/PyTorch 报告了进程组未显式销毁和 1 个 shared-memory 对象清理
警告；进程退出码为 0，85 条结果和所有模型/评估文件均已完整写入，不影响本次指标。

## 产物

```text
datasets/organized_rca_v2_stratified_60_40_seed42/
archive/organized_data_source_manifest_seed42.json
artifacts/organized_rca_v2_60_40_seed42_baseline/
artifacts/organized_rca_v2_60_40_seed42_deepseek32b_vllm/
scripts/prepare_organized_stratified.py
```
