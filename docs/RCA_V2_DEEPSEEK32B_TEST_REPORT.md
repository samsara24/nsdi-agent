# RCA v2 DeepSeek 32B 完整测试报告

## 结论

2026-07-19 已使用本地 `DeepSeek-R1-Distill-Qwen-32B` 完成 RCA v2 的真实 LLM
端到端测试。数据准备、训练、模型序列化、异常图、RAG、guided JSON LLM 推理、
符号规则和冲突融合均可运行。后 68 条中 68/68 为有效 `llm_path_reasoning`，没有
退回确定性路径推理。

工程链路通过，但效果没有超过 `backend=none` 基线：真实 LLM 结果为 37/68
（54.41%），基线为 38/68（55.88%），两者的 fiber recall 都为 0。

## 运行配置

```text
model: DeepSeek-R1-Distill-Qwen-32B
backend: vLLM 0.6.6.post1
dtype: bfloat16
GPU: 2 x NVIDIA RTX A6000 48GB
tensor_parallel_size: 2
gpu_memory_utilization: 0.85
max_model_len: 8192
max_new_tokens: 512
guided_json: true
enforce_eager: true
disable_custom_all_reduce: true
split: first 200 train / last 68 evaluation
```

机器为无 NVLink 的纯 PCIe 多卡拓扑。默认 vLLM custom all-reduce 在 NCCL 初始化后
停滞，因此框架新增了 `--disable-custom-all-reduce`，本次运行同时使用
`NCCL_P2P_DISABLE=1` 和 `NCCL_IB_DISABLE=1`。

完整命令：

```bash
CUDA_VISIBLE_DEVICES=4,5 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
/home/shibinpeng/anaconda3/envs/vllm2/bin/python -m rca_framework.cli train-evaluate \
  --data-dir datasets/rca_v2 \
  --train-size 200 \
  --output-dir artifacts/rca_v2_deepseek32b_vllm \
  --backend vllm \
  --model-path /home/shibinpeng/pretrained_models/DeepSeek-R1-Distill-Qwen-32B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-new-tokens 512 \
  --enforce-eager \
  --disable-custom-all-reduce
```

## 评估结果

| 指标 | `backend=none` | DeepSeek 32B |
|---|---:|---:|
| Accuracy | 38/68, 55.88% | 37/68, 54.41% |
| L1 recall | 45.83% | 45.83% |
| L2 recall | 71.05% | 68.42% |
| fiber recall | 0% | 0% |
| 有效真实 LLM 输出 | 0/68 | 68/68 |

DeepSeek 方法一路的预测分布为 L1=25、L2=43、fiber=0；最终融合分布为
L1=27、L2=41、fiber=0。符号规则一路产生过 3 个 fiber 预测，但融合后没有保留
任何 fiber 结果。与确定性基线相比，最终结果只有 1 条发生变化：
`case_3380daa891a5` 从正确的 L2 变为 L1，因此总正确数减少 1。

这说明当前主要瓶颈不是 LLM 接口或 JSON 解析，而是训练数据中的 fiber 少数类、
异常语义对 fiber 的区分能力，以及方法一路和融合策略对 fiber 证据的利用。
后续实验不能使用这 68 条测试标签直接调参，应另建训练内验证切分或交叉验证。

## 完整性检查

- 自动化测试：7 passed。
- 原始数据：366/366 SHA-256 与归档清单一致。
- 数据集：268 条 schema-v2 case；敏感模式残留均为 0；side violation 为 0。
- 知识图谱：38 个节点、75 条异常边。
- 符号规则：L1=40、L2=40、fiber=35，规则前件交集为 0。
- LLM 输出：68/68 为 `llm_path_reasoning`；无空输出或解析 fallback。
- LLM 路径：所有返回的 `path_ids` 都属于对应 case 实际提取的异常。
- 融合证据：支持证据和冲突证据没有重复项。
- 模型产物：保存后可以重新加载并完成单 case 推理。
- 标签泄漏：目标标签在异常提取、图查询、检索、prompt 和规则匹配前删除。

## 测试中修复的问题

1. 新增 `--disable-custom-all-reduce`，支持纯 PCIe 两卡稳定加载 32B 模型，并将
   该配置写入运行清单。
2. 修复两路结论一致时 `conflicting_evidence` 重复支持证据的问题，并增加回归测试。

## 产物

```text
artifacts/rca_v2_deepseek32b_smoke.json
artifacts/rca_v2_deepseek32b_vllm/evaluation_summary.json
artifacts/rca_v2_deepseek32b_vllm/predictions.json
artifacts/rca_v2_deepseek32b_vllm/run_manifest.json
artifacts/rca_v2_deepseek32b_vllm/model/
```
