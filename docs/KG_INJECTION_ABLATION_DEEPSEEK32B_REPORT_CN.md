# KG 分层注入与独立 LLM 打分消融实验报告

配套代码说明：`docs/KG_INJECTION_EXPERIMENT_CODE_CHANGES_CN.md`。

> **状态：本实验所用代码已于 2026-08-06 回退，实验结论与产物全部有效。**
> 第 8 节列出的 `rca_framework/llm.py`、`pipeline.py`、`cli.py` 及两个消融脚本，
> 现位于 `archive/rca_framework_snapshot_20260806_layered_injection/`，
> 不再是活动代码。本实验验证过的四项设计将以 Agent 形态重新引入，
> 落点见 `docs/AGENT_REFACTOR_MODULE_STRATEGY_CN.md` 第 6 节。

## 1. 结论

本次在 `organized_rca_v2_stratified_60_40_seed42` 固定分层切分上，使用
DeepSeek-R1-Distill-Qwen-32B 对 85 条测试 case 完成四组端到端消融。实验结论是：

1. **改造前行为被精确复现。** `full + legacy` 得到 59/85（69.41%），与历史
   DeepSeek-32B 结果完全一致，三类 recall 和混淆矩阵也一致。
2. **分层注入没有提升强制三分类 accuracy。** 推荐配置 `layered + llm_only`
   为 58/85（68.24%），比原实现少正确 1 条；L1/L2/fiber recall 分别为
   45.83%、85.45%、0。
3. **改造改善了“不确定性表达”，但没有产生新的可判别证据。** 分层 prompt
   把 22 条 `uncovered` 和 2 条 `partial` 标成 `insufficient`；这些 case 的
   accuracy 为 14/24（58.33%），而 `sufficient` 子集为 44/61（72.13%）。
4. **多数类倾向有所减弱，但 fiber 问题完全未解决。** LLM 路的 L2 预测从
   75 条降到 64 条、L1 从 10 条增到 21 条，但仍然没有任何 fiber 预测。
5. **不能据此把原先的 KG 先验注入描述成纯粹有害。** 在当前强制分类指标下，
   先验恰好对 `uncovered` 中占多数的 L2 有利；去掉 KG 二次打分还使一个
   `covered` L1 case 从正确变为错误。更合理的落地方式是把分层注入用于
   `abstain/request_evidence` 门控，而不是期待它单独提高三分类 accuracy。

因此，建议暂时保留 `full + legacy` 作为准确率基线，将 `layered + llm_only`
作为影子诊断/证据充分性实验配置。下一步优先评估选择性分类和补证据质量，不应
直接把后者替换成生产默认分类器。

## 2. 改动与消融设计

### 2.1 改造前实现

原实现对所有 case 无条件向 LLM 注入：

- `candidate_path_scores`
- `root_cause_paths`
- `candidate_feature_profile_scores`
- `matched_kg_feature_rules`
- `retrieved_training_cases`

LLM 输出后，又将 KG 分数以 0.35 权重混入 LLM 路，再把该路与符号规则路融合。
这使 KG 分数在最终决策中被重复使用。

### 2.2 最小改造

改造增加两组正交开关：

- `kg_injection=full|layered`
  - `full`：完整复现改造前 prompt。
  - `layered`：根据 KG 覆盖状态选择注入字段。
- `llm_score_mode=legacy|llm_only`
  - `legacy`：保留 0.35 KG 分数回灌。
  - `llm_only`：LLM 路分数只由 LLM 的 prediction/confidence 构造。

KG 覆盖状态不使用人工调参阈值，而由查询结构直接定义：

- `covered`：至少命中一条 KG feature rule，允许注入聚合候选分数、匹配规则和路径。
- `partial`：有原子异常路径但无组合规则，只注入原子路径统计，屏蔽聚合候选分数。
- `uncovered`：无 KG 路径、无规则，屏蔽候选分数和路径，仅保留目标遥测、物理定义及
  “无结构证据”的显式说明。

分层输出 schema 新增 `evidence_sufficiency=sufficient|insufficient`。本次
`insufficient_confidence_scale=1.0`，即只记录充分性，不人为降低融合置信度，
以便将 accuracy 变化归因于 prompt 与打分方式本身。

### 2.3 四组实验

| 组别 | KG prompt | LLM 路打分 | 用途 |
| --- | --- | --- | --- |
| `full__legacy` | 全量注入 | KG 0.35 回灌 | 改造前复现基线 |
| `full__llm_only` | 全量注入 | 仅 LLM | 单独验证去除二次 KG 打分 |
| `layered__legacy` | 分层注入 | KG 0.35 回灌 | 单独验证 prompt 分层 |
| `layered__llm_only` | 分层注入 | 仅 LLM | 推荐配置 |

四组共享同一次模型加载，数据顺序、模型、采样温度、guided JSON 和融合参数均不变。

## 3. 实验配置与完整性

数据切分：

- 有效 case：211
- 训练集：126
- 测试集：85
- 固定分层随机种子：42
- 测试标签只在推理完成后用于评估

模型与运行时：

```text
model: DeepSeek-R1-Distill-Qwen-32B
backend: vLLM 0.11.0
dtype: bfloat16
GPU: 2 x NVIDIA RTX A6000 48GB（GPU 6,7）
tensor_parallel_size: 2
gpu_memory_utilization: 0.85
max_model_len: 8192
max_new_tokens: 512
guided_json: true
enforce_eager: true
disable_custom_all_reduce: true
temperature: 0
```

当前主机上 NCCL 2.27.3 的默认 P2P/SHM 通道在 GPU 6、7 间初始化死锁。独立
`all_reduce` 测试确认问题后，正式实验使用：

```text
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_IB_DISABLE=1
VLLM_USE_V1=1
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

禁用后两卡 `all_reduce` 正常完成。模型权重加载耗时 724.5 秒，每卡加载
30.73 GiB 权重，并分配 7.86 GiB KV cache。四组总墙钟时间约 41.6 分钟。

实验结束后进程退出码为 0；GPU 6、7 均回落到 6 MiB、0% 利用率，无
`run_injection_ablation`、`Worker_TP` 或模型服务残留。

## 4. 总体结果

| 组别 | 正确数 | Accuracy | L1 recall | L2 recall | fiber recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full__legacy` | **59/85** | **69.41%** | 45.83% | **87.27%** | 0 |
| `full__llm_only` | 58/85 | 68.24% | 41.67% | **87.27%** | 0 |
| `layered__legacy` | 58/85 | 68.24% | 45.83% | 85.45% | 0 |
| `layered__llm_only` | 58/85 | 68.24% | 45.83% | 85.45% | 0 |

推荐配置 `layered__llm_only` 的混淆矩阵：

| 实际 \ 预测 | L1 | L2 | fiber |
| --- | ---: | ---: | ---: |
| L1 | 11 | 13 | 0 |
| L2 | 8 | 47 | 0 |
| fiber | 4 | 2 | 0 |

四组的 LLM 路自身 accuracy 都是 56/85（65.88%）。也就是说，prompt 分层改变了
预测类别分布，但正确数没有变化；最终 1 条 accuracy 差异来自融合边界。

## 5. 消融结果解释

### 5.1 分层注入改变了 LLM 倾向，但只改变 1 条最终结论

`full__legacy` 与 `layered__legacy` 对比：

- LLM 路有 11/85 条 prediction 变化。
- LLM 路 L2 数量从 75 降到 64，L1 从 10 增到 21。
- 最终融合只改变 1 条 case。
- 该 case 为 `case_b81d18ac89b5`，实际标签 L2、KG 状态 `uncovered`；
  原实现预测 L2，分层 prompt 预测 L1，并标记 `insufficient`，因此从正确变错误。

这说明当前融合器会吸收大部分 prompt 变化；仅改 prompt 无法显著改变最终决策。
同时，这条退化也不是“新证据推错”，而是在没有结构证据时从多数类先验猜 L2
变为无先验猜 L1。若评价仍要求每条 case 强制三分类，类别先验天然占优。

### 5.2 去除二次 KG 打分没有带来收益

在全量 prompt 下，`full__legacy` 与 `full__llm_only` 的 85 条原始 LLM 输出完全
相同，因此是干净的打分消融：

- LLM prediction 变化：0
- 最终 prediction 变化：1
- accuracy：59/85 → 58/85

变化 case 为 `case_3e392e75f20c`，实际 L1、KG 状态 `covered`。LLM 自身预测
L2，但 legacy 的 KG 分数回灌使最终融合保留 L1；去掉回灌后变成 L2。

因此，“LLM 路必须完全去 KG 分数”并不适用于所有覆盖状态。在 `covered` 场景，
KG 聚合分数确有有效模式依据。后续可以改成条件打分：

- `covered`：允许图分数参与校准。
- `partial/uncovered`：LLM 路不回灌图分数。

不过按本次逐 case 结果，这种条件打分仍不会提高总体 accuracy，因为分层 prompt
在 `uncovered` 上损失的那一条仍然存在。

### 5.3 分层注入提供了有效但有限的充分性信号

测试集 KG 覆盖分布：

| 状态 | case 数 | 实际标签分布 | 推荐配置 Accuracy |
| --- | ---: | --- | ---: |
| `covered` | 52 | L1=16, L2=30, fiber=6 | 37/52（71.15%） |
| `partial` | 11 | L1=1, L2=10 | 7/11（63.64%） |
| `uncovered` | 22 | L1=7, L2=15 | 14/22（63.64%） |

22 条 `uncovered` 中，21 条完全未提取到异常；另 1 条虽有异常，但异常节点未进入
训练 KG。分层 LLM 将全部 22 条 `uncovered` 和 2 条 `partial` 标成
`insufficient`：

| 充分性 | case 数 | 正确数 | Accuracy |
| --- | ---: | ---: | ---: |
| `sufficient` | 61 | 44 | 72.13% |
| `insufficient` | 24 | 14 | 58.33% |

如果把 `insufficient` 直接视为弃权，则系统覆盖率为 61/85（71.76%），覆盖部分
accuracy 为 72.13%。这比推荐配置强制分类的 68.24% 高 3.89 个百分点，但会同时
拒绝 14 条原本正确的 case，选择性收益仍不够强，不能直接设为生产门控。

### 5.4 分层注入无法触及本次 fiber 缺陷

六条 fiber 测试 case 全部位于 `covered` 组，因此分层逻辑仍向它们提供完整 KG
信息。四组都没有产生任何 fiber LLM 预测或最终预测。由此可以排除“未见模式下
注入多数类先验”是当前 fiber recall=0 的主因。

fiber 问题仍指向原缺陷分析中的数据和证据限制：

- fiber 总样本仅 14，训练集仅 8。
- 当前特征无法稳定区分双端设备异常与介质异常。
- `directional_loss`、`bidirectional_loss` 在既有产物中未触发。
- 没有 OTDR、FEC/CRC 时序或邻链共因等独立介质证据。

## 6. 对架构问题的回答

本实验支持更精确的表述：

- 未见组合下，LLM 仍应接收 KG 的**本体定义、原子异常语义、原子路径统计和
  “无匹配”的负信息**。
- 未见组合下，不应把 KG 的**聚合类别分数**包装成 case 特异证据；当没有路径时，
  该分数实质上只是训练集先验。
- 已见组合下，KG 聚合分数仍可能有校准价值，不能一刀切删除。
- 如果系统继续强制三分类，屏蔽先验未必提高 accuracy；分层注入的主要价值是
  支持 `abstain` 和 `request_evidence`，而不是替代缺失的物理证据。

## 7. 后续建议

按优先级建议：

1. **保留两种模式。** `full + legacy` 作为准确率回归基线，
   `layered + llm_only` 作为影子模式输出充分性和补采建议。
2. **把打分改成按覆盖状态条件化。** 仅 `covered` 允许 KG 分数校准 LLM 路，
   `partial/uncovered` 保持独立。
3. **单独评估弃权。** 报告 coverage-accuracy 曲线、错误检出率和每类覆盖率，
   不再只看强制三分类 accuracy。
4. **增强 evidence sufficiency。** 当前只依据结构覆盖状态和 LLM 自报；需要加入
   独立证据数、缺失字段严重度、冲突强度和同源证据识别。
5. **不要靠 prompt 解决 fiber。** 优先补 OTDR、FEC/CRC、逐 lane 时序和邻链共因，
   或将证据不足的 fiber 候选转为主动补采，而不是继续调整类别先验。

## 8. 产物与复现入口

代码快照与实现：

```text
archive/rca_framework_snapshot_20260805_pre_layered_injection/
rca_framework/llm.py
rca_framework/pipeline.py
rca_framework/cli.py
scripts/run_injection_ablation.py
scripts/summarize_injection_ablation.py
tests/test_kg_injection.py
```

实验产物：

```text
artifacts/layered_injection_20260805/baseline_none/
artifacts/layered_injection_20260805/deepseek32b_ablation/
  ablation_comparison.json
  analysis.json
  run_manifest.json
  full__legacy/
  full__llm_only/
  layered__legacy/
  layered__llm_only/
```

运行日志：

```text
artifacts/layered_injection_20260805/ablation_run_final.log
```

正式实验所用命令等价于：

```bash
VLLM_USE_V1=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
NCCL_P2P_DISABLE=1 \
NCCL_SHM_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=6,7 \
PYTHONPATH=/home/chenziang/nsdi \
/home/chenziang/miniconda3/envs/logsy/bin/python \
scripts/run_injection_ablation.py \
  --output-dir artifacts/layered_injection_20260805/deepseek32b_ablation \
  --backend vllm \
  --model-path /home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-new-tokens 512 \
  --dtype bfloat16 \
  --enforce-eager \
  --disable-custom-all-reduce
```
