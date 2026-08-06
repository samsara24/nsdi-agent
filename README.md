# L1/L2 光链路 RCA v2

本仓库是 `nsdi/` 的 Skill 化 / Agent 化改造树；开发前先阅读 [AGENTS.md](AGENTS.md) 与 [Progress.md](Progress.md)，确认项目边界、冻结基线与当前重构进度。

项目已切换到新的双路 RCA 框架，正式三分类标签为：

- `L1`：400G 端口或其设备侧根因
- `L2`：200G 端口或其设备侧根因
- `fiber`：L1 与 L2 之间的光纤/链路介质根因

新实现位于 `rca_framework/`。旧脚本、旧报告、`saved_methods/` 和 `outputs/` 已统一迁入 `before/`，作为历史探索保留，不再构成 v2 方法的一部分。

## 当前产物

```text
data/                                      # 366 条原始数据，保持不变
before/                                    # 旧探索代码、报告、outputs 和 saved_methods
archive/legacy_exploration/                # 原始数据 SHA-256 清单与旧探索说明
datasets/rca_v2/                           # 268 条脱敏、L1/L2 归一化数据
rca_framework/                             # 新方法实现
artifacts/rca_v2_baseline/                 # 前 200 训练、后 68 验证的可加载模型与结果
docs/RCA_V2_ARCHITECTURE.md                # 架构、路径、规则和冲突策略
docs/rca_v2_code_report/report.html        # 可独立打开的代码详解与实现审计报告
tests/                                     # 框架不变量测试
```

原始 366 条中，212 条为 local=200G/remote=400G，56 条为 local=400G/remote=200G，均进入 v2；98 条 400G–400G case 无法满足 L1=400G、L2=200G 定义，保留在原始数据及校验清单中，但不会被强行改写后混入训练。

## 方法概览

第一路 `KG + RAG + LLM` 以 L1/L2/fiber 根因为中心，只让异常行为形成图边。新 case 被投影为“case→异常名词→根因”路径，并检索训练集相似异常 case；LLM 只接收这些路径与检索证据。KG 同时从训练集按类别自动生成 `feature_profiles` 和 `feature_rules`：单特征要求类内覆盖率、全局精确率和 lift 同时达标；少数类（尤其 fiber）优先使用满足支持度和精确率的双特征组合，避免把少数类的普通异常误当成特有规则。推理时这些规则参与 KG 分数，并作为可审计证据输出。未启用大模型时，框架使用可审计的确定性路径推理。

第二路 `KG + RCA` 从同一异常语义层独立学习三套符号规则。单异常和异常组合按置信度、lift、支持度及排他 margin 唯一归属某个根因，保证 L1/L2/fiber 规则前件不重合。

两路一致时合并路径和规则补全解释；不一致时按校准置信度和加权证据决策，分差不足则给出暂定三分类结果并标记人工复核。

## 快速运行

```bash
# 1. 从不覆盖原始 data/，生成新的脱敏数据。
python -m rca_framework.cli prepare \
  --input-dir data \
  --output-dir datasets/rca_v2 \
  --archive-manifest archive/legacy_exploration/source_data_manifest.json

# 2. 训练并进行无测试标签泄漏的固定切分评估。
python -m rca_framework.cli train-evaluate \
  --data-dir datasets/rca_v2 \
  --train-size 200 \
  --output-dir artifacts/rca_v2_baseline \
  --backend none

# 3. 用冻结模型推理一个新增的 schema-v2 case。
python -m rca_framework.cli infer \
  --model artifacts/rca_v2_baseline/model \
  --case datasets/rca_v2/case_000268.json \
  --output artifacts/single_case_result.json

# 生成的 KG 规则位于模型文件中，可直接检查每类特征和规则。
jq '.feature_rules' artifacts/rca_v2_feature_kg_baseline_v3/model/knowledge_graph.json

# 4. 验证关键不变量。
pytest -q
```

启用真实 LLM 时，将 `--backend` 改为 `vllm` 或 `transformers`，同时传入 `--model-path`。中文框架说明见 [RCA_V2_FRAMEWORK_CN.md](docs/RCA_V2_FRAMEWORK_CN.md)，详细设计与冲突策略见 [RCA_V2_ARCHITECTURE.md](docs/RCA_V2_ARCHITECTURE.md)。

面向新同学的逐文件代码说明、完整运行命令、常见问题和交接讲解提纲见
[PROJECT_HANDOVER_CN.md](docs/PROJECT_HANDOVER_CN.md)。

纯 PCIe 多卡机器如果在 vLLM custom all-reduce 初始化阶段停滞，可增加
`--disable-custom-all-reduce`，并按机器环境设置 NCCL 通信参数。DeepSeek-R1-Distill-Qwen-32B
的两卡实测命令与完整结果见
[RCA_V2_DEEPSEEK32B_TEST_REPORT.md](docs/RCA_V2_DEEPSEEK32B_TEST_REPORT.md)。

## 基线说明

`artifacts/rca_v2_baseline/` 是框架连通性基线，LLM 后端为 `none`，并非最终模型效果：后 68 条 accuracy 为 38/68（55.88%），fiber recall 为 0。该结果明确暴露了少数类与分布偏移问题，后续应在冻结框架后单独进行训练划分、阈值、类平衡和真实 LLM 的实验设计，不能通过读取测试标签调参。

旧方法的 63.24% 等结果仍保留在 [before/](before/) 中，仅作为探索对照，不与 v2 基线混称。

## 真实 LLM 实测

2026-07-19 使用本地 `DeepSeek-R1-Distill-Qwen-32B`、两张 RTX A6000 和 vLLM
完成了后 68 条的真实 LLM 评估。68/68 均为有效 `llm_path_reasoning` 输出，无 fallback；
最终 accuracy 为 37/68（54.41%），fiber recall 仍为 0。该实验确认了 LLM 工程链路已跑通，
但没有证明模型效果优于确定性基线，因此不能据此把当前方法描述为效果达标。
