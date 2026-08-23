# nsdi-agent 当前交付状态

本文记录活动数据、框架适配、验证结果和正式实验状态。长期约束见 `AGENTS.md`，验收标准见 `Validation.md`。

## 1. 当前目标

项目使用历史证据图、物理约束、专家 SOP 和 LLM 校验完成光链路 RCA。统一根因标签为：

- `L1`：本端根因。
- `L2`：对端根因。
- `fiber`：两端之间的链路介质根因。

活动阶段已经完成数据固定和框架适配。下一步是在 GPU 实验机执行正式全流程，分别验收两个测试集。

## 2. 活动数据

固定数据位于 `datasets/filtered_rule_temporal_2025_06_09_v1/`，共 608 条。

| 划分 | 来源 | L1 | L2 | fiber | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | 两个来源合并 | 50 | 63 | 11 | 124 |
| test | `all_data` | 144 | 258 | 15 | 417 |
| test | `rule1_channel_not_4` | 37 | 29 | 1 | 67 |

训练月份固定为 2025-06 至 2025-09，其余月份进入测试。两个测试来源独立评估。

来源标签统一规则：

- `all_data`: `l1 -> L1`, `l2 -> L2`
- `rule1_channel_not_4`: `l3 -> L1`, `l4 -> L2`
- `fiber -> fiber`

Expert label 通过核心遥测精确指纹应用：命中 49 条，修正 27 条，其中训练 6 条、测试 21 条；未命中 case 不做推断式改标。

## 3. 拓扑与 lane 契约

活动拓扑版本为 `filtered-rule-topology-v1`。

| 来源 | 本端 L1 | 对端 L2 | 光学 lane | SerDes lane |
| --- | --- | --- | --- | --- |
| `all_data` | 400G | 200G | 4×4 | 4×4 |
| `rule1_channel_not_4` | 400G | 400G | 8×8 | 通常 4×4，部分缺失 |

数据审计确认 `transmission` 是同编号光学 lane 的跨端差值字段，因此同编号光学 lane 是明确的逻辑配对。当前实现保留：

- `tx_ok_rx_down`、`tx_down`、同 lane 双向触底等状态证据。
- single / partial / all-lanes 影响范围。
- case 内单 lane 相对离群证据。
- 来源、拓扑 ID、每指标实际 lane 宽度和缺测状态。

当前实现不使用绝对 Tx/Rx 差值判定链路损耗，不把 SerDes lane 映射到光学 lane。历史检索优先同拓扑正相似候选；同拓扑没有任何证据重叠时才显式启用跨拓扑兜底。

## 4. 已完成实现

### 4.1 数据与标签

- 数据准备脚本生成固定 split、统一标签、expert label 审计和文件哈希。
- Manifest adapter 同时支持活动字段 `output_file` 与 legacy 字段 `file`。
- 数据 API 支持 `train`、`test/all_data`、`test/rule1_channel_not_4`。
- `EvidencePack` 从每条 case 契约恢复真实来源、拓扑和 lane profile。

### 4.2 特征与证据图

- 活动特征 profile：`filtered_rule_v1`。
- 特征字典：`filtered-rule-feature-dictionary-v1`，当前 hash 为 `f399d9758b670f8d`。
- 新增逻辑同 lane 状态与范围 token，不生成绝对链路损耗 token。
- `GraphCase`、候选和匹配结果保存来源、拓扑及 lane profile。
- Top-K 截断前执行同拓扑优先，避免小 K 丢弃兼容历史。

### 4.3 Prompt 与推理

- L1/L2 在协议中统一为 local/remote endpoint。
- Prompt 输入包含来源拓扑、lane profile、lane 宽度、同 lane 逻辑契约和禁止推断项。
- 活动 Prompt 使用物理约束库与量测契约库，不使用旧数据统计型 measured constraints。
- Prompt 路由按数据契约隔离：legacy N5c 保持 400G/200G 语义和
  `rca-dual-sop-multidim-v14-full-step-ids`，活动数据使用
  `filtered-rule-general-structured-retry-v4`；推理 trace 和 manifest 分别记录实际版本。
- 活动正式流程使用 `filtered-rule-three-channel-v2`：先由训练集冻结可解释特征模型和
  证据图，再分别计算完整 token 的 `S_feature` 和语义前缀图的 `S_graph`，把每条 case
  唯一分到 N5a/N5b/N5c。N5a 要求双相似度均为 1.0，N5b 要求双相似度均不低于 0.70。
  N6 只做受约束推理后的置信度与降级门禁，不再作为推理前第四通道。
- 三个分支使用独立载荷：N5a 注入历史证据链，N5b 注入 shared/missing/conflict 与
  关键缺失证据，N5c 注入完整专家 SOP；每个请求同时携带当前五层物理路径和真实 lane 数值。
- 正式 vLLM 开启 JSON Schema 结构化解码。每条 case 首轮生成一次；仅对 JSON 解析失败
  或物理 checker 未通过的 case 重试，最多 3 轮。三轮后仍失败时保留最后一个可解析候选
  并把物理合规分降为 0；完全不可解析时进入低置信 forced/fallback。
- 活动 Prompt 使用一套通用推理协议，不强制固定步骤数。`sop_step_id` 与
  `cited_predicates` 只在输入确实提供相应内容时引用，不再混用 S1-S5 与 Q0/P/R/L/D。
- N8 自动回灌保持关闭；测试标签只在推理完成后参与指标计算。

### 4.4 正式实验入口

`scripts/run_filtered_rule_temporal_experiment.py` 从 124 条 train 以确定性代码构建知识包，
训练阶段不调用 LLM；知识包落盘并重新加载后，GPU vLLM 只处理两个测试集。每个测试集
独立输出 summary、outcomes、traces 和逐 case HTML。

`scripts/build_filtered_rule_deterministic_knowledge.py` 是本地训练知识构建与逐 case 审计入口。
固定产物位于 `artifacts/filtered_rule_deterministic_knowledge_v1/`，包含知识包、124 条
逐 case 特征/数值/SOP/留一法历史候选、signature 分组和 token 支持统计。

`scripts/run_filtered_rule_temporal_gpu_experiment.sh` 不执行 CPU 模型 dry run；它检测空闲显存和模型结构，在 1–4 张 GPU 中选择最大的合法 tensor parallel size，并保存运行前后 GPU 快照、命令和日志。

`scripts/run_synced_filtered_rule_experiment.sh` 要求干净工作树，固定切换本地 `main`，从 `origin/main` 执行 `pull --ff-only`，成功后只提交本轮结果目录并推送远端 `main`。

## 5. 已验证结果

数据完整性检查：

```text
python3 scripts/prepare_filtered_rule_temporal_split.py --check
ok=true, case_count=608, errors=[]
```

针对 manifest、标签映射、拓扑、EvidencePack、同拓扑 Top-K 优先和跨拓扑兜底的 10 个无 fixture 断言均通过。新增 Python 文件通过 `py_compile`，两个 shell 入口通过 `bash -n`。

活动训练核心构建检查：

- 124 条训练 case 可在约 5 秒内构建活动特征字典、证据图和 learned SOP，LLM 调用数为 0。
- 图中来源分布为 `all_data=88`、`rule1_channel_not_4=36`。
- 证据图版本为 `evidence-graph-v1:124:affc399cf8706073`，包含 124 个历史 case 和
  124 条训练确认诊断子图；可跨机器复现的知识包 hash 为 `23a39fe3ced1910e`。
- 118 个 signature 中 117 个标签纯净；只有 1 个混合标签 signature，覆盖 2 条 case。
  其中 112 个 signature 只有单条支持，因此训练纯度不能直接解释为泛化能力。
- 数值 learned SOP hash 为 `2e84eb36c2257ea7`；训练内命中 79/124，fiber 为 0/11。
  SerDes SNR 数值尺度仍未完成量测语义确认，该树只作为统计先验和审计路径。
- 417 条 `all_data` 核心检索中 415 条 Top-1 来自同来源，2 条显式跨拓扑兜底。
- 67 条 `rule1_channel_not_4` 核心检索全部 Top-1 来自同来源，无跨拓扑兜底。
- 逻辑同 lane token 在两个测试集中分别覆盖 157 条和 15 条。

无 LLM 的 train/test 分布审计确认：

- `all_data` 测试双相似度精确匹配 12/417，S_feature/S_graph 中位数为 0.671/0.703，
  最近历史标签直接复用准确率为 52.76%，有 11 种同来源训练未见 token。
- `rule1_channel_not_4` 测试精确匹配 1/67，S_feature/S_graph 中位数为 0.729/0.747，
  最近历史标签直接复用准确率为 49.25%，有 18 种同来源训练未见 token。
- `rule1_channel_not_4` 存在显著时间 schema 漂移：同来源训练仅 6/36 的两端 SerDes
  缺失，测试为 67/67 缺失。缺测只降低完整度，不参与根因投票。
- 两个来源均存在标签先验漂移；统一训练池共享物理知识，但历史标签、阈值和正式指标
  必须按来源分层解释。

本机项目虚拟环境已安装 pytest 9.1.1。完整回归结果为：

```text
.venv/bin/python -m pytest -q
355 passed in 19.97s
```

回归同时锁定 legacy 证据图 hash `5e10b5b25d559777`、legacy Prompt v14、活动
local/remote Prompt 独立版本与 topology-aware hash。正式实验机仍需在拉取最新 `main`
后按同步入口再次执行门禁。

活动数据三通道静态路由分布：

| split | N5a | N5b | N5c | 推理前 N6 | LLM 请求数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train LOO | 10 | 42 | 72 | 0 | 0 |
| test/all_data | 12 | 146 | 259 | 0 | 417 |
| test/rule1_channel_not_4 | 1 | 36 | 30 | 0 | 67 |

首轮生成请求固定为两个测试集共 484 条。后续批次只包含前一轮解析或校验失败的 case；已经通过的
case 不重复生成。每条 case 的 `attempt_count` 必须位于 1–3。

## 6. 正式配置

默认模型为 `/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B`。

- routing policy：`filtered-rule-three-channel-v2`
- M9 candidate order：仅 `branch`
- Top-K：全量候选
- N8：冻结
- seed：42
- dtype：BF16
- max model length：32768
- max new tokens：16384
- max attempts：3（仅失败 case 重试）
- structured output：JSON Schema guided decoding
- tensor parallel：根据空闲 GPU 和模型结构自动选择，最多 4

正式同步运行：

```bash
scripts/run_synced_filtered_rule_experiment.sh
```

跳过 Git 同步直接运行：

```bash
scripts/run_filtered_rule_temporal_gpu_experiment.sh
```

## 7. 待完成工作

1. 在 GPU 实验机通过同步入口执行正式实验。
2. 审核两个独立 HTML 报告、fiber 个案、降级比例和跨拓扑兜底 case。
3. 根据训练内标定和正式 bad case 归因开展消融，不修改同轮测试知识。
4. 正式结果稳定后归档不再使用的旧实验说明和重复文档；legacy 代码与基线 artifact 暂时保留。

旧 organized、l2fixed 和 expanded 结果只作为历史参考，不与活动数据指标混表。
