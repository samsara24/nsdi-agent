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
  `filtered-rule-three-channel-single-pass-v2`；推理 trace 和 manifest 分别记录实际版本。
- 活动正式流程使用 `filtered-rule-three-channel-v1`：先由训练集冻结可解释特征模型和
  证据图，再按 IDF-Jaccard 相似度与特征覆盖率把每条 case 唯一分到 N5a/N5b/N5c。
  N6 只做单次推理后的置信度与降级门禁，不再作为推理前第四通道。
- 每条训练或测试 case 固定只生成一次，失败输出直接进入低置信 forced/fallback，
  不再向 GPU 发起重写请求；运行时逐 trace 强制校验 `attempt_count == 1`。
- N8 自动回灌保持关闭；测试标签只在推理完成后参与指标计算。

### 4.4 正式实验入口

`scripts/run_filtered_rule_temporal_experiment.py` 仅使用 GPU vLLM，从 124 条 train 构建一次知识包并落盘，重新加载后依次运行两个测试集。每个测试集独立输出 summary、outcomes、traces 和逐 case HTML。

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

- 124 条训练 case 可构建活动特征字典和证据图。
- 图中来源分布为 `all_data=88`、`rule1_channel_not_4=36`。
- 417 条 `all_data` 核心检索中 415 条 Top-1 来自同来源，2 条显式跨拓扑兜底。
- 67 条 `rule1_channel_not_4` 核心检索全部 Top-1 来自同来源，无跨拓扑兜底。
- 逻辑同 lane token 在两个测试集中分别覆盖 157 条和 15 条。

本机项目虚拟环境已安装 pytest 9.1.1。完整回归结果为：

```text
.venv/bin/python -m pytest -q
350 passed in 16.31s
```

回归同时锁定 legacy 证据图 hash `5e10b5b25d559777`、legacy Prompt v14、活动
local/remote Prompt 独立版本与 topology-aware hash。正式实验机仍需在拉取最新 `main`
后按同步入口再次执行门禁。

活动数据三通道静态路由分布：

| split | N5a | N5b | N5c | 推理前 N6 | LLM 请求数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train LOO | 10 | 25 | 89 | 0 | 124 |
| test/all_data | 12 | 110 | 295 | 0 | 417 |
| test/rule1_channel_not_4 | 1 | 18 | 48 | 0 | 67 |

总生成请求固定为 608 条，每条 case 一次；不再出现 `124→103→86` 或
`417→361→330` 形式的多轮重写批次。

## 6. 正式配置

默认模型为 `/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B`。

- routing policy：`filtered-rule-three-channel-v1`
- M9 candidate order：仅 `branch`
- Top-K：全量候选
- N8：冻结
- seed：42
- dtype：BF16
- max model length：32768
- max new tokens：16384
- max attempts：1（单次生成，无重写）
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
