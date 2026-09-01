# nsdi-agent 光链路证据图 RCA

`nsdi-agent` 是一套面向光链路故障的可审计根因分析框架。系统从告警、两端光模块遥测和 lane 级测量构建证据包，通过历史证据图匹配、物理约束、专家 SOP 和 LLM 校验，输出根因、证据链、置信度以及补采或人工复核建议。

正式标签空间为：

- `L1`：本端根因。
- `L2`：对端根因。
- `fiber`：两端之间的光纤或链路介质根因。

L1/L2 表示本端/对端；速率和 lane 数由每条 case 的来源拓扑契约单独描述。

## 项目文档

- [AGENTS.md](AGENTS.md)：项目章程、活动数据契约、架构和开发规范。
- [Progress.md](Progress.md)：当前交付状态、已验证事实和实施路线。
- [Validation.md](Validation.md)：正式实验验收门禁。
- [个人整体思路](docs/个人整体思路.md)：证据图 RCA 主链路设计依据。

## 活动数据

活动数据集位于：

`datasets/filtered_rule_temporal_2025_06_09_v1/`

| 划分 | 来源 | L1 | L2 | fiber | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | 两个来源合并 | 50 | 63 | 11 | 124 |
| test | `all_data` | 144 | 258 | 15 | 417 |
| test | `rule1_channel_not_4` | 37 | 29 | 1 | 67 |

训练集使用 2025-06 至 2025-09 的 case。其余月份作为测试，两个来源分别评估。

来源标签统一为：

- `all_data`: `l1/l2 -> L1/L2`
- `rule1_channel_not_4`: `l3/l4 -> L1/L2`
- `fiber -> fiber`

每条 case 保留来源、原始标签、源文件哈希、lane 宽度和字段缺失状态。

两类拓扑分别为：`all_data` 的本端 400G / 对端 200G、4×4 逻辑光学 lane；
`rule1_channel_not_4` 的两端均为 400G、8×8 逻辑光学 lane。跨端同编号光学 lane
是数据字段定义中的逻辑配对，可用于状态和相对异常证据；不用于计算绝对链路损耗，
也不建立 SerDes lane 到光学 lane 的强制映射。

## 方法概览

```mermaid
flowchart LR
    input["告警与遥测"] --> pack["EvidencePack"]
    pack --> feature["可解释特征"]
    feature --> graph["历史证据图 Top-N"]
    graph --> route{"相似度与证据路由"}
    route --> exact["完全匹配"]
    route --> partial["部分匹配"]
    route --> general["低匹配"]
    constraints["物理约束"] --> partial
    constraints --> general
    sop["专家 SOP"] --> general
    exact --> decision["置信度与降级"]
    partial --> decision
    general --> decision
    decision --> result["根因与证据链"]
    decision --> review["补采 / 人工复核"]
```

核心原则：

- 证据图历史匹配是主干。
- 测试标签在推理边界结构性隔离。
- 物理约束和训练统计分层管理。
- 缺测、拓扑和 lane 数差异显式建模。
- 低置信度 case 允许降级，不强制三分类。
- 测试结果不自动回灌训练知识。

## 目录结构

```text
rca_framework/                 # RCA 主框架
  evidence_pack.py             # 标签隔离与标准证据包
  features/                    # 可解释特征字典与抽取
  evidence_graph/              # 历史图、Top-N 匹配与路由
  constraints/                 # 物理约束与可执行检查
  branches/                    # 完全/部分/低匹配分支
  llm/                         # Prompt、协议与推理后端
scripts/                       # 数据准备与实验入口
datasets/                      # 版本化数据契约
tests/                         # 单元与回归测试
experiments/                   # 正式实验归档
```

## 当前运行入口

检查活动数据完整性：

```bash
python3 scripts/prepare_filtered_rule_temporal_split.py --check
```

运行代码回归：

```bash
python -m pytest -q
```

直接运行正式 GPU 实验（自动选择最多 4 张合法空闲 GPU）：

```bash
scripts/run_filtered_rule_temporal_gpu_experiment.sh
```

实验机从 `main` 同步、运行、提交结果并推回 `main`：

```bash
scripts/run_synced_filtered_rule_experiment.sh
```

正式入口不执行 CPU 模型 dry run。两个测试集使用同一个持久化只读知识包，分别生成指标、逐 case 结果和 HTML 报告。

## 参考资产

`organized_data`、`datasets/rca_v2_l2fixed` 及相关 artifacts 用于 legacy 回归和方法参考，不属于活动数据实验。不同数据契约的指标、阈值、IDF、SOP 和证据图不得混用。
