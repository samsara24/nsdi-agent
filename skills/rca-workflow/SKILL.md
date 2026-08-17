---
name: rca-workflow
description: RCA v2 N1-N8 主流程、降级策略和实验门禁。
---

# RCA Workflow

1. N1：构造 EvidencePack，结构性剥离标签。
2. N2：抽取可解释 token，连续阈值只从 train split 拟合。
3. N3：证据图检索 Top-N，输出相似度、覆盖率、缺失和冲突证据。
4. N4：用当前数据集 train-LOO 重新标定路由，不沿用旧 70% 阈值。
5. N5：N5a 复用纯历史链，N5b 用物理约束判关键证据并仲裁，N5c 走专家 SOP + 约束 LLM。
6. N6：正式默认只接受 branch 候选；expert / learned SOP 只能显式消融或作报告字段。
7. N7：生成含主流程图、调整点、根因、证据链、SOP 路径、置信来源和逐 bad case 分析的报告。
8. N8：本阶段冻结；只保留人工确认回灌语义，不用测试 bad case 自动更新知识。

## Loop 实验门禁

- 每轮实验必须先说明遵循 `docs/个人整体思路.md`，并从测试 bad case 出发提出假设。
- 允许的核心调整只有：证据图约束 / schema、阈值或路由、大模型 prompt、代码 bug fix。
- 每轮实验必须归档到 `experiments/<YYYYMMDD>_<short-name>/`，并生成 `report.html`。
- `report.html` 必须展示当前主流程图，并标注本轮调整了哪里。
- 报告必须记录当前证据图、物理约束、SOP 版本、prompt 版本、阈值和 M9 candidate order。
- 正确 case 按分支和做对的步骤归纳；bad case 必须逐条分析失败步骤、错因和下一步动作。
- 疑似标签问题写入 `label_suspects.json`；当前不可安全提升的 case 写入 `irreducible_cases.json`，后续保留但不继续围绕它刷指标。
