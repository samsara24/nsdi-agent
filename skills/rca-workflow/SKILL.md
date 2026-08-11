---
name: rca-workflow
description: RCA v2 N1-N8 主流程、降级策略和实验门禁。
---

# RCA Workflow

1. N1：构造 EvidencePack，结构性剥离标签。
2. N2：抽取可解释 token，连续阈值只从 train split 拟合。
3. N3：证据图检索 Top-N，输出相似度、覆盖率、缺失和冲突证据。
4. N4：用当前数据集 train-LOO 重新标定路由，不沿用旧 70% 阈值。
5. N5：N5a 复用纯历史链，N5b 补采/仲裁，N5c 走约束 + learned SOP。
6. N6：按历史覆盖率、SOP 叶子校准、约束合规、证据完整度和推导缺口决定 final / request_evidence / human_review。
7. N7：生成含根因、证据链、SOP 路径和置信来源的报告。
8. N8：只回灌人工确认结果。
