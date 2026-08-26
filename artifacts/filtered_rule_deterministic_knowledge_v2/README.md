# Filtered-rule 确定性训练知识审计

本目录仅由固定训练 split 生成，不调用 LLM，不读取测试标签，也不执行 N8 回灌。

## 已沉淀资产

- 训练 case：124 条；来源分布 `{'all_data': 88, 'rule1_channel_not_4': 36}`。
- 可解释 token：64 种。
- 证据图：evidence-graph-v1:124:b60df2407a47cbde，包含 124 个历史 case。
- learned SOP：numeric-decision-tree-v1，hash `2e84eb36c2257ea7`。
- 留一法路由：`{'N5a': 9, 'N5b': 36, 'N5c': 79}`。
- 每条训练 case 的特征、数值量测、SOP 路径、留一法路由和 Top-5 历史候选保存在 `case_audit.json`。

## 可复核结论

- 117 个 signature 中 111 个仅有 1 条支持；训练 signature 高纯度不能直接解释为可泛化准确率。
- 混合标签 signature 覆盖 7 条，不能作为 N5a 自动复用模式。
- 数值 learned SOP 训练内命中 79/124；fiber 命中 0/11。该树只能作为统计先验，不能作为 fiber 或端点归因的物理证据。
- SerDes SNR 数值尺度尚未完成量测语义确认；树中相关分位数切分只保留审计用途。
- LLM calibration 与 LLM trace 均为空。正式测试应在加载本知识包后才调用 LLM。

## 文件

- `knowledge_bundle.json`：可重新加载的训练知识包。
- `training_summary.json`：图纯度、SOP、分支与决策阈值。
- `case_audit.json`：124 条逐 case 审计。
- `signature_audit.json`：完整 signature 分组和标签纯度。
- `token_audit.json`：每个可解释 token 的支持数和标签分布。
- `audit_summary.json`：版本、hash 和关键计数。
