# rca_framework 快照：分层 KG 注入改造之前

冻结时间：2026-08-05。

这份副本保留 `docs/DEFECT_ANALYSIS_CN.md` 分析所针对的那一版实现，用途只有一个：
在评估分层 KG 注入改造时，可以随时回到改造前的确切代码。

冻结的行为特征：

- `llm.py::build_path_prompt` 对所有 case 无条件全量注入 KG 的 `candidate_path_scores`、
  `root_cause_paths`、`matched_kg_feature_rules` 和 `retrieved_training_cases`，
  不区分该 case 是否落在 KG 已覆盖的模式内。
- `llm.py::_parse_or_fallback` 把 KG 分数以固定 0.35 权重回灌进 LLM 结果，
  随后 `fusion.py::fuse_results` 又以 0.55 权重把该结果当作独立一路融合，
  KG 的类别先验因此被计入两次。
- 输出 schema 强制 `prediction` 三选一，没有任何表达证据不足的字段。

对应的历史基线（`datasets/organized_rca_v2_stratified_60_40_seed42`，126 训练 / 85 测试）：

- 确定性基线 58/85，accuracy 68.24%，fiber recall 0。
- DeepSeek-R1-Distill-Qwen-32B 59/85，accuracy 69.41%，fiber recall 0。

改造后的代码通过 `--kg-injection full --llm-score-mode legacy` 复现这一版的 LLM 行为，
不需要切回本副本即可做对照；本副本仅作为最终的回退保障。
