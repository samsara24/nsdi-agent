# rca_framework 快照：分层 KG 注入改造之后

冻结时间：2026-08-06。

这份副本保存的是执行 `docs/KG_INJECTION_ABLATION_DEEPSEEK32B_REPORT_CN.md` 四组消融实验时
所使用的代码。活动代码树已在同一天回退到改造前状态
（`archive/rca_framework_snapshot_20260805_pre_layered_injection/`），因此本目录是这一版实现的
唯一留存。

## 内容

```text
rca_framework/            分层注入版本的 10 个模块
tests/test_kg_injection.py
scripts/run_injection_ablation.py        四组消融运行器
scripts/summarize_injection_ablation.py  逐 case 汇总分析
run_main_experiment.sh                   当时的一键脚本，与改造前一致
layered_injection.diff                   相对改造前基线的 unified diff，580 行
SHA256SUMS.txt
```

`layered_injection.diff` 只覆盖三个实际被改动的文件：`cli.py`、`llm.py`、`pipeline.py`。
`data.py`、`anomaly.py`、`graph.py`、`rules.py`、`fusion.py`、`types.py` 与改造前完全一致。

## 这一版的行为特征

- `llm.py` 增加 `INJECTION_MODES = ("full", "layered")` 与 `SCORE_MODES = ("legacy", "llm_only")`
  两组正交开关，`full + legacy` 精确复现改造前行为。
- `classify_kg_coverage()` 按 `covered / partial / uncovered` 三档判定 KG 覆盖状态，
  `build_layered_prompt()` 按覆盖状态屏蔽聚合 KG 分数。
- `LAYERED_OUTPUT_SCHEMA` 增加必填 `evidence_sufficiency`，但 `insufficient_confidence_scale`
  固定为 1.0，即只记录充分性，不改变融合权重。
- `pipeline._get_reasoner()` 的缓存键只含模型加载期参数，四组消融共享一次 32B 加载。
- CLI 默认值为 `layered + llm_only`，与实验结论推荐的准确率基线 `full + legacy` 不一致。

## 对应的实验结果

`datasets/organized_rca_v2_stratified_60_40_seed42`，126 训练 / 85 测试，
DeepSeek-R1-Distill-Qwen-32B：

```text
full__legacy        59/85   69.41%
full__llm_only      58/85   68.24%
layered__legacy     58/85   68.24%
layered__llm_only   58/85   68.24%
确定性基线           58/85   68.24%
```

四组 LLM 路自身 accuracy 均为 56/85。fiber recall 四组全为 0。

## 为什么回退

分层注入没有提升强制三分类 accuracy，其真正价值在于产出证据充分性信号，
而这需要 `abstain` 与 `request_evidence` 出口才能兑现。因此活动代码树回到改造前基线，
分层注入的四项可复用设计以 Agent 形态重新引入，方案见
`docs/AGENT_REFACTOR_MODULE_STRATEGY_CN.md` 第 6 节。

## 取回方式

需要重跑消融时，从本目录复制回活动树，而不是在 `rca_framework/` 中并行维护两套实现：

```bash
SNAP=archive/rca_framework_snapshot_20260806_layered_injection
(cd $SNAP && sha256sum -c SHA256SUMS.txt)      # 先校验快照完整性
cp $SNAP/rca_framework/*.py rca_framework/
cp $SNAP/tests/*.py tests/
cp $SNAP/scripts/*.py scripts/
```

实验产物保留在 `artifacts/layered_injection_20260805/`，未随代码回退删除。
