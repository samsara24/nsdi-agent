#!/usr/bin/env python3
"""Audit filtered-rule explainable features and render a standalone HTML review."""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.evidence_pack import EvidencePack  # noqa: E402
from rca_framework.features.dictionary import dictionary_for  # noqa: E402
from rca_framework.features.extractor import extract_features  # noqa: E402
from rca_framework.knowledge import OfflineKnowledgeBundle  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/filtered_rule_temporal_2025_06_09_v1"
KNOWLEDGE = ROOT / "artifacts/filtered_rule_deterministic_knowledge_v2/knowledge_bundle.json"
OUTPUT = ROOT / "artifacts/filtered_rule_feature_review_v2"
LABELS = ("L1", "L2", "fiber")

PLAIN = {
    "signal_drop": "看某一端有几条 lane 已经掉到底：一条、部分还是全部。它不关心具体是 lane 0 还是 lane 7。",
    "status_fault": "看模块有没有报 LOS/LOL 等硬状态故障，只记录哪一端、哪种状态，不记录 lane 编号。",
    "lane_imbalance": "看同一端各 lane 是否明显不齐。例如八条中有一条特别差，但不记住差的是第几条。",
    "paired_lane_state": "把同号发送和接收当作一对，判断发得出但收不到、发送本身已掉或双向同 lane 异常；最后只保留异常类型和影响范围。",
    "level_tail": "把每端整体光功率/SNR分成偏低、正常、偏高三档，只输出偏低或偏高，不保留原始数值和 lane 号。",
    "telemetry_gap": "区分数据采全、只采到一部分、完全没采到，避免把“没数据”误当成“都正常”。",
    "serdes_state": "只判断每端 SerDes 指标是有效、无效还是缺失，不拿数值直接判断根因。",
    "signal_drop_ratio": "看异常 lane 占整端口的比例，而不是死记异常了第几条或一共几条；4 lane 和 8 lane 因此能使用同一种范围表达。",
    "topology_level_tail": "只和相同端口结构、相同 lane 数的历史数据比较整体高低，避免八条 lane 天然比四条更容易出现一个极小值。",
}

VERDICTS = {
    "signal_drop": ("REVISE", "物理语义合理且置换不变，但 host_snr/media_snr/serdes_snr 共用 DOWN_THRESHOLDS 的工程含义并不等价；稀有 token 很多。应按指标契约拆分并合并低支持档位。"),
    "status_fault": ("KEEP", "硬状态语义清楚、维度少、与 lane 编号无关。应继续作为强事件证据，但不能单独完成端点根因终裁。"),
    "lane_imbalance": ("REVISE", "max-min 天然置换不变，但 4-lane 与 8-lane 的极差分布不可直接比较，且对单个尖峰敏感。建议改为稳健离群 lane 比例与归一化离散度。"),
    "paired_lane_state": ("KEEP_WITH_GUARD", "同步重编号下严格不变，并保留有价值的跨端方向细节；但它只对数据契约确认的同号光学 lane 有效，不能对两端独立乱序，也不能映射 SerDes lane。"),
    "level_tail": ("REVISE", "聚合后置换不变，但 media_snr_min 随 lane 数增加更容易变低，4/8 lane 存在宽度偏差；训练分位数还可能混合不同拓扑。应按来源/宽度标定并用低分位代替 min。"),
    "telemetry_gap": ("MOVE_TO_QUALITY", "置换不变，但它描述采集质量而非根因。放进相似度 signature 会让相同缺测模式产生虚假相似，应移到 N6 数据质量门禁。"),
    "serdes_state": ("MOVE_TO_QUALITY", "置换不变，但 valid 在绝大多数 case 恒定出现，主要增加公共 token；missing/invalid 应作为量测契约和降级条件，不作为根因相似度票。"),
    "signal_drop_ratio": ("KEEP", "使用异常 lane 比例分档，不绑定 lane 身份，并统一 4/8 lane 的影响范围语义。仍需持续审核每个指标的 DOWN_THRESHOLDS 量测契约。"),
    "topology_level_tail": ("KEEP_WITH_GUARD", "按 topology、side、statistic 和实际 width 独立冻结边界，media_snr 使用低四分位而不是 min。小支持分组不产出 token。"),
}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def permute_lane_keys(value: Any) -> Any:
    """Reverse numeric lane keys at every lane-valued leaf, preserving all pairings."""
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    keys = list(value)
    if keys and all(str(key).isdigit() for key in keys):
        ordered = sorted(keys, key=lambda key: int(str(key)))
        reversed_keys = list(reversed(ordered))
        mapping = {old: new for old, new in zip(ordered, reversed_keys)}
        return {mapping[key]: copy.deepcopy(value[key]) for key in ordered}
    return {key: permute_lane_keys(item) for key, item in value.items()}


def entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n) if total else 0.0


def association(counts: Mapping[str, int], baseline: Mapping[str, float]) -> Dict[str, Any]:
    total = sum(counts.values())
    rates = {label: counts.get(label, 0) / total if total else 0.0 for label in LABELS}
    lifts = {label: rates[label] / baseline[label] if baseline[label] else 0.0 for label in LABELS}
    dominant = max(LABELS, key=lambda label: (lifts[label], rates[label])) if total else "none"
    return {"rates": rates, "lifts": lifts, "dominant_label_by_lift": dominant, "entropy": entropy(counts)}


def analyze(data_dir: Path, bundle: OfflineKnowledgeBundle, feature_profile: str) -> Dict[str, Any]:
    dictionary = dictionary_for(feature_profile)
    splits = ("train", "test/all_data", "test/rule1_channel_not_4")
    cases: list[Dict[str, Any]] = []
    split_names: list[str] = []
    features = []
    for split in splits:
        rows = cases_by_manifest_split(data_dir, split)
        _, extracted = bundle.extract_test_features(rows)
        cases.extend(rows); split_names.extend([split] * len(rows)); features.extend(extracted)

    label_total = Counter(case["label"] for case in cases)
    baseline = {label: label_total[label] / len(cases) for label in LABELS}
    family_cases: Dict[str, list[int]] = defaultdict(list)
    token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    token_splits: Dict[str, Counter[str]] = defaultdict(Counter)
    token_families: Dict[str, str] = {}
    for index, (case, split, feature) in enumerate(zip(cases, split_names, features)):
        for family, tokens in feature.by_family.items():
            family_cases[family].append(index)
            for token in tokens:
                token_counts[token][case["label"]] += 1
                token_splits[token][split] += 1
                token_families[token] = family

    permutation_changed = []
    family_changed = Counter()
    for case, feature in zip(cases, features):
        changed_case = copy.deepcopy(case)
        changed_case.pop("label", None)
        changed_case = permute_lane_keys(changed_case)
        changed_case["case_id"] = case["case_id"]
        pack = EvidencePack.from_case(changed_case)
        altered = extract_features(pack, bundle.thresholds, bundle.feature_model, dictionary=dictionary)
        if altered.tokens != feature.tokens:
            changed = sorted(set(feature.by_family) | set(altered.by_family))
            changed = [name for name in changed if feature.by_family.get(name) != altered.by_family.get(name)]
            permutation_changed.append({"case_id": case["case_id"], "families": changed,
                                        "before": list(feature.tokens), "after": list(altered.tokens)})
            family_changed.update(changed)

    families = []
    for family in dictionary.families:
        indices = family_cases.get(family.name, [])
        labels = Counter(cases[i]["label"] for i in indices)
        tokens = []
        for token in sorted(t for t, name in token_families.items() if name == family.name):
            counts = token_counts[token]
            tokens.append({"token": token, "support": sum(counts.values()), "label_distribution": dict(counts),
                           "split_distribution": dict(token_splits[token]), **association(counts, baseline)})
        verdict, issue = VERDICTS[family.name]
        families.append({
            "name": family.name, "dimension": family.dimension, "tier": family.tier, "status": family.status,
            "physical_meaning": family.physical_meaning, "plain_explanation": PLAIN[family.name],
            "unit": family.unit, "value_domain": list(family.value_domain), "extraction_rule": family.extraction_rule,
            "token_template": family.token_template, "sparsity": family.sparsity,
            "case_support": len(indices), "coverage": len(indices) / len(cases), "label_distribution": dict(labels),
            "association": association(labels, baseline), "token_count": len(tokens), "tokens": tokens,
            "permutation_changed_cases": family_changed[family.name], "verdict": verdict, "review_conclusion": issue,
        })

    signatures = Counter(feature.signature for feature in features)
    token_lengths = [len(feature.tokens) for feature in features]
    return {
        "schema_version": "filtered-rule-feature-review-v1", "case_count": len(cases),
        "split_counts": dict(Counter(split_names)), "label_distribution": dict(label_total), "label_baseline": baseline,
        "dictionary_version": dictionary.version, "dictionary_hash": dictionary.content_hash(),
        "feature_profile": feature_profile,
        "family_count": len(families), "token_count": len(token_counts),
        "mean_tokens_per_case": sum(token_lengths) / len(token_lengths), "max_tokens_per_case": max(token_lengths),
        "signature_count": len(signatures), "singleton_signature_count": sum(n == 1 for n in signatures.values()),
        "permutation_test": {"method": "reverse all numeric lane keys consistently within every lane-valued block",
                             "case_count": len(cases), "changed_case_count": len(permutation_changed),
                             "passed": not permutation_changed, "changed_cases": permutation_changed},
        "families": families,
    }


def render(report: Mapping[str, Any]) -> str:
    e = html.escape
    optimized = report.get("feature_profile") == "filtered_rule_v2"
    family_sections = []
    for f in report["families"]:
        rows = "".join(
            f"<tr><td><code>{e(t['token'])}</code></td><td>{t['support']}</td><td>{e(str(t['label_distribution']))}</td>"
            f"<td>{e(t['dominant_label_by_lift'])}</td><td>{max(t['lifts'].values()):.2f}×</td><td>{e(str(t['split_distribution']))}</td></tr>"
            for t in sorted(f["tokens"], key=lambda x: (-x["support"], x["token"]))
        )
        family_sections.append(f"""<section class='card family' id='{e(f['name'])}' data-status='{e(f['verdict'])}'>
<div class='title'><div><h2>{e(f['name'])}</h2><p>{e(f['dimension'])}</p></div><span class='badge {e(f['verdict'].lower())}'>{e(f['verdict'])}</span></div>
<div class='stats'><b>覆盖 {f['case_support']}/{report['case_count']} ({f['coverage']:.1%})</b><span>{f['token_count']} 种 token</span><span>标签 {e(str(f['label_distribution']))}</span><span>置换变化 {f['permutation_changed_cases']}</span></div>
<div class='explain'><article><h3>专业解释</h3><p>{e(f['physical_meaning'])}</p><p><b>抽取：</b>{e(f['extraction_rule'])}</p><p><b>单位：</b>{e(f['unit'])}　<b>取值域：</b>{e(' / '.join(f['value_domain']))}</p></article><article class='plain'><h3>通俗解释</h3><p>{e(f['plain_explanation'])}</p></article></div>
<div class='conclusion'><b>Review 结论：</b>{e(f['review_conclusion'])}</div>
<details><summary>查看全部 {f['token_count']} 个维度及类别关联</summary><p class='hint'>“关联类别”按相对全体标签先验的 lift 最大值计算，只表示统计关联，不表示物理因果；低支持 token 不应直接用于终裁。</p><div class='table'><table><thead><tr><th>特征维度</th><th>支持数</th><th>标签分布</th><th>关联类别</th><th>最大 lift</th><th>split 分布</th></tr></thead><tbody>{rows}</tbody></table></div></details></section>""")
    nav = "".join(f"<a href='#{e(f['name'])}'><span>{e(f['name'])}</span><small>{e(f['verdict'])}</small></a>" for f in report["families"])
    quality_sentence = (
        "`telemetry_gap`、`serdes_state` 已从检索 signature 移到 EvidencePack/N6 数据质量门禁。"
        if optimized else
        "`telemetry_gap`、`serdes_state` 应从检索 signature 移到数据质量门禁。"
    )
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>可解释性特征逐维 Review</title><style>
:root{{--bg:#f4f6fa;--ink:#172033;--muted:#667085;--line:#d9dfeb;--blue:#2457d6;--green:#067647;--amber:#b54708;--red:#b42318}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.65 system-ui,-apple-system,sans-serif}}header{{background:linear-gradient(120deg,#101828,#163a8c);color:white;padding:36px max(24px,calc((100% - 1380px)/2))}}header h1{{margin:0 0 8px;font-size:30px}}header p{{max-width:980px;color:#dce6ff}}.layout{{display:grid;grid-template-columns:250px minmax(0,1fr);max-width:1440px;margin:auto}}nav{{position:sticky;top:0;height:100vh;padding:20px 12px;border-right:1px solid var(--line);background:#fff}}nav a{{display:flex;justify-content:space-between;text-decoration:none;color:var(--ink);padding:9px;border-radius:7px}}nav a:hover{{background:#eef3ff}}nav small{{color:var(--muted)}}main{{padding:24px;max-width:1180px}}.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:18px;box-shadow:0 1px 3px #1018280b}}.hero-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.metric{{background:#f7f9fc;border-radius:8px;padding:13px}}.metric b{{display:block;font-size:22px;color:var(--blue)}}.title,.stats{{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}}.title h2{{margin:0;font-size:24px}}.title p{{margin:0;color:var(--muted)}}.badge{{padding:5px 9px;border-radius:999px;font-weight:700;background:#eee}}.keep{{background:#dcfae6;color:var(--green)}}.keep_with_guard,.revise{{background:#fef0c7;color:var(--amber)}}.move_to_quality{{background:#fee4e2;color:var(--red)}}.stats{{justify-content:flex-start;background:#f7f9fc;padding:9px;margin:14px 0}}.stats span,.stats b{{margin-right:14px}}.explain{{display:grid;grid-template-columns:1.45fr 1fr;gap:14px}}article{{border:1px solid var(--line);border-radius:9px;padding:14px}}article h3{{margin-top:0}}.plain{{background:#f0f5ff;border-color:#b9ccff}}.conclusion{{margin:14px 0;padding:12px;border-left:4px solid var(--amber);background:#fffaeb}}summary{{cursor:pointer;font-weight:700}}.table{{overflow:auto;margin-top:10px}}table{{width:100%;border-collapse:collapse;min-width:800px}}th,td{{padding:7px;border:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#f2f4f7}}code{{font-size:12px}}.hint{{color:var(--muted)}}.decision li{{margin:7px 0}}@media(max-width:850px){{.layout{{display:block}}nav{{position:static;height:auto}}.hero-grid,.explain{{grid-template-columns:1fr 1fr}}}}@media(max-width:600px){{.hero-grid,.explain{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Filtered-rule 可解释性特征逐维 Review</h1><p>审计对象：{e(report['dictionary_version'])}。覆盖固定新数据集全部 {report['case_count']} 条 case。重点验证 lane 交换不变性、特征数量、跨 4/8 lane 可比性、类别关联和物理解释有效性。</p></header><div class='layout'><nav><b>{report['family_count']} 个活动特征族</b>{nav}</nav><main>
<section class='card'><h2>整体结论</h2><div class='hero-grid'><div class='metric'><b>{report['family_count']}</b>活动特征族</div><div class='metric'><b>{report['token_count']}</b>实际 token 维度</div><div class='metric'><b>{report['mean_tokens_per_case']:.2f}</b>平均 token/case</div><div class='metric'><b>{report['signature_count']}</b>不同 signature</div></div><ul class='decision'><li><b>同步 lane 置换测试：{'PASS' if report['permutation_test']['passed'] else 'FAIL'}。</b> 对全部 {report['case_count']} 条 case 统一反转数值 lane key，变化 {report['permutation_test']['changed_case_count']} 条。当前 token 不绑定 lane 0/1/2… 的具体编号。</li><li><b>“置换不变”不等于“4 lane 与 8 lane 可直接同分布比较”。</b> v2 使用异常 lane 比例，并按 topology 和实际 lane width 标定连续量；media_snr 使用低四分位替代最小值。</li><li><b>质量上下文与根因 signature 分离。</b> {report['token_count']} 种 token、{report['signature_count']} 个 signature，单例 signature {report['singleton_signature_count']} 个。{e(quality_sentence)}</li><li><b>保留方向细节但不要保留 lane 身份。</b> `paired_lane_state` 的异常类型和 single/partial/all scope 合理；只有对两端同步重编号才不变，因为同号 lane 配对是数据契约，独立打乱任一端会破坏真实对应关系。</li></ul></section>
<section class='card'><h2>推荐改造后的三层特征</h2><ol><li><b>根因证据层：</b>status_fault、受契约保护的 paired_lane_state，以及按指标重新定义的 signal_drop。</li><li><b>稳健统计层：</b>lane_imbalance 改为异常 lane 比例 + MAD/IQR；level_tail 按 source/topology/lane-width 标定，media_snr_min 改为低分位。</li><li><b>质量与上下文层：</b>telemetry_gap、serdes_state、lane width、告警类型只进入 N6/prompt，不参与主相似度或降低权重。</li></ol><p><b>标签关联解释：</b>下面每个 token 的“关联类别”只根据当前标签分布相对总体先验计算。它用于发现候选判别力和标签冲突，不能被写成物理规则。</p></section>{''.join(family_sections)}</main></div></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--knowledge", type=Path, default=KNOWLEDGE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--feature-profile", default="filtered_rule_v2", choices=("filtered_rule_v1", "filtered_rule_v2"))
    args = parser.parse_args()
    report = analyze(args.data_dir, OfflineKnowledgeBundle.load(args.knowledge), args.feature_profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump(args.output_dir / "feature_review.json", report)
    (args.output_dir / "feature_review.html").write_text(render(report), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Filtered-rule 可解释性特征 Review\n\n"
        f"本报告审计活动 `{args.feature_profile}` 特征，不修改数据标签。\n\n"
        f"- case: {report['case_count']}\n- families: {report['family_count']}\n- tokens: {report['token_count']}\n"
        f"- permutation invariant: {report['permutation_test']['passed']}\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("case_count", "family_count", "token_count", "mean_tokens_per_case", "signature_count", "singleton_signature_count", "permutation_test") if key != "permutation_test"} | {"permutation_test": {k: report["permutation_test"][k] for k in ("case_count", "changed_case_count", "passed")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
