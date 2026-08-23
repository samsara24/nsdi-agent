#!/usr/bin/env python3
"""Run label-blind cold-start and train-knowledge RCA reviews, then evaluate."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.decision_tree import numeric_features_from_packs  # noqa: E402
from rca_framework.evidence_graph import FILTERED_RULE_THREE_CHANNEL_POLICY, match_many, route_many  # noqa: E402
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.expert import diagnose_many  # noqa: E402
from rca_framework.expanded_evidence import physical_evidence_paths  # noqa: E402
from rca_framework.knowledge import OfflineKnowledgeBundle  # noqa: E402
from rca_framework.types import ROOT_CAUSES  # noqa: E402


SPLITS = {"test_all_data": "test/all_data", "test_rule1_channel_not_4": "test/rule1_channel_not_4"}
POLICY_VERSION = "codex-blind-physical-plus-train-knowledge-v1"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def _cold_prediction(pack: Any, diagnosis: Any) -> Dict[str, Any]:
    paths = physical_evidence_paths(pack.telemetry)
    verdict = diagnosis.verdict or "L2"
    direct = [row for row in paths if row.get("predicate_type") not in ("data_quality",)]
    confidence = 0.35
    if diagnosis.group == "expert:port_status_gate":
        confidence = 0.72
    elif diagnosis.group == "expert:both_anomaly":
        confidence = 0.62
    elif diagnosis.group != "expert:no_anomaly":
        confidence = 0.58
    if pack.optical_blackout or not direct:
        confidence = min(confidence, 0.25)
    return {
        "case_id": pack.case_id,
        "verdict": verdict,
        "confidence": confidence,
        "policy": "cold-start-physical-only-v1",
        "reasoning": [
            "仅检查当前case量测质量、两端异常方向和lane关系，不使用任何训练历史",
            diagnosis.reason,
            f"可执行物理路径 {len(direct)} 条；缺测字段 {len(pack.missing_fields)} 个",
        ],
        "expert_diagnosis": diagnosis.to_dict(),
        "physical_evidence_paths": paths,
        "missing_information": list(pack.missing_fields),
    }


def _knowledge_prediction(cold: Mapping[str, Any], feature: Any, numeric: Any, result: Any, route: Any, bundle: Any) -> Dict[str, Any]:
    scores = {label: 0.0 for label in ROOT_CAUSES}
    reasons = []
    expert = cold["expert_diagnosis"]
    expert_candidate = bundle.expert_calibration.prediction(
        type("D", (), {"verdict": expert["verdict"], "group": expert["group"], "priority": expert["priority"], "reason": expert["reason"]})()
    ) if bundle.expert_calibration else None
    expert_weight = max(0.25, float((expert_candidate or {}).get("confidence_lower_bound", 0.0)))
    scores[cold["verdict"]] += expert_weight
    reasons.append(f"当前物理判断 {cold['verdict']}，训练内可靠性权重 {expert_weight:.3f}")

    candidates = list(result.dual_top_candidates or result.retrieval_candidates[:5])
    history_distribution: Counter[str] = Counter()
    for candidate in candidates[:5]:
        if candidate.label not in ROOT_CAUSES:
            continue
        weight = min(candidate.feature_similarity, candidate.graph_similarity)
        scores[candidate.label] += 0.75 * weight
        history_distribution[candidate.label] += 1
    if candidates:
        reasons.append(
            f"同拓扑历史候选标签 {dict(history_distribution)}；S_feature={result.max_feature_similarity:.3f}，S_graph={result.max_graph_similarity:.3f}"
        )

    sop = bundle.sop.predict(numeric).to_dict()
    if sop.get("verdict") in ROOT_CAUSES and sop.get("support", 0) >= 10:
        sop_weight = 0.35 * float(sop.get("confidence_lower_bound", 0.0))
        scores[sop["verdict"]] += sop_weight
        reasons.append(f"learned SOP {sop['verdict']}，support={sop['support']}，权重 {sop_weight:.3f}")

    fiber_direct = any("bidirectional_same_lane" in token for token in feature.tokens)
    if not fiber_direct and cold["verdict"] != "fiber":
        scores["fiber"] = min(scores["fiber"], max(scores["L1"], scores["L2"]) * 0.75)
        reasons.append("缺少双向同lane直接证据，fiber分数受准入门限制")
    best = max(scores.values())
    winners = [label for label in ROOT_CAUSES if scores[label] == best]
    verdict = winners[0]
    total = sum(scores.values()) or 1.0
    confidence = round(best / total, 6)
    if route.branch == "N5a" and result.is_label_pure:
        confidence = max(confidence, 0.72)
    return {
        "case_id": feature.case_id,
        "verdict": verdict,
        "confidence": confidence,
        "policy": POLICY_VERSION,
        "branch": route.branch,
        "scores": {key: round(value, 6) for key, value in scores.items()},
        "reasoning": reasons,
        "feature_tokens": list(feature.tokens),
        "history_candidates": [row.to_dict() for row in candidates[:5]],
        "learned_sop": sop,
    }


def predict(args: Any) -> None:
    bundle = OfflineKnowledgeBundle.load(args.knowledge_bundle)
    for name, split in SPLITS.items():
        raw_cases = cases_by_manifest_split(args.data_dir, split)
        # Prediction functions receive EvidencePack/CaseFeatures only. Labels are
        # structurally absent from both types and are not read in this phase.
        packs = tuple(build_packs(raw_cases, source_dataset=args.data_dir.name))
        _, features = bundle.extract_test_features(raw_cases, source_dataset=args.data_dir.name)
        numeric = numeric_features_from_packs(packs)
        results = match_many(bundle.graph, features, top_k=0)
        routes = route_many(results, FILTERED_RULE_THREE_CHANNEL_POLICY)
        cold = [_cold_prediction(pack, diag) for pack, diag in zip(packs, diagnose_many(packs))]
        enriched = [
            _knowledge_prediction(c, f, n, r, route, bundle)
            for c, f, n, r, route in zip(cold, features, numeric, results, routes)
        ]
        payload = {
            "schema_version": "blind-case-review-predictions-v1",
            "split": name,
            "label_access": False,
            "case_count": len(cold),
            "knowledge_bundle_hash": bundle.content_hash(),
            "cold_start": cold,
            "knowledge_enriched": enriched,
        }
        payload["prediction_hash"] = _hash(payload)
        _write(args.output_dir / name / "blind_predictions.json", payload)


def _metrics(rows: Sequence[Mapping[str, Any]], truth: Mapping[str, str]) -> Dict[str, Any]:
    correct = sum(row["verdict"] == truth[row["case_id"]] for row in rows)
    matrix: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[truth[row["case_id"]]][row["verdict"]] += 1
    return {
        "correct": correct, "total": len(rows), "accuracy": round(correct / len(rows), 6),
        "confusion": {label: dict(matrix[label]) for label in ROOT_CAUSES},
    }


def _render_html(path: Path, split: str, cold_m: Mapping[str, Any], rich_m: Mapping[str, Any], reviews: Sequence[Mapping[str, Any]]) -> None:
    cards = []
    for row in reviews:
        cards.append(
            f"<details><summary>{html.escape(row['case_id'])} · actual={row['actual']} · cold={row['cold']['verdict']} · enriched={row['enriched']['verdict']} · {row['category']}</summary>"
            f"<p><b>冷启动：</b>{html.escape('；'.join(row['cold']['reasoning']))}</p>"
            f"<p><b>知识增强：</b>{html.escape('；'.join(row['enriched']['reasoning']))}</p>"
            f"<p><b>复盘：</b>{html.escape(row['review'])}</p></details>"
        )
    document = f"""<!doctype html><meta charset='utf-8'><title>{split} blind review</title>
<style>body{{font-family:system-ui;max-width:1200px;margin:30px auto;line-height:1.55}}.metric{{display:inline-block;padding:14px;margin:6px;background:#eef;border-radius:8px}}details{{border:1px solid #ddd;padding:10px;margin:8px}}summary{{cursor:pointer;font-weight:600}}</style>
<h1>{split} 冷启动 vs 训练知识增强盲测</h1>
<div class='metric'>冷启动 {cold_m['correct']}/{cold_m['total']} = {cold_m['accuracy']:.2%}</div>
<div class='metric'>知识增强 {rich_m['correct']}/{rich_m['total']} = {rich_m['accuracy']:.2%}</div>
<p>两套预测均在读取测试标签前冻结。标签仅用于本页评估和复盘。</p>
<h2>逐 case</h2>{''.join(cards)}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def evaluate(args: Any) -> None:
    summaries = {}
    for name, split in SPLITS.items():
        payload = json.loads((args.output_dir / name / "blind_predictions.json").read_text(encoding="utf-8"))
        expected_hash = payload.pop("prediction_hash")
        if _hash(payload) != expected_hash or payload.get("label_access") is not False:
            raise ValueError(f"prediction freeze check failed for {name}")
        cases = cases_by_manifest_split(args.data_dir, split)
        truth = {str(row["case_id"]): str(row["label"]) for row in cases}
        cold_m = _metrics(payload["cold_start"], truth)
        rich_m = _metrics(payload["knowledge_enriched"], truth)
        reviews = []
        suspects = []
        for cold, rich in zip(payload["cold_start"], payload["knowledge_enriched"]):
            actual = truth[cold["case_id"]]
            c_ok, r_ok = cold["verdict"] == actual, rich["verdict"] == actual
            if not c_ok and r_ok:
                category, review = "knowledge_improved", "训练知识纠正了冷启动方向判断"
            elif c_ok and not r_ok:
                category, review = "knowledge_interference", "历史候选或learned SOP覆盖了正确的当前物理判断"
            elif c_ok and r_ok:
                category, review = "both_correct", "当前物理证据与训练知识一致"
            elif rich["branch"] == "N5a" and rich["confidence"] >= 0.72:
                category, review = "label_suspect", "纯净精确历史模式与当前标注冲突，需要人工复核标签或模式缺边"
                suspects.append({"case_id": cold["case_id"], "actual": actual, "predicted": rich["verdict"], "reason": review})
            elif rich["confidence"] < 0.45:
                category, review = "data_unidentifiable", "当前证据、历史和SOP均无法形成稳定区分"
            else:
                category, review = "model_or_knowledge_error", "物理方向规则、历史检索或融合权重需要复盘"
            reviews.append({"case_id": cold["case_id"], "actual": actual, "cold": cold, "enriched": rich, "category": category, "review": review})
        result = {"schema_version": "blind-case-review-evaluation-v1", "split": name, "prediction_hash": expected_hash, "cold_start": cold_m, "knowledge_enriched": rich_m, "category_counts": dict(Counter(row["category"] for row in reviews)), "reviews": reviews, "label_suspects": suspects}
        _write(args.output_dir / name / "evaluation.json", result)
        _write(args.output_dir / name / "label_suspects.json", {"cases": suspects})
        _render_html(args.output_dir / name / "report.html", name, cold_m, rich_m, reviews)
        summaries[name] = {
            "cold_start": cold_m,
            "knowledge_enriched": rich_m,
            "category_counts": result["category_counts"],
            "label_suspect_count": len(suspects),
        }
        print(name, "cold", cold_m, "enriched", rich_m, "categories", result["category_counts"])
    summary = {
        "schema_version": "blind-case-review-summary-v1",
        "policy_version": POLICY_VERSION,
        "prediction_label_access": False,
        "splits": summaries,
        "conclusion": (
            "当前训练知识融合未达到80%-90%，且两个split均低于冷启动；"
            "测试标签不得用于回调同轮权重，下一版必须只用train LOO重新设计门禁。"
        ),
    }
    _write(args.output_dir / "summary.json", summary)
    links = "".join(
        f"<li><a href='{name}/report.html'>{name}逐case报告</a></li>" for name in SPLITS
    )
    rows = "".join(
        f"<tr><td>{name}</td><td>{row['cold_start']['correct']}/{row['cold_start']['total']} ({row['cold_start']['accuracy']:.2%})</td>"
        f"<td>{row['knowledge_enriched']['correct']}/{row['knowledge_enriched']['total']} ({row['knowledge_enriched']['accuracy']:.2%})</td>"
        f"<td>{html.escape(str(row['category_counts']))}</td></tr>"
        for name, row in summaries.items()
    )
    (args.output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Blind RCA review</title>"
        "<style>body{font-family:system-ui;max-width:1100px;margin:30px auto;line-height:1.55}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:10px}</style>"
        "<h1>冷启动 vs 训练知识增强盲测总览</h1>"
        "<p>两套预测均在读取测试标签前冻结并记录hash。当前知识融合在两个split均造成净退化，不能宣称达到80%-90%。</p>"
        f"<table><tr><th>split</th><th>冷启动</th><th>知识增强</th><th>变化分类</th></tr>{rows}</table><ul>{links}</ul>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "evaluate", "all"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/filtered_rule_temporal_2025_06_09_v1"))
    parser.add_argument("--knowledge-bundle", type=Path, default=Path("artifacts/filtered_rule_deterministic_knowledge_v1/knowledge_bundle.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/filtered_rule_blind_case_review_v1"))
    args = parser.parse_args()
    if args.phase in ("predict", "all"):
        predict(args)
    if args.phase in ("evaluate", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
