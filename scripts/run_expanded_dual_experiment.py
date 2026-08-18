#!/usr/bin/env python3
"""Run the expanded expert-clean dual-similarity/SOP/LLM experiment.

The production decision is selective.  ``forced_prediction`` is emitted only
as an observational metric for comparison with the legacy forced classifier.
All fitting and reuse calibration use the clean training split only.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.branches.general import DiagnosisRequest
from rca_framework.constraints.library import CONSTRAINT_LIBRARY
from rca_framework.expanded_dual import (
    DUAL_MATCH_VERSION,
    LEARNED_PREDICATES,
    ROUTING_POLICY_VERSION,
    SOP_EXECUTOR_VERSION,
    DualMatcher,
    build_views,
    calibrate_dual_policy,
    execute_sop,
    raw_measurement_snapshot,
    route_dual,
    validate_expanded_llm_response,
)
from rca_framework.llm import ConstrainedReasoner, NoneBackend, VLLMBackend
from rca_framework.llm.backend import validate_context_window
from rca_framework.llm.prompts import build_prompt
from rca_framework.llm.prompts import PROMPT_TEMPLATE_VERSION, prompt_template_hash
from rca_framework.types import ROOT_CAUSES


SCHEMA_VERSION = "expanded-dual-sop-experiment-v1"
ABLATIONS = ("prior_only", "dual_history", "dual_history_sop", "dual_history_sop_llm")
METRICS = ("txpower", "rxpower", "media_snr", "host_snr", "serdes_snr", "bias")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten_numeric(value: Any) -> List[float]:
    if isinstance(value, Mapping):
        return [number for nested in value.values() for number in flatten_numeric(nested)]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return [float(value)]
    return []


def metric_summary(case: Mapping[str, Any], side: str, metric: str) -> Optional[float]:
    value = case.get(metric)
    if isinstance(value, Mapping):
        value = value.get(side)
    values = flatten_numeric(value)
    return median(values) if values else None


def train_scales(cases: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for side in ("L1", "L2"):
        for metric in METRICS:
            values = sorted(
                value for case in cases
                if (value := metric_summary(case, side, metric)) is not None
                and value not in (-40.0, 0.0, 1.0)
            )
            if not values:
                scales[f"{side}.{metric}"] = 1.0
                continue
            q1, q3 = values[len(values) // 4], values[(3 * len(values)) // 4]
            scales[f"{side}.{metric}"] = max(q3 - q1, 1e-9)
    return scales


def largest_differences(
    query: Mapping[str, Any], historical: Mapping[str, Any], scales: Mapping[str, float], limit: int = 8,
) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for side in ("L1", "L2"):
        for metric in METRICS:
            left, right = metric_summary(query, side, metric), metric_summary(historical, side, metric)
            if left is None or right is None:
                continue
            gap = abs(left - right) / scales[f"{side}.{metric}"]
            rows.append({
                "feature": f"{side}.{metric}.median", "current": left, "historical": right,
                "normalized_gap": round(gap, 6),
                "severity": "large" if gap >= 2.0 else "medium" if gap >= 0.75 else "small",
            })
    return tuple(sorted(rows, key=lambda item: -item["normalized_gap"])[:limit])


def class_metrics(rows: Sequence[Dict[str, Any]], prediction_key: str) -> Dict[str, Any]:
    confusion = {actual: {pred: 0 for pred in ROOT_CAUSES} for actual in ROOT_CAUSES}
    valid = [row for row in rows if row.get(prediction_key) in ROOT_CAUSES]
    for row in valid:
        confusion[row["actual_label"]][row[prediction_key]] += 1
    per_class = {}
    recalls = []
    f1s = []
    for label in ROOT_CAUSES:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in ROOT_CAUSES if actual != label)
        fn = sum(confusion[label][pred] for pred in ROOT_CAUSES if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
        recalls.append(recall)
        f1s.append(f1)
    return {
        "case_count": len(rows), "valid_prediction_count": len(valid),
        "accuracy": sum(row[prediction_key] == row["actual_label"] for row in valid) / len(valid) if valid else 0.0,
        "confusion_matrix": confusion, "per_class": per_class,
        "macro_f1": sum(f1s) / len(f1s), "balanced_recall": sum(recalls) / len(recalls),
    }


def summarize(rows: Sequence[Dict[str, Any]], ablation: str) -> Dict[str, Any]:
    selected = [row for row in rows if row["ablations"][ablation]["decision_label"] in ROOT_CAUSES]
    fiber_gold = sum(row["actual_label"] == "fiber" for row in rows)
    fiber_candidates = sum(
        row["actual_label"] == "fiber" and "fiber" in row["ablations"][ablation]["candidates"]
        for row in rows
    )
    selective_correct = sum(
        row["ablations"][ablation]["decision_label"] == row["actual_label"] for row in selected
    )
    forced_rows = [dict(row, _prediction=row["ablations"][ablation]["forced_prediction"]) for row in rows]
    result = class_metrics(forced_rows, "_prediction")
    result.update({
        "coverage": len(selected) / len(rows) if rows else 0.0,
        "selected_count": len(selected),
        "selected_correct": selective_correct,
        "precision_at_coverage": selective_correct / len(selected) if selected else 0.0,
        "human_intervention_rate": 1.0 - len(selected) / len(rows) if rows else 0.0,
        "fiber_candidate_recall": fiber_candidates / fiber_gold if fiber_gold else 0.0,
        "fiber_automatic_conclusion_count": sum(
            row["ablations"][ablation]["decision_label"] == "fiber" for row in rows
        ),
    })
    return result


def review_index(contract: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["case_id"]): row for row in contract.get("cases", [])
        if row.get("included") and row.get("split") == "test"
    }


def declared_predicates() -> Tuple[Dict[str, Any], ...]:
    physical = (
        {"predicate_id": "P_tx_or_rx_exact_minus_40", "boundary": "== -40.0", "source": "physical_sentinel"},
        {"predicate_id": "P_optical_engineering_drop", "boundary": "<= -39.0", "source": "engineering_constraint"},
        {"predicate_id": "P_media_or_host_snr_floor", "boundary": "<= 0.0", "source": "physical_sentinel"},
        {"predicate_id": "P_serdes_invalid", "boundary": "<= 1.0", "source": "measurement_contract"},
    )
    return physical + tuple(item.to_dict() for item in LEARNED_PREDICATES)


def make_request(case: Dict[str, Any], view: Any, match: Any, route: Any, sop: Any,
                 train_by_id: Mapping[str, Dict[str, Any]], scales: Mapping[str, float]) -> DiagnosisRequest:
    joint = match.joint_candidates
    nearest = match.candidates[0] if match.candidates else None
    opposing = tuple(
        item.to_dict() for item in match.candidates[:5]
        if joint and item.label != joint[0].label
    )
    diffs = largest_differences(case, train_by_id[nearest.case_id], scales) if nearest else ()
    constraints = tuple(item.constraint_id for item in CONSTRAINT_LIBRARY.constraints)
    return DiagnosisRequest(
        case_id=str(case["case_id"]), evidence_tokens=view.feature_tokens,
        missing_fields=view.missing_measurements, telemetry_status=view.quality,
        candidate_root_causes=sop.candidates or tuple(ROOT_CAUSES), exclusions=(),
        constraint_ids=constraints,
        nearest_similarity=max(match.max_feature_similarity, match.max_graph_similarity),
        branch=route.branch, routing_reason=route.reason,
        historical_case_ids=tuple(item.case_id for item in match.candidates[:5]),
        historical_label_distribution=tuple(sorted(match.label_distribution.items())),
        raw_measurements=raw_measurement_snapshot(case),
        feature_similarity=match.max_feature_similarity, graph_similarity=match.max_graph_similarity,
        evidence_paths=tuple(view.paths), opposing_historical_cases=opposing,
        largest_differences=diffs, critical_missing_evidence=sop.missing_information,
        declared_predicates=declared_predicates(),
        sop_trace=tuple(item.to_dict() for item in sop.steps), sop_candidates=sop.candidates,
        require_sop_step_ids=True,
    )


def decision(label: Optional[str], forced: str, candidates: Sequence[str], action: str, source: str) -> Dict[str, Any]:
    return {"decision_label": label, "forced_prediction": forced, "candidates": list(candidates),
            "decision_action": action, "source": source}


def calibrate_sop(cases: Sequence[Dict[str, Any]], views: Sequence[Any], majority: str) -> Dict[str, Any]:
    verdicts = [execute_sop(case, view, majority_label=majority).deterministic_verdict
                for case, view in zip(cases, views)]
    selected = [(verdict, case["label"]) for verdict, case in zip(verdicts, cases) if verdict]
    correct = sum(verdict == label for verdict, label in selected)
    precision = correct / len(selected) if selected else 0.0
    return {
        "support": len(selected), "correct": correct, "precision": precision,
        "maximum_selective_risk": 0.15, "minimum_support": 20,
        "automatic_conclusion_enabled": len(selected) >= 20 and precision >= 0.85,
    }


def render_html(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    cards = "".join(
        f"<article><h3>{html.escape(name)}</h3><p>coverage {metrics['coverage']:.1%} · "
        f"precision@coverage {metrics['precision_at_coverage']:.1%} · forced accuracy {metrics['accuracy']:.1%}</p></article>"
        for name, metrics in summary["all_test"].items()
    )
    details = []
    for row in rows:
        details.append(
            "<details><summary>" + html.escape(row["case_id"]) + " · " + html.escape(row["branch"])
            + " · label=" + html.escape(row["actual_label"]) + "</summary><pre>"
            + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + "</pre></details>"
        )
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Expanded dual RCA</title>
<style>body{{font:14px system-ui;margin:32px;max-width:1500px}}section{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}article,details{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:8px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.warn{{background:#fff4d6;padding:12px}}</style>
<h1>双相似度路由与受约束 SOP 实验</h1><p class='warn'>生产口径允许补采/人工介入；forced_prediction 仅用于历史对比。未审核样本不称为真实标签。N8 冻结。</p>
<h2>四组消融（341 条）</h2><section>{cards}</section>
<h2>LLM 权限建议</h2><pre>{html.escape(json.dumps(summary['llm_intervention_policy'], ensure_ascii=False, indent=2))}</pre>
<h2>逐 case 审计</h2>{''.join(details)}</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("none", "vllm"), default="none")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-preflight-only", action="store_true")
    args = parser.parse_args()

    train, test = read_jsonl(args.train_jsonl), read_jsonl(args.test_jsonl)
    if (len(train), len(test)) != (122, 341):
        raise ValueError(f"expanded-expert-clean-v1 requires 122/341, got {len(train)}/{len(test)}")
    contract = json.loads(args.data_contract.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "expanded-expert-clean-v1":
        raise ValueError("unexpected data contract")
    train_views, thresholds, feature_model, _ = build_views(train)
    test_views, _, _, test_packs = build_views(test, thresholds=thresholds, feature_model=feature_model)
    matcher = DualMatcher(train, train_views)
    policy = calibrate_dual_policy(train, train_views, matcher)
    majority = Counter(case["label"] for case in train).most_common(1)[0][0]
    sop_policy = calibrate_sop(train, train_views, majority)
    reviewed = review_index(contract)
    train_by_id = {str(case["case_id"]): case for case in train}
    scales = train_scales(train)

    work = []
    requests, request_indices = [], []
    for index, (case, view, pack) in enumerate(zip(test, test_views, test_packs)):
        match = matcher.match(case, view, feature_threshold=policy.feature_threshold,
                              graph_threshold=policy.graph_threshold)
        route = route_dual(match)
        sop = execute_sop(case, view, majority_label=majority)
        history_label = next(iter(match.label_distribution)) if match.strict_reuse else None
        record = {"case": case, "view": view, "pack": pack, "match": match, "route": route,
                  "sop": sop, "history_label": history_label, "request": None, "trace": None,
                  "llm_label": None, "llm_violations": ()}
        if route.branch in ("N5b", "N5c"):
            request = make_request(case, view, match, route, sop, train_by_id, scales)
            record["request"] = request
            requests.append(request)
            request_indices.append(index)
        work.append(record)

    if args.prompt_preflight_only:
        if not args.model_path:
            raise ValueError("--model-path is required for --prompt-preflight-only")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True,
        )
        backend_renderer = VLLMBackend(model_path=args.model_path)
        backend_renderer._tokenizer = tokenizer
        rendered = [backend_renderer._render(build_prompt(request)) for request in requests]
        lengths = validate_context_window(
            rendered, tokenizer, max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps({
            "schema_version": "expanded-prompt-context-preflight-v1",
            "request_count": len(lengths), "maximum_prompt_tokens": max(lengths, default=0),
            "minimum_prompt_tokens": min(lengths, default=0),
            "max_model_len": args.max_model_len, "max_new_tokens": args.max_new_tokens,
            "safety_tokens": 32,
            "remaining_tokens_at_maximum": args.max_model_len - max(lengths, default=0),
            "passed": True,
        }, ensure_ascii=False, indent=2))
        return 0

    backend = NoneBackend() if args.backend == "none" else VLLMBackend(
        model_path=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens, dtype=args.dtype, enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce, seed=args.seed,
    )
    try:
        traces = ConstrainedReasoner(backend=backend).reason_many(
            requests, [work[index]["pack"] for index in request_indices]
        ) if requests and args.backend != "none" else []
    finally:
        backend.close()
    for index, trace in zip(request_indices, traces):
        item = work[index]
        violations = validate_expanded_llm_response(trace.accepted, item["request"]) if trace.accepted else ("no response",)
        item["trace"], item["llm_violations"] = trace, violations
        if trace.accepted and not trace.accepted.forced and not violations:
            item["llm_label"] = trace.accepted.verdict

    rows = []
    for item in work:
        case, match, route, sop = item["case"], item["match"], item["route"], item["sop"]
        history = item["history_label"]
        trusted_sop_label = sop.deterministic_verdict if sop_policy["automatic_conclusion_enabled"] else None
        sop_label = history or trusted_sop_label
        llm_label = history or trusted_sop_label or item["llm_label"]
        forced_history = history or majority
        forced_sop = history or sop.forced_prediction
        forced_llm = llm_label or (item["trace"].accepted.verdict if item["trace"] and item["trace"].accepted else forced_sop)
        status = reviewed.get(str(case["case_id"]), {}).get("label_status", "unreviewed")
        rows.append({
            "case_id": str(case["case_id"]), "actual_label": str(case["label"]), "label_status": status,
            "branch": route.branch, "routing_reason": route.reason, "quality": item["view"].quality,
            "dual_match": match.to_dict(), "sop": sop.to_dict(),
            "llm_trace": item["trace"].to_dict() if item["trace"] else None,
            "llm_expanded_violations": list(item["llm_violations"]),
            "ablations": {
                "prior_only": decision(None, majority, ROOT_CAUSES, "human_review", "train_prior"),
                "dual_history": decision(history, forced_history, match.label_distribution or ROOT_CAUSES,
                                         "automatic_conclusion" if history else "human_review", "pure_dual_history"),
                "dual_history_sop": decision(sop_label, forced_sop, sop.candidates,
                                             "automatic_conclusion" if sop_label else sop.decision_action,
                                             "pure_dual_history" if history else "calibrated_deterministic_sop"),
                "dual_history_sop_llm": decision(llm_label, forced_llm, sop.candidates,
                                                 "automatic_conclusion" if llm_label else "request_evidence",
                                                 "pure_dual_history" if history else "calibrated_deterministic_sop" if trusted_sop_label else "constrained_llm"),
            },
            "production_decision": decision(
                sop_label, forced_sop, sop.candidates,
                "automatic_conclusion" if sop_label else sop.decision_action,
                "pure_dual_history" if history else "calibrated_deterministic_sop",
            ),
        })

    reviewed_rows = [row for row in rows if row["label_status"] == "expert_reviewed"]
    unreviewed_rows = [row for row in rows if row["label_status"] != "expert_reviewed"]
    all_metrics = {name: summarize(rows, name) for name in ABLATIONS}
    reviewed_metrics = {name: summarize(reviewed_rows, name) for name in ABLATIONS}
    unreviewed_metrics = {name: summarize(unreviewed_rows, name) for name in ABLATIONS}
    base, with_llm = reviewed_metrics["dual_history_sop"], reviewed_metrics["dual_history_sop_llm"]
    positive_gain = (
        with_llm["selected_correct"] > base["selected_correct"]
        and with_llm["precision_at_coverage"] >= base["precision_at_coverage"]
        and all_metrics["dual_history_sop_llm"]["precision_at_coverage"]
        >= all_metrics["dual_history_sop"]["precision_at_coverage"]
    )
    llm_policy = {
        "diagnostic_intervention_enabled": bool(args.backend == "vllm" and positive_gain),
        "default_permission": "diagnostic_intervention" if args.backend == "vllm" and positive_gain else "explanation_and_evidence_request_only",
        "criterion": "positive coverage/correctness gain on 42 reviewed cases without lower selective precision",
        "reviewed_base": base, "reviewed_with_llm": with_llm,
        "note": "The same reviewed set is used only to recommend future authority; production_decision in this run never consumes that recommendation.",
    }
    summary = {
        "schema_version": SCHEMA_VERSION, "train_size": len(train), "test_size": len(test),
        "reviewed_test_size": len(reviewed_rows), "unreviewed_test_size": len(unreviewed_rows),
        "branch_counts": dict(Counter(row["branch"] for row in rows)),
        "calibrated_policy": policy.to_dict(), "sop_selective_policy": sop_policy, "all_test": all_metrics,
        "expert_reviewed_test": reviewed_metrics, "unreviewed_test_observational": unreviewed_metrics,
        "llm_intervention_policy": llm_policy,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION, "backend": args.backend, "model_path": args.model_path,
        "seed": args.seed, "train_size": len(train), "test_size": len(test),
        "train_sha256": file_hash(args.train_jsonl), "test_sha256": file_hash(args.test_jsonl),
        "data_contract_sha256": file_hash(args.data_contract), "feature_dictionary_version": "expanded-explainable-features-v2",
        "evidence_graph_version": "expanded-five-layer-observable-graph-v2",
        "dual_match_version": DUAL_MATCH_VERSION, "routing_policy_version": ROUTING_POLICY_VERSION,
        "sop_version": SOP_EXECUTOR_VERSION, "constraint_library_version": CONSTRAINT_LIBRARY.version,
        "prompt_version": PROMPT_TEMPLATE_VERSION, "prompt_hash": prompt_template_hash(),
        "top_k": 5, "minimum_joint_candidates": 3, "n8_feedback_update": False,
        "label_leakage": False, "production_mode": "selective", "forced_mode": "observational_only",
        "calibrated_dual_policy": policy.to_dict(), "calibrated_sop_policy": sop_policy,
        "declared_predicates": list(declared_predicates()),
        "llm_config": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len, "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype, "enforce_eager": args.enforce_eager,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "requested_case_count": len(requests),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "predictions.json", rows)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "run_manifest.json", manifest)
    (args.output_dir / "report.html").write_text(render_html(summary, rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
