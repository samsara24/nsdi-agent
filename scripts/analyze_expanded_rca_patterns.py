#!/usr/bin/env python3
"""Build the expanded RCA test contract and an offline label-conflict HTML report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import (
    DOWN_THRESHOLDS,
    METRIC_ALIASES,
    STATUS_KEYS,
    abnormal_status,
    fit_thresholds,
    metric_values,
    percentile,
)
from rca_framework.evidence_graph import EvidenceGraph, match_many
from rca_framework.evidence_graph.match import weighted_jaccard
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.expanded_evidence import (
    EVIDENCE_STATE_VERSION,
    case_quality_state,
    fit_edge_idf,
    physical_evidence_paths,
    quality_compatible,
    weighted_edge_jaccard,
)
from rca_framework.features.dictionary import dictionary_for
from rca_framework.features.extractor import LEVEL_STATISTICS, extract_features, fit_feature_model, side_statistic


VOLATILE_FIELDS = {
    "case_id", "_meta", "alarm_time", "region", "link_location", "vendor", "vendor_sn",
    "alarm_ip_interface", "link_side_ip_interface_map", "task_id", "chip",
}


def load_cases(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("case_*.json"))]


def normalized_alarm(value: Any) -> str:
    text = str(value or "")
    return re.sub(r"^数通设备syslog告警:", "", text)


def physical_fingerprint(case: dict[str, Any]) -> str:
    value = {key: item for key, item in case.items() if key not in VOLATILE_FIELDS}
    if "alarm_name" in value:
        value["alarm_name"] = normalized_alarm(value["alarm_name"])
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value if isinstance(value, list) else value.get("predictions", [])
    return {str(row.get("case_id")): row for row in rows if row.get("case_id")}


def load_expert_annotations(path: Path | None, original_labels: dict[str, str]) -> dict[str, Any]:
    """Deduplicate pair annotations into a consistent per-case adjudication map."""
    if path is None:
        return {"schema_version": "expanded-expert-adjudication-v1", "cases": {}, "pairs": [], "changed_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "rca-expert-label-review-v1":
        raise ValueError(f"unsupported expert annotation schema: {payload.get('schema_version')}")
    cases: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    for annotation in payload.get("annotations", []):
        if not annotation.get("completed"):
            continue
        row = dict(annotation)
        row["requires_secondary_physics_review"] = bool(
            re.search(r"TX\s*-\s*RX|两端(?:的)?光衰", str(row.get("notes", "")), re.IGNORECASE)
        )
        pairs.append(row)
        for side in ("left", "right"):
            case_id = str(row[f"{side}_case_id"])
            if case_id not in original_labels:
                raise ValueError(f"expert annotation references unknown case: {case_id}")
            proposed = str(row.get(f"{side}_label", "keep"))
            if proposed in {"keep", "", "uncertain"}:
                adjudicated = original_labels[case_id]
            elif proposed in {"L1", "L2", "fiber"}:
                adjudicated = proposed
            else:
                raise ValueError(f"unsupported proposed label {proposed!r} for {case_id}")
            candidate = {
                "case_id": case_id,
                "original_label": original_labels[case_id],
                "adjudicated_label": adjudicated,
                "label_status": "reviewed_uncertain" if proposed == "uncertain" else "expert_reviewed",
                "expert_decisions": [row.get("decision", "")],
                "evidence_statuses": [row.get("evidence_status", "")],
                "notes": [row.get("notes", "")],
                "reviewed_at": row.get("updated_at") or payload.get("exported_at"),
                "review_source": str(path),
                "pattern_ids": [row.get("pattern_id")],
                "requires_secondary_physics_review": row["requires_secondary_physics_review"],
            }
            existing = cases.get(case_id)
            if existing is None:
                cases[case_id] = candidate
                continue
            if existing["adjudicated_label"] != adjudicated:
                raise ValueError(
                    f"inconsistent expert labels for {case_id}: "
                    f"{existing['adjudicated_label']} vs {adjudicated}"
                )
            for key in ("expert_decisions", "evidence_statuses", "notes", "pattern_ids"):
                existing[key] = sorted({*existing[key], *candidate[key]})
            existing["requires_secondary_physics_review"] |= candidate["requires_secondary_physics_review"]
            if candidate["reviewed_at"] and (not existing["reviewed_at"] or candidate["reviewed_at"] > existing["reviewed_at"]):
                existing["reviewed_at"] = candidate["reviewed_at"]
    return {
        "schema_version": "expanded-expert-adjudication-v1",
        "source_schema_version": payload.get("schema_version"),
        "source_path": str(path),
        "exported_at": payload.get("exported_at"),
        "pair_count": len(pairs),
        "case_count": len(cases),
        "changed_count": sum(item["original_label"] != item["adjudicated_label"] for item in cases.values()),
        "secondary_physics_review_pair_count": sum(row["requires_secondary_physics_review"] for row in pairs),
        "cases": cases,
        "pairs": pairs,
    }


def adjudicated_copy(case: dict[str, Any], adjudication: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(case)
    item = adjudication.get("cases", {}).get(str(case.get("case_id")))
    if item is not None:
        result["label"] = item["adjudicated_label"]
    return result


def augment_explainable_features(features: Any, case: dict[str, Any]) -> Any:
    """Add Q0/P/R tokens to the expanded feature dimension only."""
    physical_tokens = tuple(sorted({item["token"] for item in physical_evidence_paths(case)}))
    features.tokens = tuple(sorted(set(features.tokens) | set(physical_tokens)))
    features.by_family = dict(features.by_family)
    features.by_family["expanded_physical_state"] = physical_tokens
    features.dictionary_version = "expanded-explainable-features-v1"
    return features


def classification_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = ("L1", "L2", "fiber")
    confusion = {label: {guess: 0 for guess in labels} for label in labels}
    for truth, guess in zip(actual, predicted):
        if truth in confusion and guess in confusion[truth]:
            confusion[truth][guess] += 1
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[truth][label] for truth in labels)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": support, "predicted": predicted_count,
            "precision": round(precision, 8), "recall": round(recall, 8), "f1": round(f1, 8),
        }
    count = len(actual)
    correct = sum(truth == guess for truth, guess in zip(actual, predicted))
    majority = Counter(actual).most_common(1)[0] if actual else ("", 0)
    return {
        "case_count": count, "correct": correct, "accuracy": round(correct / count, 8) if count else 0.0,
        "label_distribution": dict(Counter(actual)), "prediction_distribution": dict(Counter(predicted)),
        "majority_label": majority[0], "majority_accuracy": round(majority[1] / count, 8) if count else 0.0,
        "balanced_recall": round(sum(float(per_label[label]["recall"]) for label in labels) / len(labels), 8),
        "macro_f1": round(sum(float(per_label[label]["f1"]) for label in labels) / len(labels), 8),
        "per_label": per_label, "confusion_matrix": confusion,
    }


def evaluate_prediction_rows(
    rows: list[dict[str, Any]], actual_by_case: dict[str, str], *, include: set[str] | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("case_id")) in actual_by_case and (include is None or str(row.get("case_id")) in include)]
    return classification_metrics(
        [actual_by_case[str(row["case_id"])] for row in selected],
        [str(row.get("prediction")) for row in selected],
    )


def token_title(token: str) -> str:
    return token.replace(":", " → ")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_value(value: Any, unit: str = "") -> str:
    if value is None:
        return "缺失"
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def chips(values: Iterable[str], css: str = "chip") -> str:
    rows = list(values)
    if not rows:
        return '<span class="muted">无</span>'
    return "".join(f'<span class="{css}">{esc(token_title(item))}</span>' for item in rows)


def token_logic(token: str) -> str:
    """Translate a feature token into a conservative physical statement."""
    parts = token.split(":")
    if len(parts) < 3:
        return f"记录到特征 {token}，但该 token 没有可展开的物理维度。"
    family, side, metric = parts[:3]
    side_name = f"{side} 端"
    metric_name = {
        "rxpower": "接收光功率", "txpower": "发送光功率", "media_snr": "介质侧 SNR",
        "host_snr": "主机侧 SNR", "serdes_snr": "SerDes 有效性",
        "RxLOS": "接收失光", "RxLOL": "接收失锁", "TxLOS": "发送失光", "TxLOL": "发送失锁",
    }.get(metric, metric)
    if family == "drop":
        bucket = parts[3] if len(parts) > 3 else "unknown"
        scope = {"single_lane": "单 lane", "partial_lanes": "多 lane 但非全部", "all_lanes": "全部 lane"}.get(bucket, bucket)
        if metric == "txpower":
            return f"{side_name}{scope}发送功率触底，首先说明该端发射链路没有形成有效出光；需先排除遥测 blackout，才能把它用于定位本端发射器。"
        if metric in {"rxpower", "media_snr"}:
            return f"{side_name}{scope}{metric_name}触底，说明信号到达/解调在接收方向失败；仅凭这一点无法区分对端发射、链路介质和本端接收器。"
        return f"{side_name}{scope}{metric_name}触底，表示该层信号失效。"
    if family == "status":
        if metric.startswith("Tx"):
            return f"{side_name}模块上报{metric_name}，这是本端发送方向的硬状态证据，但仍需与发送功率读数互相校验。"
        return f"{side_name}模块上报{metric_name}，确认接收方向异常；状态位能定位症状端，不能单独区分对端、fiber 或本端接收器。"
    if family == "imbalance":
        return f"{side_name}{metric_name}的 lane 间极差超过训练集稳健上界，提示单通道/局部问题，弱化“整束链路统一变化”的解释，但不直接给出 L1/L2/fiber 标签。"
    if family == "level" and len(parts) > 3:
        direction = parts[3]
        if "low" in direction:
            if metric.startswith("txpower"):
                return f"{side_name}平均发送功率处于训练分布低尾，支持本端发射偏弱；它可能让对端接收同步偏低。"
            if metric.startswith("rxpower"):
                return f"{side_name}平均接收功率处于训练分布低尾，支持入射光不足；原因仍可能是对端发射偏弱、fiber 衰减或本端接收耦合。"
            return f"{side_name}{metric_name}处于训练分布低尾，支持接收质量下降，但没有唯一根因方向。"
        return f"{side_name}{metric_name}处于训练分布高尾；这是幅值/质量偏高的上下文，不能按链路衰减证据解释。"
    return f"{side_name}{metric_name}产生特征 {token}；该特征只作为证据节点，不直接等价于根因标签。"


def _number_key(value: float | None) -> str:
    if value is None:
        return "unfitted"
    return f"{value:.8g}".replace("-", "minus_").replace(".", "p").replace("+", "")


def _gini(rows: list[tuple[float, str]]) -> float:
    counts = Counter(label for _, label in rows)
    total = len(rows)
    return 1.0 - sum((count / total) ** 2 for count in counts.values()) if total else 0.0


def _best_supervised_split(rows: list[tuple[float, str]], min_leaf: int) -> dict[str, Any] | None:
    ordered = sorted(rows)
    total = len(ordered)
    if total < 2 * min_leaf:
        return None
    base = _gini(ordered)
    best: dict[str, Any] | None = None
    for index in range(min_leaf, total - min_leaf + 1):
        if index >= total or ordered[index - 1][0] == ordered[index][0]:
            continue
        threshold = (ordered[index - 1][0] + ordered[index][0]) / 2.0
        left, right = ordered[:index], ordered[index:]
        gain = base - len(left) / total * _gini(left) - len(right) / total * _gini(right)
        candidate = {"threshold": threshold, "gain": gain, "left": left, "right": right}
        if best is None or (gain, -threshold) > (best["gain"], -best["threshold"]):
            best = candidate
    return best


def _predicate_value(case: dict[str, Any], kind: str, side: str, metric: str) -> float | None:
    if kind == "level":
        return side_statistic(case, side, metric)
    values = metric_values(case, metric, side)
    if metric in {"txpower", "rxpower"}:
        values = [value for value in values if value > -39.0]
    elif metric in {"media_snr", "host_snr"}:
        values = [value for value in values if value > 0.0]
    elif metric == "serdes_snr":
        values = [value for value in values if value > 1.0]
    return max(values) - min(values) if len(values) >= 2 else None


def _branch_audit(rows: list[tuple[float, str]]) -> dict[str, Any]:
    counts = Counter(label for _, label in rows)
    majority, majority_count = counts.most_common(1)[0]
    return {
        "support": len(rows),
        "label_distribution": {label: counts.get(label, 0) for label in ("L1", "L2", "fiber")},
        "majority_label": majority,
        "purity": round(majority_count / len(rows), 6),
    }


def fit_learned_predicate_model(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Learn stable one-dimensional ranges from train labels, never from test data.

    The label distribution is retained only as threshold provenance.  It is not
    materialized as a root-cause node in the observable evidence graph.
    """
    specs: list[tuple[str, str, str, str]] = []
    for side in ("L1", "L2"):
        specs.extend((f"level:{side}:{statistic}", "level", side, statistic) for statistic in sorted(LEVEL_STATISTICS))
        specs.extend((f"spread:{side}:{metric}", "spread", side, metric) for metric in sorted(METRIC_ALIASES))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case.get("label"))].append(case)
    folds = [
        [case for group in groups.values() for index, case in enumerate(group) if index % 5 != fold]
        for fold in range(5)
    ]
    baseline_counts = Counter(str(case.get("label")) for case in cases)
    baseline_total = len(cases)
    candidates: list[dict[str, Any]] = []
    accepted: dict[str, dict[str, Any]] = {}
    for key, kind, side, metric in specs:
        rows = [(_predicate_value(case, kind, side, metric), str(case.get("label"))) for case in cases]
        observed = [(value, label) for value, label in rows if value is not None]
        split = _best_supervised_split(observed, min_leaf=10)
        if split is None:
            candidates.append({"key": key, "accepted": False, "reason": "insufficient_support", "observed_count": len(observed)})
            continue
        fold_thresholds: list[float] = []
        for fold_cases in folds:
            fold_rows = [(_predicate_value(case, kind, side, metric), str(case.get("label"))) for case in fold_cases]
            fold_split = _best_supervised_split([(value, label) for value, label in fold_rows if value is not None], min_leaf=8)
            if fold_split is not None:
                fold_thresholds.append(fold_split["threshold"])
        value_span = max(value for value, _ in observed) - min(value for value, _ in observed)
        stability = (max(fold_thresholds) - min(fold_thresholds)) / value_span if len(fold_thresholds) == 5 and value_span else float("inf")
        left_audit, right_audit = _branch_audit(split["left"]), _branch_audit(split["right"])
        for audit in (left_audit, right_audit):
            audit["distribution_shift"] = round(0.5 * sum(
                abs(audit["label_distribution"][label] / audit["support"] - baseline_counts[label] / baseline_total)
                for label in ("L1", "L2", "fiber")
            ), 8)
        salient_branch = max(("le", "gt"), key=lambda branch: (
            (left_audit if branch == "le" else right_audit)["distribution_shift"],
            -(left_audit if branch == "le" else right_audit)["support"],
        ))
        reasons = []
        if len(observed) < 40:
            reasons.append("observed_count<40")
        if split["gain"] < 0.03:
            reasons.append("gini_gain<0.03")
        if stability > 0.15:
            reasons.append("fold_threshold_dispersion>0.15_span")
        if left_audit["majority_label"] == right_audit["majority_label"]:
            reasons.append("adjacent_ranges_same_majority")
        item = {
            "key": key, "kind": kind, "side": side, "metric": metric,
            "threshold": round(split["threshold"], 8), "gini_gain": round(split["gain"], 8),
            "observed_count": len(observed), "observed_range": [min(value for value, _ in observed), max(value for value, _ in observed)],
            "fold_thresholds": [round(value, 8) for value in fold_thresholds],
            "fold_dispersion_over_span": round(stability, 8) if stability != float("inf") else None,
            "branches": {"le": left_audit, "gt": right_audit},
            "salient_branch": salient_branch,
            "accepted": not reasons, "reason": "accepted" if not reasons else ";".join(reasons),
        }
        candidates.append(item)
        if not reasons:
            accepted[key] = item
    return {
        "version": "learned-predicate-ranges-v1",
        "fitted_case_count": len(cases),
        "label_distribution": dict(Counter(str(case.get("label")) for case in cases)),
        "parameters": {"max_depth": 1, "min_full_leaf": 10, "min_fold_leaf": 8, "min_observed": 40, "min_gini_gain": 0.03, "max_fold_dispersion_over_span": 0.15, "folds": 5, "require_different_adjacent_majority": True},
        "candidate_count": len(candidates), "accepted_count": len(accepted),
        "accepted": accepted, "candidates": candidates,
    }


def _learned_graph_specs(telemetry: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in model.get("accepted", {}).values():
        value = _predicate_value(telemetry, item["kind"], item["side"], item["metric"])
        if value is None:
            continue
        branch = "le" if value <= item["threshold"] else "gt"
        if branch != item["salient_branch"]:
            continue
        audit = item["branches"][branch]
        metric = item["metric"]
        metric_base = metric.removesuffix("_mean").removesuffix("_min")
        layer = "local_tx" if metric_base == "txpower" else "receive_path" if metric_base in {"rxpower", "media_snr"} else "local_electrical"
        relation = "le" if branch == "le" else "gt"
        predicate_key = f"learned_{relation}_{_number_key(item['threshold'])}"
        if item["kind"] == "spread":
            symptom = f"{metric_base}_lanes_within_learned_range" if branch == "le" else f"{metric_base}_lane_spread_above_learned_boundary"
        else:
            symptom = f"{metric_base}_in_learned_lower_range" if branch == "le" else f"{metric_base}_in_learned_upper_range"
        unit = "dBm" if "power" in metric else "dB" if metric_base != "serdes_snr" else "原字段单位"
        interval = f"(-∞, {item['threshold']:.6g}]" if branch == "le" else f"({item['threshold']:.6g}, +∞)"
        source = f"learned-range:{item['side']}:{metric}:{branch}:{_number_key(item['threshold'])}"
        counts = audit["label_distribution"]
        criterion = (
            f"{item['side']}.{metric}={value:.6g} {unit} ∈ {interval}；"
            f"阈值由清洗、专家裁决后的 train 监督学习；range support={audit['support']}，"
            f"labels=L1:{counts['L1']}/L2:{counts['L2']}/fiber:{counts['fiber']}；"
            f"Gini gain={item['gini_gain']:.4f}，5-fold 阈值离散/span={item['fold_dispersion_over_span']:.3f}"
        )
        specs.append({
            "side": f"side:{item['side']}", "measurement": f"measurement:{item['side']}:{metric}",
            "predicate": f"predicate:{item['side']}:{metric}:{predicate_key}",
            "symptom": f"symptom:{symptom}", "layer": f"physical-layer:{layer}",
            "token": source, "criterion": criterion, "learned": True,
        })
    return specs


def _comparison_features(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    metric_names = {
        "rxpower": "接收光功率", "txpower": "发送光功率", "media_snr": "介质侧 SNR",
        "host_snr": "主机侧 SNR", "serdes_snr": "SerDes 数值",
    }
    for side in ("L1", "L2"):
        for metric in sorted(METRIC_ALIASES):
            values = metric_values(case, metric, side)
            healthy = [value for value in values if value > DOWN_THRESHOLDS[metric]]
            unit = "dBm" if "power" in metric else "dB" if metric != "serdes_snr" else "原字段单位"
            prefix = f"{side}.{metric}"
            result[f"{prefix}.observed_lanes"] = {"value": len(values), "label": f"{side} {metric_names[metric]}观测 lane 数", "unit": "lane"}
            result[f"{prefix}.down_lanes"] = {"value": len(values) - len(healthy), "label": f"{side} {metric_names[metric]}掉底 lane 数", "unit": "lane"}
            if healthy:
                result[f"{prefix}.healthy_mean"] = {"value": sum(healthy) / len(healthy), "label": f"{side} {metric_names[metric]}健康 lane 均值", "unit": unit}
                result[f"{prefix}.healthy_min"] = {"value": min(healthy), "label": f"{side} {metric_names[metric]}健康 lane 最小值", "unit": unit}
                result[f"{prefix}.healthy_max"] = {"value": max(healthy), "label": f"{side} {metric_names[metric]}健康 lane 最大值", "unit": unit}
                if len(healthy) >= 2:
                    result[f"{prefix}.lane_spread"] = {"value": max(healthy) - min(healthy), "label": f"{side} {metric_names[metric]}lane 极差", "unit": unit}
        for status in STATUS_KEYS:
            block = case.get(status)
            value = block.get(side) if isinstance(block, dict) else None
            result[f"{side}.{status}.abnormal"] = {
                "value": None if value is None else int(abnormal_status(value)),
                "label": f"{side} {status} 是否异常", "unit": "布尔",
            }
    return result


def fit_comparison_model(cases: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    collected: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        for key, item in _comparison_features(case).items():
            value = item["value"]
            if isinstance(value, (int, float)):
                collected[key].append(float(value))
    model: dict[str, dict[str, float]] = {}
    for key, values in collected.items():
        q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
        median = percentile(values, 0.5) or 0.0
        iqr = (q3 - q1) if q1 is not None and q3 is not None else 0.0
        model[key] = {"scale": max(iqr, abs(median) * 0.05, 0.1), "q1": q1 or 0.0, "q3": q3 or 0.0}
    return model


def compare_case_features(
    left: dict[str, Any], right: dict[str, Any], model: dict[str, dict[str, float]],
) -> dict[str, Any]:
    left_values, right_values = _comparison_features(left), _comparison_features(right)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(left_values) | set(right_values)):
        left_item, right_item = left_values.get(key), right_values.get(key)
        left_value = left_item.get("value") if left_item else None
        right_value = right_item.get("value") if right_item else None
        meta = left_item or right_item or {"label": key, "unit": ""}
        if left_value is None and right_value is None:
            continue
        if left_value is None or right_value is None:
            score, severity, reason = 3.0, "large", "一侧缺失"
        else:
            delta = abs(float(left_value) - float(right_value))
            scale = model.get(key, {"scale": 1.0})["scale"]
            score = delta / scale
            severity = "large" if score >= 2.0 else "medium" if score >= 0.75 else "small"
            reason = f"差值={delta:.6g}，相当于训练 IQR 尺度的 {score:.2f} 倍"
            if delta == 0:
                continue
        rows.append({
            "key": key, "label": meta["label"], "unit": meta["unit"],
            "left": left_value, "right": right_value,
            "score": round(score, 6), "severity": severity, "reason": reason,
        })
    rows.sort(key=lambda item: (-item["score"], item["key"]))
    counts = Counter(item["severity"] for item in rows)
    return {
        "ranked": rows,
        "significant": [item for item in rows if item["severity"] in {"large", "medium"}],
        "small": [item for item in rows if item["severity"] == "small"],
        "counts": {key: counts.get(key, 0) for key in ("large", "medium", "small")},
        "largest": rows[0] if rows else None,
    }


def _token_graph_spec(token: str, thresholds: Any = None, feature_model: Any = None) -> dict[str, str] | None:
    """Map one feature token to the five-layer observable evidence schema."""
    parts = token.split(":")
    if len(parts) < 3:
        return None
    family, side, metric = parts[:3]
    bucket = parts[3] if len(parts) > 3 else "present"
    if family not in {"drop", "status", "imbalance", "level"}:
        return None
    if family == "level" and bucket == "high_tail":
        return None

    metric_base = metric.removesuffix("_mean").removesuffix("_min")
    if metric_base in {"txpower", "TxLOS", "TxLOL"}:
        layer = "local_tx"
    elif metric_base in {"rxpower", "media_snr", "RxLOS", "RxLOL"}:
        layer = "receive_path"
    elif metric_base in {"host_snr", "serdes_snr"}:
        layer = "local_electrical"
    else:
        return None

    if family == "drop":
        limit = {"rxpower": -39.0, "txpower": -39.0, "media_snr": 0.0, "host_snr": 0.0, "serdes_snr": 1.0}.get(metric)
        if limit is None:
            return None
        predicate_key = f"value_le_{_number_key(limit)}:{bucket}"
        symptom = {
            "txpower": "no_emitted_light",
            "rxpower": "no_received_light",
            "media_snr": "media_signal_unusable",
            "host_snr": "host_signal_unusable",
            "serdes_snr": "serdes_signal_unusable",
        }[metric]
    elif family == "status":
        predicate_key = "normalized_in_abnormal_status_set"
        symptom = {
            "TxLOS": "tx_loss_of_signal", "TxLOL": "tx_loss_of_lock",
            "RxLOS": "rx_loss_of_signal", "RxLOL": "rx_loss_of_lock",
        }.get(metric)
        if symptom is None:
            return None
    elif family == "imbalance":
        limit = thresholds.spread_upper.get(f"{side}:{metric}") if thresholds is not None else None
        predicate_key = f"healthy_lane_spread_gt_{_number_key(limit)}"
        symptom = {
            "txpower": "tx_lane_power_imbalance",
            "rxpower": "rx_lane_power_imbalance",
            "media_snr": "media_lane_quality_imbalance",
            "host_snr": "host_lane_quality_imbalance",
            "serdes_snr": "serdes_lane_quality_imbalance",
        }.get(metric)
        if symptom is None:
            return None
    elif family == "level" and bucket == "low_tail":
        low, _ = feature_model.level_edges.get(f"{side}:{metric}", (None, None)) if feature_model is not None else (None, None)
        predicate_key = f"value_lt_training_q25_{_number_key(low)}"
        symptom = {
            "txpower_mean": "weak_transmit_level",
            "rxpower_mean": "weak_receive_level",
            "media_snr_min": "degraded_media_signal_quality",
            "host_snr_min": "degraded_host_signal_quality",
            "serdes_snr_min": "degraded_serdes_signal_quality",
        }.get(metric)
        if symptom is None:
            return None
    else:
        return None

    return {
        "side": f"side:{side}",
        "measurement": f"measurement:{side}:{metric}",
        "predicate": f"predicate:{side}:{metric}:{predicate_key}",
        "symptom": f"symptom:{symptom}",
        "layer": f"physical-layer:{layer}",
    }


def observable_graph(
    tokens: Iterable[str],
    thresholds: Any = None,
    feature_model: Any = None,
    *,
    telemetry: dict[str, Any] | None = None,
    learned_predicates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project feature tokens into an auditable, label-free evidence subgraph.

    This is deliberately narrower than the feature vector.  Only tokens that can be
    mapped to a stable ``side -> measurement -> predicate -> symptom -> physical-layer``
    relation enter the graph;
    distributional high-tail context and unknown token families remain feature-only.
    The projection therefore measures graph coverage instead of pretending that every
    extracted feature already has a causal edge.
    """
    source_tokens = tuple(sorted(set(tokens)))
    mapped: list[str] = []
    paths: list[dict[str, Any]] = []
    nodes: set[str] = set()
    edges: set[str] = set()
    node_sources: dict[str, set[str]] = defaultdict(set)
    edge_sources: dict[str, set[str]] = defaultdict(set)
    graph_specs: list[dict[str, Any]] = []
    for token in source_tokens:
        if learned_predicates is not None and token.startswith(("level:", "imbalance:")):
            # Q25/Q75 and 3×IQR tokens remain in feature dimension A, but the graph
            # dimension replaces them with supervised, stability-gated ranges.
            continue
        # Raw physical boundary paths below supersede the generic drop projection.
        # Keeping both would count the same measurement twice and would retain the
        # historical serdes<=0 bug in the graph dimension.
        if telemetry is not None and token.startswith("drop:"):
            continue
        spec = _token_graph_spec(token, thresholds, feature_model)
        if spec is None:
            continue
        graph_specs.append({**spec, "token": token, "criterion": token_criterion(token, thresholds, feature_model), "learned": False})
        mapped.append(token)
    if telemetry is not None and learned_predicates is not None:
        graph_specs.extend(_learned_graph_specs(telemetry, learned_predicates))
    quality_state = case_quality_state(telemetry) if telemetry is not None else None
    if telemetry is not None:
        physical_specs = physical_evidence_paths(telemetry)
        graph_specs.extend(physical_specs)
        mapped.extend(item["token"] for item in physical_specs if item["token"] in source_tokens)
    for spec in graph_specs:
        token = spec["token"]
        current_nodes = tuple(spec[key] for key in ("side", "measurement", "predicate", "symptom", "layer"))
        current_edges = (
            f"{spec['side']}|has_measurement|{spec['measurement']}",
            f"{spec['measurement']}|satisfies|{spec['predicate']}",
            f"{spec['predicate']}|indicates|{spec['symptom']}",
            f"{spec['symptom']}|belongs_to|{spec['layer']}",
        )
        nodes.update(current_nodes)
        edges.update(current_edges)
        for node in current_nodes:
            node_sources[node].add(token)
        for edge in current_edges:
            edge_sources[edge].add(token)
        paths.append(spec)
    return {
        "nodes": tuple(sorted(nodes)),
        "edges": tuple(sorted(edges)),
        "mapped_tokens": tuple(sorted(set(mapped))),
        "unmapped_tokens": tuple(sorted(set(source_tokens) - set(mapped))),
        "feature_coverage": round(len(set(mapped)) / len(source_tokens), 6) if source_tokens else 0.0,
        "learned_range_path_count": sum(bool(item.get("learned")) for item in paths),
        "node_sources": {key: tuple(sorted(value)) for key, value in sorted(node_sources.items())},
        "edge_sources": {key: tuple(sorted(value)) for key, value in sorted(edge_sources.items())},
        "paths": tuple(sorted(paths, key=lambda item: (item["predicate"], item["token"]))),
        "quality_state": quality_state,
    }


def set_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return round(len(left_set & right_set) / len(union), 8)


def graph_match(
    left: dict[str, Any], right: dict[str, Any], edge_idf: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare typed graph edges independently from feature similarity.

    Nodes are retained for explanation only.  The score is IDF-weighted Jaccard
    over typed edges, so a shared generic symptom cannot hide different predicates.
    """
    left_nodes, right_nodes = set(left["nodes"]), set(right["nodes"])
    left_edges, right_edges = set(left["edges"]), set(right["edges"])
    node_similarity = set_jaccard(left_nodes, right_nodes)
    edge_similarity = weighted_edge_jaccard(left_edges, right_edges, edge_idf or {})
    score = edge_similarity
    shared_edges = tuple(sorted(left_edges & right_edges))
    right_paths = {item["predicate"]: item for item in right.get("paths", ())}
    shared_predicate_paths = tuple(
        item for item in left.get("paths", ()) if item["predicate"] in right_paths
    )
    return {
        "similarity": score,
        "node_similarity": node_similarity,
        "edge_similarity": edge_similarity,
        "formula": "IDF-weighted Jaccard over typed edges",
        "query_node_coverage": round(len(left_nodes & right_nodes) / len(left_nodes), 6) if left_nodes else 0.0,
        "query_edge_coverage": round(len(left_edges & right_edges) / len(left_edges), 6) if left_edges else 0.0,
        "shared_nodes": tuple(sorted(left_nodes & right_nodes)),
        "query_only_nodes": tuple(sorted(left_nodes - right_nodes)),
        "train_only_nodes": tuple(sorted(right_nodes - left_nodes)),
        "shared_edges": shared_edges,
        "shared_edge_sources": tuple(
            {
                "edge": edge,
                "query_tokens": tuple(left.get("edge_sources", {}).get(edge, ())),
                "train_tokens": tuple(right.get("edge_sources", {}).get(edge, ())),
            }
            for edge in shared_edges
        ),
        "shared_predicate_paths": shared_predicate_paths,
        "query_only_edges": tuple(sorted(left_edges - right_edges)),
        "train_only_edges": tuple(sorted(right_edges - left_edges)),
    }


def token_criterion(token: str, thresholds: Any, feature_model: Any) -> str:
    """Return the exact trigger/range behind a graph-mapped feature token."""
    parts = token.split(":")
    if len(parts) < 3:
        return "无可复核触发条件"
    family, side, metric = parts[:3]
    bucket = parts[3] if len(parts) > 3 else ""
    if family == "drop":
        limit = {"rxpower": -39.0, "txpower": -39.0, "media_snr": 0.0, "host_snr": 0.0, "serdes_snr": 1.0}.get(metric)
        scope = {
            "single_lane": "恰好 1 条 lane 达到阈值",
            "partial_lanes": "1 < 达阈值 lane 数 < 观测 lane 总数",
            "all_lanes": "所有已观测 lane 都达到阈值",
        }.get(bucket, bucket)
        unit = "dBm" if metric in {"rxpower", "txpower"} else "数值"
        return f"{side}.{metric} <= {limit:g} {unit}；{scope}" if limit is not None else f"{side}.{metric}；{scope}"
    if family == "status":
        return f"{side}.{metric} 归一化值∈{{Abnormal, Down, Fault, Error, LOS, LOL, True, 1}}"
    if family == "imbalance":
        limit = thresholds.spread_upper.get(f"{side}:{metric}") if thresholds is not None else None
        unit = "dB" if metric in {"rxpower", "txpower", "media_snr", "host_snr"} else "原字段单位"
        return f"{side}.{metric} 健康 lane 极差 > {limit:.4g} {unit}（清洗 train 拟合的 3×IQR 上界，仅保留在特征维度）" if limit is not None else "训练集未拟合出 spread 上界"
    if family == "level" and bucket == "low_tail":
        low, _ = feature_model.level_edges.get(f"{side}:{metric}", (None, None)) if feature_model is not None else (None, None)
        unit = "dBm" if "power" in metric else "dB"
        return f"{side}.{metric} < {low:.6g} {unit}（清洗 train 的 Q25，仅保留在特征维度）" if low is not None else "训练集未拟合出 Q25"
    return "该 token 是特征上下文，不建证据图决策边"


def render_html(data: dict[str, Any]) -> str:
    summary = data["summary"]
    learned_model = data["learned_predicates"]
    adjudication = data.get("adjudication", {"pairs": []})
    annotation_by_pair = {
        frozenset((str(row["left_case_id"]), str(row["right_case_id"]))): row
        for row in adjudication.get("pairs", [])
    }
    initial_annotations: dict[str, dict[str, Any]] = {}
    for pattern in data["patterns"]:
        train_id = next(str(row["case_id"]) for row in pattern["cases"] if row["split"] == "train")
        test_id = next(str(row["case_id"]) for row in pattern["cases"] if row["split"] == "test")
        existing = annotation_by_pair.get(frozenset((train_id, test_id)))
        if existing:
            initial_annotations[pattern["pattern_id"]] = {**existing, "pattern_id": pattern["pattern_id"]}
    initial_annotations_json = json.dumps(initial_annotations, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    raw_cases_json = json.dumps(data["raw_cases"], ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    metric_titles = {
        "original_343": "原始 343 标签",
        "clean_original": "清洗 341 原标签",
        "clean_partially_adjudicated": "清洗 341 部分专家标签",
        "expert_reviewed_clean_subset": "已审核测试子集",
        "clean_deterministic_rebuilt": "清洗后重建（无 LLM）",
    }
    metric_rows = "".join(
        "<tr>"
        f"<td><b>{esc(metric_titles[key])}</b></td><td>{value['case_count']}</td>"
        f"<td>{value['correct']}/{value['case_count']} = {value['accuracy']:.2%}</td>"
        f"<td>{value['majority_accuracy']:.2%}</td><td>{value['balanced_recall']:.2%}</td>"
        f"<td>{value['macro_f1']:.2%}</td>"
        f"<td>{value['per_label']['L1']['recall']:.2%}</td><td>{value['per_label']['L2']['recall']:.2%}</td>"
        f"<td>{value['per_label']['fiber']['recall']:.2%} (n={value['per_label']['fiber']['support']})</td>"
        "</tr>"
        for key, value in summary.get("remote_metrics", {}).items()
    ) or '<tr><td colspan="9" class="muted">尚未加载远端预测。</td></tr>'
    threshold_rows = "".join(
        f"<tr><td>{row['threshold']:.2f}</td><td>{row['both_high_different_label_pairs']}</td></tr>"
        for row in summary.get("threshold_sensitivity", [])
    )
    added_rows = "".join(
        f"<tr><td>{esc(row['case_id'])}</td><td><b>{esc(row['label'])}</b></td><td>{esc(row['source_file'])}</td><td>是</td></tr>"
        for row in data["added_cases"]
    )
    learned_rows = "".join(
        "<tr>"
        f"<td><code>{esc(item['key'])}</code></td><td>{item['threshold']:.6g}</td>"
        f"<td>{item['gini_gain']:.4f}</td><td>{item['fold_dispersion_over_span']:.3f}</td><td>{esc(item['salient_branch'])}</td>"
        f"<td>≤: n={item['branches']['le']['support']}，{esc(item['branches']['le']['label_distribution'])}<br>"
        f">: n={item['branches']['gt']['support']}，{esc(item['branches']['gt']['label_distribution'])}</td>"
        "</tr>"
        for item in learned_model["accepted"].values()
    ) or '<tr><td colspan="6">没有候选范围通过稳定性门禁。</td></tr>'
    rejected_rows = "".join(
        f"<tr><td><code>{esc(item['key'])}</code></td><td>{esc(item['reason'])}</td></tr>"
        for item in learned_model["candidates"] if not item["accepted"]
    )
    focus_cards = "".join(
        f'''<section><h3>{esc(row["left_case_id"])} ↔ {esc(row["right_case_id"])}</h3>
        <button class="compare-primary" type="button" data-left="{esc(row["left_case_id"])}" data-right="{esc(row["right_case_id"])}">打开这两条 case 的原始数据对比</button>
        <div class="cards"><div class="card"><span>S_feature</span><strong>{row["feature_similarity"]:.3f}</strong></div><div class="card"><span>S_graph</span><strong>{row["graph_similarity"]:.3f}</strong></div><div class="card"><span>左侧特征入图</span><strong>{row["left_graph_coverage"]:.1%}</strong></div><div class="card"><span>右侧特征入图</span><strong>{row["right_graph_coverage"]:.1%}</strong></div></div>
        <p>{esc(row["interpretation"])}</p></section>'''
        for row in data.get("focus_pairs", [])
    )
    pattern_cards = []
    for pattern in data["patterns"]:
        test_id = next(row["case_id"] for row in pattern["cases"] if row["split"] == "test")
        train_id = next(row["case_id"] for row in pattern["cases"] if row["split"] == "train")
        comparisons = "".join(
            "<tr>"
            f"<td><button class=\"case-link\" type=\"button\" data-left=\"{esc(train_id if row['split'] == 'test' else row['case_id'])}\" data-right=\"{esc(row['case_id'] if row['split'] == 'test' else test_id)}\">{esc(row['case_id'])}</button></td><td>{esc(row['split'])}</td>"
            f"<td>{esc(row['original_label'])}</td><td><b>{esc(row['label'])}</b><br><span class=\"muted\">{esc(row['label_status'])}</span></td>"
            f"<td>{esc(row.get('prediction') or '待远端')}</td><td>{row['feature_similarity']:.3f}</td><td>{row['graph_similarity']:.3f}</td><td>{esc(row['quadrant'])}</td>"
            f"<td>{chips(row['tokens'])}</td></tr>"
            for row in pattern["cases"]
        )
        logic_rows = "".join(
            f'<li><code>{esc(item["token"])}</code><span>{esc(item["logic"])}</span></li>'
            for item in pattern["physical_logic"]
        )
        shared_terms = "".join(
            f'<tr><td><code>{esc(item["token"])}</code></td><td>{item["idf"]:.4f}</td><td>{esc(item["meaning"])}</td></tr>'
            for item in pattern["weighted_shared_terms"]
        )
        graph_shared_nodes = chips(pattern["graph_match"]["shared_nodes"], "chip shared")
        graph_query_only = chips(pattern["graph_match"]["query_only_nodes"], "chip extra")
        graph_train_only = chips(pattern["graph_match"]["train_only_nodes"], "chip missing")
        graph_edges = "".join(
            f'<tr><td><code>{esc(edge.replace("|", " → "))}</code></td></tr>'
            for edge in pattern["graph_match"]["shared_edges"]
        ) or '<tr><td class="muted">无共享边</td></tr>'
        graph_criteria = "".join(
            "<tr>"
            f'<td><code>{esc(" → ".join(item["path"]))}</code></td>'
            f'<td>{chips(item["source_tokens"])}</td>'
            f'<td>{"<br>".join(esc(value) for value in item["criteria"])}</td>'
            f'<td>{esc(item["predicate_type"])}</td><td>{esc(item["quantifier"])}</td><td>{esc(item["provenance"])}</td>'
            "</tr>"
            for item in pattern["shared_relation_criteria"]
        ) or '<tr><td colspan="6" class="muted">无共享关系触发条件</td></tr>'
        evidence_paths = "".join(
            f'''<div class="evidence-path">
              <div class="graph-node"><b>端口侧</b><span>{esc(item["path"][0].removeprefix("side:"))}</span></div><div class="arrow">→</div>
              <div class="graph-node"><b>测量指标</b><span>{esc(item["path"][1].split(":")[-1])}</span></div><div class="arrow">→</div>
              <div class="graph-node predicate"><b>条件谓词</b><span>{esc(item["criteria"][0])}</span></div><div class="arrow">→</div>
              <div class="graph-node"><b>症状</b><span>{esc(item["path"][3].removeprefix("symptom:"))}</span></div><div class="arrow">→</div>
              <div class="graph-node"><b>物理层</b><span>{esc(item["path"][4].removeprefix("physical-layer:"))}</span></div>
            </div>'''
            for item in pattern["shared_relation_criteria"]
        ) or '<p class="muted">没有完整共享的五层证据路径。</p>'
        priority_label = {"critical": "最高优先", "high": "高优先", "standard": "常规复核"}[pattern["review_priority"]]
        difference_rows = "".join(
            "<tr>"
            f'<td><span class="diff-badge {esc(item["severity"])}">{"大" if item["severity"] == "large" else "中"}</span></td>'
            f'<td>{esc(item["label"])}</td><td>{esc(fmt_value(item["left"], item["unit"]))}</td>'
            f'<td>{esc(fmt_value(item["right"], item["unit"]))}</td><td>{esc(item["reason"])}</td>'
            "</tr>"
            for item in pattern["feature_differences"]["significant"][:8]
        ) or '<tr><td colspan="5" class="muted">没有达到“中/大”门槛的测量差异；这是优先核查标签的强信号。</td></tr>'
        small_difference_rows = "".join(
            "<tr>"
            f'<td>{esc(item["label"])}</td><td>{esc(fmt_value(item["left"], item["unit"]))}</td>'
            f'<td>{esc(fmt_value(item["right"], item["unit"]))}</td><td>{esc(item["reason"])}</td>'
            "</tr>"
            for item in pattern["feature_differences"]["small"]
        ) or '<tr><td colspan="4" class="muted">无细微差异</td></tr>'
        largest = pattern["feature_differences"]["largest"]
        largest_text = (
            f'{largest["label"]}：{fmt_value(largest["left"], largest["unit"])} ↔ {fmt_value(largest["right"], largest["unit"])}'
            if largest else "没有可比较的测量差异"
        )
        pattern_cards.append(f"""
        <details class="pattern" data-pattern-id="{esc(pattern['pattern_id'])}" data-priority="{esc(pattern['review_priority'])}" data-search="{esc(train_id + ' ' + test_id)}" {'open' if len(pattern_cards) < 4 else ''}>
          <summary><span class="danger">{esc(pattern['pattern_id'])}</span> · {esc(pattern['summary'])}
            <span class="priority {esc(pattern['review_priority'])}">{priority_label}</span><span class="pill">特征 {pattern['feature_similarity']:.3f}</span><span class="pill graph-pill">图 {pattern['graph_similarity']:.3f}</span></summary>
          <div class="review-core"><p><b>为什么需要专家：</b>{esc(pattern['review_reason'])}</p><p><b>最大测量差异：</b>{esc(largest_text)}</p><p><b>差异分级：</b>大 {pattern['feature_differences']['counts']['large']} · 中 {pattern['feature_differences']['counts']['medium']} · 小 {pattern['feature_differences']['counts']['small']}</p><p><b>严格完全匹配：</b>{'是' if pattern['exact_two_dimensional_match'] else '否'}；质量状态兼容={'是' if pattern['quality_compatible'] else '否'}；关键缺失冲突={esc(pattern['critical_evidence_conflicts'] or '无')}</p></div>
          <h4>优先查看：大/中测量差异</h4><div class="tablewrap compact"><table><thead><tr><th>级别</th><th>特征</th><th>{esc(train_id)}</th><th>{esc(test_id)}</th><th>归一化依据</th></tr></thead><tbody>{difference_rows}</tbody></table></div>
          <details class="minor-diffs"><summary>查看 {pattern['feature_differences']['counts']['small']} 项细微差异（默认不突出）</summary><div class="tablewrap compact"><table><thead><tr><th>特征</th><th>{esc(train_id)}</th><th>{esc(test_id)}</th><th>归一化依据</th></tr></thead><tbody>{small_difference_rows}</tbody></table></div></details>
          <button class="compare-primary" type="button" data-left="{esc(train_id)}" data-right="{esc(test_id)}">对比 {esc(train_id)} ↔ {esc(test_id)} 的原始数据</button>
          <section class="expert-review" data-pattern-id="{esc(pattern['pattern_id'])}" data-left-id="{esc(train_id)}" data-right-id="{esc(test_id)}">
            <h4>专家标注</h4><div class="annotation-grid">
              <label>审核结论<select data-annotation-field="decision"><option value="">未选择</option><option value="test_label_suspect">测试 case 标签可疑</option><option value="train_label_suspect">历史 case 标签可疑</option><option value="both_valid">两个标签都可能正确，现有模式不可辨识</option><option value="both_suspect">两个标签都需要复核</option><option value="insufficient_evidence">证据不足，暂不能判断</option></select></label>
              <label>{esc(train_id)} 建议标签<select data-annotation-field="left_label"><option value="keep">保持 {esc(str(pattern['cases'][0]['label']))}</option><option value="L1">L1</option><option value="L2">L2</option><option value="fiber">fiber</option><option value="uncertain">不确定</option></select></label>
              <label>{esc(test_id)} 建议标签<select data-annotation-field="right_label"><option value="keep">保持 {esc(str(pattern['test_label']))}</option><option value="L1">L1</option><option value="L2">L2</option><option value="fiber">fiber</option><option value="uncertain">不确定</option></select></label>
              <label>证据充分性<select data-annotation-field="evidence_status"><option value="">未选择</option><option value="sufficient">证据充分</option><option value="missing_noncritical">缺失证据可接受</option><option value="missing_critical">缺失关键证据</option><option value="telemetry_suspect">遥测值可疑</option></select></label>
            </div><label>专家备注<textarea data-annotation-field="notes" rows="3" placeholder="说明标签判断、关键证据或需要补采的字段"></textarea></label><label class="complete-check"><input type="checkbox" data-annotation-field="completed"> 本组审核完成</label><span class="save-status" aria-live="polite">自动保存在本浏览器</span>
          </section>
          <div class="evidence-paths" aria-label="五层可观测证据图"><h4>共享的五层证据路径</h4>{evidence_paths}</div>
          <p><b>可定位边界：</b>{esc(pattern['identifiability_boundary'])} <b>标签冲突：</b>train={esc(pattern['train_labels'])}；test={esc(pattern['test_label'])}</p>
          <div class="grid two">
            <section><h4>逐 token 物理解释</h4><ul class="logic-list">{logic_rows}</ul></section>
            <section><h4>Step by step（保留）</h4><ol class="steps"><li><b>N1</b> 去标签后标准化两端遥测</li><li><b>N2</b> 抽取固定 feature-dictionary-v1 token</li><li><b>N3-A 特征召回</b> 独立计算 IDF 加权 Jaccard={pattern['feature_similarity']:.3f}</li><li><b>N3-B 图校验</b> 离散物理谓词与训练监督学习范围共同构造五层子图，节点/边匹配得分={pattern['graph_similarity']:.3f}</li><li><b>谓词校验</b> 指标名、学习区间或离散契约、lane 范围必须同时一致，才共享完整谓词路径</li><li><b>图覆盖</b> 当前 case 生成 {pattern['query_graph']['learned_range_path_count']} 条学习范围路径；未通过范围门禁的连续证据保留给后续 LLM 检查</li><li><b>确认路径</b> 数据集无人工确认 reasoning_path，路径相似度标记为不可评估</li><li class="danger"><b>结果</b> 二维坐标为 {esc(pattern['quadrant'])}，且历史标签与测试标签冲突</li></ol></section>
          </div>
          <details class="formula-detail"><summary>维度 A：查看可解释特征相似度明细</summary>
            <p><code>S_feature = Σ IDF(交集 token) / Σ IDF(并集 token) = {pattern['shared_weight']:.4f} / {pattern['union_weight']:.4f} = {pattern['feature_similarity']:.3f}</code></p>
            <div class="tablewrap compact"><table><thead><tr><th>共享 token</th><th>IDF 权重</th><th>物理含义</th></tr></thead><tbody>{shared_terms}</tbody></table></div>
            <p><b>当前 case 独有：</b>{chips(pattern['query_only_evidence'], 'chip extra')}<br><b>历史 case 独有：</b>{chips(pattern['train_only_evidence'], 'chip missing')}</p>
          </details>
          <details class="formula-detail"><summary>维度 B：查看可观测证据子图匹配明细</summary>
            <p class="warn"><b>图类型：五层可观测证据关系图，不是决策树。</b>它保留指标、谓词阈值和物理症状，但没有互斥分支、顺序判定和 root-cause 叶子。</p>
            <p><code>S_graph = Σ IDF(共享 typed edge) / Σ IDF(并集 typed edge) = {pattern['graph_similarity']:.3f}</code></p>
            <p>节点 Jaccard={pattern['graph_match']['node_similarity']:.3f} 只用于解释覆盖，不进入图分数。</p>
            <p><b>共享节点：</b>{graph_shared_nodes}<br><b>当前独有节点：</b>{graph_query_only}<br><b>历史独有节点：</b>{graph_train_only}</p>
            <div class="tablewrap compact"><table><thead><tr><th>共享有向类型边</th></tr></thead><tbody>{graph_edges}</tbody></table></div>
            <h4>共享五层路径的触发阈值 / 范围</h4><div class="tablewrap compact"><table><thead><tr><th>五层关系链</th><th>来源 token</th><th>必须满足的条件</th><th>谓词类型</th><th>量词</th><th>来源</th></tr></thead><tbody>{graph_criteria}</tbody></table></div>
            <p><b>图覆盖边界：</b>测试 case 有 {pattern['query_graph']['learned_range_path_count']} 条学习范围路径，历史 case 有 {pattern['train_graph']['learned_range_path_count']} 条。未通过稳定性门禁的连续特征和未映射 token 不会被丢弃，只是不参与当前图分数，留给后续 LLM 判断缺失或特殊值。</p>
            <p class="warn"><b>确认归因路径：不可评估。</b>当前训练 case 没有人工确认的 SOPStep/ConstraintCheck 边，因此本报告只比较可观测子图，不伪造因果路径相似度。</p>
          </details>
          <p><b>同模式依据：</b>{esc(pattern['why_same_pattern'])}</p>
          <p><b>标签不一致解释：</b>{esc(pattern['label_conflict_analysis'])}</p>
          <p><b>评估影响：</b>{esc(pattern['impact'])}</p>
          <div class="tablewrap"><table><thead><tr><th>case_id</th><th>split</th><th>原标签</th><th>当前评估标签 / 状态</th><th>prediction</th><th>S_feature</th><th>S_graph</th><th>二维分区</th><th>feature tokens</th></tr></thead><tbody>{comparisons}</tbody></table></div>
        </details>""")
    missing_old = "".join(f"<li>{esc(row['case_id'])} · {esc(row['label'])} · {esc(row['source_file'])}</li>" for row in data["missing_old_cases"])
    flow = """
    <div class="flow"><span>N1 标准化</span><b>→</b><span>N2 特征 token</span><b>→</b><span class="changed">N3 历史匹配冲突审计（本轮）</span><b>→</b><span>N4/N5 保持不变</span><b>→</b><span>N6 降级</span></div>
    """
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>扩充 RCA 证据模式冲突分析</title><style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#667085;--line:#dfe4ee;--accent:#2855d9;--danger:#b42318;--warn:#b54708;--green:#067647;--purple:#6941c6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}}button{{font:inherit}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:30px;margin:0 0 6px}}h2{{margin-top:30px}}h4{{margin:0 0 10px}}.muted{{color:var(--muted)}}.warn{{color:var(--warn)}}.banner{{background:#fffaeb;border:1px solid #fedf89;padding:14px 18px;border-radius:12px;margin:18px 0}}.explain{{border-left:5px solid var(--accent)}}.formula{{font-size:16px;background:#f8fafc;padding:12px;border-radius:8px;overflow:auto}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px}}.card,section,.pattern{{background:var(--card);border:1px solid var(--line);border-radius:12px}}.card{{padding:16px}}.card strong{{font-size:28px;display:block}}section{{padding:18px;margin:12px 0}}.flow{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;background:#fff;padding:16px;border-radius:12px;border:1px solid var(--line)}}.flow span{{padding:8px 10px;background:#eef2ff;border-radius:8px}}.flow .changed{{background:#fee4e2;color:var(--danger);font-weight:700}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid var(--line)}}th{{background:#f8fafc;position:sticky;top:0}}.tablewrap{{overflow:auto;max-height:520px}}.compact{{max-height:300px}}.pattern{{margin:12px 0;padding:0 16px}}.pattern>summary{{cursor:pointer;padding:16px 0;font-size:16px;font-weight:700}}.danger{{color:var(--danger);font-weight:700}}.pill{{font-size:12px;color:var(--accent);background:#eef2ff;border-radius:999px;padding:3px 8px;margin-left:8px}}.graph-pill{{color:var(--purple);background:#f4f3ff}}.chip{{display:inline-block;padding:3px 7px;margin:2px;background:#f2f4f7;border-radius:6px;font:12px ui-monospace,SFMono-Regular,monospace}}.shared{{background:#ecfdf3;color:#067647}}.extra{{background:#eff8ff;color:#175cd3}}.missing{{background:#fff4ed;color:#b93815}}.grid.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.grid.two section{{margin:0}}code{{word-break:break-all}}.evidence-paths{{margin:14px 0}}.evidence-path{{display:grid;grid-template-columns:minmax(90px,.65fr) 24px minmax(120px,.8fr) 24px minmax(220px,1.8fr) 24px minmax(150px,1fr) 24px minmax(120px,.8fr);align-items:stretch;gap:5px;margin:8px 0}}.graph-node{{padding:11px;border-radius:9px;background:#f8fafc;border:1px solid var(--line);min-width:0}}.graph-node b,.graph-node span{{display:block}}.graph-node span{{margin-top:5px;overflow-wrap:anywhere}}.graph-node.predicate{{background:#fffaeb;border-color:#fedf89}}.arrow{{display:grid;place-items:center;font-size:20px;color:var(--muted)}}.logic-list{{padding-left:20px;margin:0}}.logic-list li{{margin:0 0 11px}}.logic-list code,.logic-list span{{display:block}}.logic-list span{{color:#344054;margin-top:3px}}.steps{{padding-left:22px}}.steps li{{margin-bottom:8px}}.compare-primary,.case-link{{color:var(--accent);background:none;border:0;text-decoration:underline;cursor:pointer;padding:0}}.compare-primary{{border:1px solid #b2ccff;background:#eff4ff;text-decoration:none;border-radius:8px;padding:8px 12px;margin-bottom:4px;font-weight:700}}.formula-detail{{border-top:1px dashed var(--line);border-bottom:1px dashed var(--line);margin:14px 0;padding:0 4px}}.formula-detail>summary{{cursor:pointer;padding:10px 0;font-weight:700}}dialog{{width:min(1400px,96vw);max-height:92vh;border:0;border-radius:14px;padding:0;box-shadow:0 20px 60px #10182855}}dialog::backdrop{{background:#10182899}}.dialog-head{{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border-bottom:1px solid var(--line);position:sticky;top:0;background:white;z-index:2}}.dialog-head h3{{margin:0}}.dialog-close{{border:1px solid var(--line);background:white;border-radius:7px;padding:6px 10px;cursor:pointer}}.dialog-body{{padding:16px;overflow:auto}}.raw-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.raw-grid pre{{white-space:pre-wrap;word-break:break-word;background:#101828;color:#f2f4f7;padding:12px;border-radius:8px;max-height:420px;overflow:auto}}.diff-row{{background:#fff7ed}}.same-row{{color:#667085}}@media(max-width:1000px){{.evidence-path{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}}}@media(max-width:800px){{main{{padding:14px}}.grid.two,.raw-grid{{grid-template-columns:1fr}}h1{{font-size:24px}}}}
/* Expert review surface */
.review-toolbar{{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:end;flex-wrap:wrap;background:#fff;border:1px solid var(--line);padding:12px;border-radius:10px;margin:14px 0}}.review-toolbar label{{display:grid;gap:4px}}.review-toolbar input,.review-toolbar select,.expert-review select,.expert-review textarea{{border:1px solid #98a2b3;border-radius:7px;padding:7px;background:#fff;color:var(--ink)}}.review-toolbar button{{border:1px solid #84adff;background:#eff4ff;color:#1849a9;border-radius:7px;padding:8px 12px;cursor:pointer}}.review-progress{{font-weight:700;margin-left:auto}}.priority,.diff-badge{{display:inline-block;border-radius:999px;padding:2px 8px;font-size:12px;margin-left:7px}}.priority.critical,.diff-badge.large{{background:#fee4e2;color:#b42318}}.priority.high,.diff-badge.medium{{background:#fef0c7;color:#b54708}}.priority.standard{{background:#f2f4f7;color:#475467}}.review-core{{background:#f8fafc;border-left:4px solid #84adff;padding:8px 12px;margin:8px 0}}.review-core p{{margin:4px 0}}.minor-diffs{{margin:8px 0 12px}}.expert-review{{background:#f0f9ff;border-color:#b9e6fe;margin:14px 0}}.annotation-grid{{display:grid;grid-template-columns:repeat(2,minmax(240px,1fr));gap:10px}}.annotation-grid label,.expert-review>label{{display:grid;gap:4px}}.complete-check{{display:flex!important;grid-template-columns:auto 1fr!important;align-items:center;margin-top:10px}}.save-status{{display:inline-block;color:var(--green);margin-top:6px}}.methodology{{margin:28px 0;border:1px solid var(--line);border-radius:12px;background:#fff;padding:0 16px}}.methodology>summary{{cursor:pointer;padding:16px 0;font-size:18px;font-weight:700}}.pattern[hidden]{{display:none}}.diff-large{{background:#fef3f2}}.diff-medium{{background:#fffaeb}}.same-row,.diff-small,.diff-metadata{{display:none}}.show-all .same-row,.show-all .diff-small,.show-all .diff-metadata{{display:table-row}}@media(max-width:800px){{.annotation-grid{{grid-template-columns:1fr}}.review-progress{{margin-left:0;width:100%}}}}
</style></head><body><main><h1>RCA 相似 Case 标签审核</h1><p class="muted">生成于 {esc(data['created_at'])} · 离线专家标注工具 · 标注自动保存在当前浏览器</p>
<h2>1. 最终远端实验效果</h2><section><p><b>结论：</b>当前 DeepSeek-32B 在原始 343 条标签上为 209/343（60.93%），低于全预测 L2 的 61.22%；fiber 仍为 0 召回。下表中的“部分专家标签”只是对同一批旧预测重新计分，不代表重训结果。</p><div class="tablewrap"><table><thead><tr><th>评估口径</th><th>case</th><th>accuracy</th><th>多数类</th><th>balanced recall</th><th>macro-F1</th><th>L1 recall</th><th>L2 recall</th><th>fiber recall</th></tr></thead><tbody>{metric_rows}</tbody></table></div></section>
<h2>2. 数据质量与清洗契约</h2><div class="cards"><div class="card"><span>清洗训练集</span><strong>{summary['clean_train_size']}</strong></div><div class="card"><span>清洗测试集</span><strong>{summary['expanded_test_size']}</strong></div><div class="card"><span>剔除 blackout</span><strong>{summary['excluded_old_count']}</strong></div><div class="card"><span>专家审核 case</span><strong>{summary['expert_reviewed_case_count']}</strong></div><div class="card"><span>标签调整</span><strong>{summary['expert_changed_case_count']}</strong></div><div class="card"><span>测试已审核 / 未审核</span><strong>{summary['reviewed_clean_test_count']} / {summary['unreviewed_clean_test_count']}</strong></div></div>
<section><p><b>标签口径：</b>专家确认 case 使用 adjudicated label；其余 case 保留 original label，但明确标记为 <code>unreviewed</code>。这不是完整真实标签集。依赖跨端 <code>TX-RX</code> 绝对损耗的 {summary['secondary_physics_review_pair_count']} 个 pair 已单独输出，未写入物理约束。</p></section>
<h2>3. 物理边界与证据优先级</h2><section><table><thead><tr><th>层</th><th>明确边界</th><th>解释规则</th></tr></thead><tbody><tr><td>Q0 数据质量</td><td>双端 TX/RX 全部精确 -40 且 media_snr 全部 ≤0</td><td>先判 optical blackout / 遥测无效，不解释为激光关断。</td></tr><tr><td>P 光功率状态</td><td>精确 -40.0；工程 drop ≤-39 dBm</td><td>精确哨兵与工程异常区间分开记录。</td></tr><tr><td>P SNR/SerDes</td><td>media/host ≤0；SerDes ≤1</td><td>SerDes 只作有效/失效状态，不能按 dB 解释。</td></tr><tr><td>R 端间关系</td><td>只做 side-level TX→RX / RX→decode 关系</td><td>禁止逐 lane 配对和绝对 TX-RX loss。</td></tr><tr><td>L 学习范围</td><td>只在清洗后的有效 train 读数上拟合</td><td>学习阈值是统计决策边界，不是物理常数。</td></tr></tbody></table></section>
<h2>4. 双高异标签候选</h2>
<div class="banner"><b>审核目标：</b>共找到 <b>{summary['pattern_count']}</b> 组 S_feature 与 S_graph 双高异标签 case；为保持页面简单，默认只显示最高/高优先的 <b>{summary['review_priority_counts'].get('critical', 0) + summary['review_priority_counts'].get('high', 0)}</b> 组，其余可在筛选器中查看。</div>
<div class="review-toolbar" aria-label="专家审核工具栏"><label>搜索 case ID<input id="review-search" type="search" placeholder="输入 case ID"></label><label>审核优先级<select id="priority-filter"><option value="priority" selected>最高 + 高优先</option><option value="all">全部</option><option value="critical">最高优先</option><option value="high">高优先</option><option value="standard">常规复核</option></select></label><label>完成状态<select id="completion-filter"><option value="all">全部</option><option value="pending">未完成</option><option value="completed">已完成</option></select></label><button id="export-annotations" type="button">导出专家标注 JSON</button><span class="review-progress"><span id="completed-count">0</span> / {summary['pattern_count']} 已完成</span></div>
{('<h2>用户指定 case 对复核</h2>' + focus_cards) if focus_cards else ''}
<h2 id="review-list">待审核：高相似但标签不同</h2>{''.join(pattern_cards) if pattern_cards else '<section>未发现达到阈值的标签冲突模式。</section>'}
<details class="methodology"><summary>方法、数据与阈值审计（标注时可跳过）</summary>
<section class="explain"><h2>为什么不运行大模型也能得到这份报告？</h2><p>本报告评估的是<b>确定性特征抽取、图构建与历史检索</b>，而不是大模型的根因预测。维度 A 使用固定 token；维度 B 使用原始遥测生成 Q0/P/R 路径，并只从清洗后的 {summary['clean_train_size']} 条 train 学习 L 层连续范围。测试标签只在匹配完成后用于检查冲突。</p><div class="grid two"><section><h3>维度 A：可解释特征</h3><div class="formula"><code>S_feature = Σ IDF(交集 token) / Σ IDF(并集 token)</code></div><p>使用 feature-dictionary-v1 加 Q0/P/R 可解释状态 token。稀有 token 权重更大。</p></section><section><h3>维度 B：五层可观测证据图</h3><div class="formula"><code>S_graph = Σ IDF(共享 typed edge) / Σ IDF(并集 typed edge)</code></div><p>节点只用于解释，只有完整的“端口侧→测量指标→条件谓词→症状→物理层”类型边进入图分数。</p></section></div><p class="warn"><b>重要限制：</b>学习区间是训练集统计决策边界，不是物理常数，也不生成候选根因节点。训练集 {summary['graph_evaluation']['confirmed_training_paths_total']} 条 case 中人工确认诊断路径为 {summary['graph_evaluation']['confirmed_training_paths_available']} 条，所以因果/SOP 路径仍不可评估。</p><p>报告用于展示的临时阈值是 S_feature≥{summary['feature_similarity_threshold']:.2f}、S_graph≥{summary['graph_similarity_threshold']:.2f}；它们只用于二维分区，没有修改线上 N4 路由。严格完全匹配还要求两个分数均为 1、质量状态兼容且无关键缺失冲突。</p></section>
{flow}<h2>五层证据图改造计划</h2><section><table><thead><tr><th>阶段</th><th>改造内容</th><th>本版状态</th></tr></thead><tbody>
<tr><td>1. 范围学习</td><td>只用清洗、专家裁决后的 {summary['clean_train_size']} 条训练 case 学习一维切分；物理哨兵先剔除，之后再做支持数、Gini gain 和 5-fold 稳定性门禁。</td><td><b>已完成</b></td></tr>
<tr><td>2. 五层 schema</td><td>每个可审计 token 构造成“端口侧→测量指标→条件谓词→症状→物理层”。</td><td><b>已完成</b></td></tr>
<tr><td>3. 精确图匹配</td><td>指标名、学习区间和 lane 量词进入节点身份；不同指标只允许共享上层症状，不再共享完整谓词路径。</td><td><b>已完成</b></td></tr>
<tr><td>4. 路由标定</td><td>用专家确认路径重新标定图权重和 N4 阈值。</td><td>待人工路径数据；当前仅离线审计</td></tr>
</tbody></table></section><h2>训练监督学习范围</h2><section><p>{learned_model['candidate_count']} 个候选连续特征中，{learned_model['accepted_count']} 个通过门禁。标签仅用于学习区间，不在图中生成 L1/L2/fiber 节点；只有相对训练基线分布偏移更大的特殊分支进入图，普通分支保留为未触发背景。</p><div class="tablewrap"><table><thead><tr><th>特征</th><th>学习阈值</th><th>Gini gain</th><th>5-fold 离散/span</th><th>入图分支</th><th>两侧训练分布</th></tr></thead><tbody>{learned_rows}</tbody></table></div><details><summary>查看未通过门禁的候选</summary><div class="tablewrap compact"><table><thead><tr><th>特征</th><th>拒绝原因</th></tr></thead><tbody>{rejected_rows}</tbody></table></div></details></section><h2>实验概览</h2><div class="cards">
<div class="card"><span>原训练集</span><strong>{summary['old_train_size']}</strong></div><div class="card"><span>清洗训练集</span><strong>{summary['clean_train_size']}</strong></div><div class="card"><span>确认保留旧样本</span><strong>{summary['retained_old_count']}</strong></div><div class="card"><span>新增样本</span><strong>{summary['added_count']}</strong></div><div class="card"><span>清洗测试集</span><strong>{summary['expanded_test_size']}</strong></div><div class="card"><span>剔除旧样本</span><strong class="danger">{summary['excluded_old_count']}</strong></div></div>
<h2>两个维度的整体评估</h2><div class="grid two"><section><h3>A. 可解释特征检索</h3><div class="cards"><div class="card"><span>Top-1 历史标签一致率</span><strong>{summary['feature_evaluation']['top1_label_agreement']:.1%}</strong></div><div class="card"><span>Top-1 特征完全相同</span><strong>{summary['feature_evaluation']['exact_top1_count']}</strong></div><div class="card"><span>Top-1 S_feature≥阈值</span><strong>{summary['feature_evaluation']['high_top1_count']}</strong></div></div><p>它评估“抽取出的特征集像不像”，不代表原始字段完全相同，也不代表归因路径相同。</p></section><section><h3>B. 五层证据图检索</h3><div class="cards"><div class="card"><span>Top-1 历史标签一致率</span><strong>{summary['graph_evaluation']['top1_label_agreement']:.1%}</strong></div><div class="card"><span>Top-1 S_graph≥阈值</span><strong>{summary['graph_evaluation']['high_top1_count']}</strong></div><div class="card"><span>平均学习范围路径</span><strong>{summary['graph_evaluation']['average_learned_range_paths']:.2f}</strong></div></div><p>它要求指标和稳定范围谓词进入图结构；没有通过门禁的连续特征留在“未覆盖证据”，供后续 LLM 判断。<b>这个图仍不是决策树</b>，也不输出候选根因。</p></section></div>
<section><h3>二维异标签冲突分区</h3><table><thead><tr><th>分区</th><th>case 数</th><th>含义</th></tr></thead><tbody><tr><td><b>both_high</b></td><td>{summary['conflict_quadrants']['both_high']}</td><td>特征与可观测图都相似，是最强的模式冲突审核集。</td></tr><tr><td><b>feature_only</b></td><td>{summary['conflict_quadrants']['feature_only']}</td><td>特征签名像，但物理关系图不够像；不应直接复用历史标签。</td></tr><tr><td><b>graph_only</b></td><td>{summary['conflict_quadrants']['graph_only']}</td><td>物理关系形态像，但完整特征差异大；适合 N5b 补证据。</td></tr></tbody></table></section>
<section><h3>审核阈值敏感性</h3><p>以下只统计候选池中两个维度同时达到同一阈值、且标签不同的 pair；不修改 N4。</p><table><thead><tr><th>S_feature 与 S_graph 阈值</th><th>异标签双高 pair</th></tr></thead><tbody>{threshold_rows}</tbody></table></section>
<section><h3>数据契约</h3><p><b>旧：</b><code>{esc(summary['old_data_dir'])}</code><br><b>新：</b><code>{esc(summary['new_data_dir'])}</code></p><p>去重依据：脱敏 ID 不同，采用剔除脱敏/时间字段、规范化 syslog 前缀后的物理遥测 + 告警语义 + label SHA-256 指纹。新集不是严格超集；下列旧 case 未在新集找到：</p><ul>{missing_old or '<li>无</li>'}</ul></section>
<h2>新增样本（{summary['added_count']}）</h2><section class="tablewrap"><table><thead><tr><th>case_id</th><th>label</th><th>来源</th><th>加入测试</th></tr></thead><tbody>{added_rows}</tbody></table></section>
<h2>远端实验状态</h2><section><p>{esc(summary['remote_status'])}</p><p>脚本：<code>run_expanded_remote_experiment.sh</code>；预期回传：<code>predictions_expanded.json</code>、<code>evaluation_expanded.json</code>、<code>run_manifest.json</code>、<code>run.log</code>、<code>metadata.json</code>。</p><p>当前 pattern 判断分别使用训练集拟合的 feature-model-v1 和可观测证据子图；两种分数都不使用大模型，也没有伪造预测值。</p></section>
<h2>总结与建议</h2><section><ul><li>旧报告的“证据图相似度”实际上是 feature token IDF-Jaccard；本版已更名并与图节点/边相似度拆开。</li><li><b>both_high</b> 是优先人工审核集；<b>feature_only</b> 与 <b>graph_only</b> 都不允许直接当作 N5a 完全匹配。</li><li>无人工确认诊断边时，不能声称“推理路径相同”；本报告只能证明可观测子图相似。</li><li>只有 S_feature=1.00 且 S_graph=1.00 的异标签 case 进入当前 label_suspects / irreducible 强审核集。</li><li>先人工核查 6 条旧样本消失原因；远端预测回传后再检查模型是否被历史标签误导。</li></ul></section>
</details>
<dialog id="case-dialog"><div class="dialog-head"><h3 id="dialog-title">原始 case 对比</h3><button class="dialog-close" type="button">关闭</button></div><div class="dialog-body"><div id="diff-summary"></div><label><input id="show-all-raw-diffs" type="checkbox"> 显示细微、元数据和相同字段</label><div class="tablewrap"><table><thead><tr><th>字段路径</th><th id="left-head">训练 case</th><th id="right-head">测试 case</th></tr></thead><tbody id="diff-body"></tbody></table></div><div class="raw-grid"><details><summary>训练 case 完整 JSON</summary><pre id="left-raw"></pre></details><details><summary>测试 case 完整 JSON</summary><pre id="right-raw"></pre></details></div></div></dialog>
<script id="raw-cases" type="application/json">{raw_cases_json}</script><script id="initial-annotations" type="application/json">{initial_annotations_json}</script><script>
const rawCases=JSON.parse(document.getElementById('raw-cases').textContent);const dlg=document.getElementById('case-dialog');
const initialAnnotations=JSON.parse(document.getElementById('initial-annotations').textContent);
const storageKey='rca-expert-annotations:20260816-expanded-pattern-conflict:v1';
function loadAnnotations(){{try{{return JSON.parse(localStorage.getItem(storageKey)||'{{}}');}}catch(_e){{return {{}};}}}}
let annotations={{...initialAnnotations,...loadAnnotations()}};
function persistAnnotations(){{try{{localStorage.setItem(storageKey,JSON.stringify(annotations));return true;}}catch(_e){{return false;}}}}
function flat(v,p='',o={{}}){{if(v&&typeof v==='object'&&!Array.isArray(v)){{Object.keys(v).sort().forEach(k=>flat(v[k],p?`${{p}}.${{k}}`:k,o));}}else if(Array.isArray(v)){{v.forEach((x,i)=>flat(x,`${{p}}[${{i}}]`,o));}}else{{o[p]=v;}}return o;}}
function rawSeverity(key,a,b){{if(JSON.stringify(a)===JSON.stringify(b))return 'same-row';if(/(^|\\.)(case_id|label|_meta|alarm_time|region|vendor|vendor_sn|task_id|link_location|alarm_ip_interface)/.test(key))return 'diff-metadata';if(a===undefined||b===undefined)return 'diff-large';if(typeof a==='number'&&typeof b==='number'){{const ratio=Math.abs(a-b)/Math.max(Math.abs(a),Math.abs(b),1);return ratio>=.2?'diff-large':ratio>=.05?'diff-medium':'diff-small';}}return 'diff-medium';}}
function showCompare(leftId,rightId){{const l=rawCases[leftId],r=rawCases[rightId];if(!l||!r)return;const lf=flat(l.raw),rf=flat(r.raw),keys=[...new Set([...Object.keys(lf),...Object.keys(rf)])].sort();const counts={{large:0,medium:0,small:0,metadata:0,same:0}};const body=document.getElementById('diff-body');body.replaceChildren();body.classList.remove('show-all');document.getElementById('show-all-raw-diffs').checked=false;for(const k of keys){{const cls=rawSeverity(k,lf[k],rf[k]);if(cls==='diff-large')counts.large++;else if(cls==='diff-medium')counts.medium++;else if(cls==='diff-small')counts.small++;else if(cls==='diff-metadata')counts.metadata++;else counts.same++;const tr=document.createElement('tr');tr.className=cls;for(const value of [k,lf[k],rf[k]]){{const td=document.createElement('td');td.textContent=value===undefined?'∅':typeof value==='string'?value:JSON.stringify(value);tr.appendChild(td);}}body.appendChild(tr);}}document.getElementById('dialog-title').textContent=`原始数据追溯：${{leftId}} ↔ ${{rightId}}`;document.getElementById('left-head').textContent=`${{leftId}} · train · ${{l.label}}`;document.getElementById('right-head').textContent=`${{rightId}} · test · ${{r.label}}`;document.getElementById('diff-summary').innerHTML=`<p>默认只显示原始字段中的大/中差异：大 <b class="danger">${{counts.large}}</b>，中 <b>${{counts.medium}}</b>。细微 ${{counts.small}}、元数据 ${{counts.metadata}}、相同 ${{counts.same}} 项默认隐藏。证据强弱请以前面的训练 IQR 归一化特征表为准。</p>`;document.getElementById('left-raw').textContent=JSON.stringify(l.raw,null,2);document.getElementById('right-raw').textContent=JSON.stringify(r.raw,null,2);if(dlg&&typeof dlg.showModal==='function')dlg.showModal();}}
function collectAnnotation(section){{const row={{pattern_id:section.dataset.patternId,left_case_id:section.dataset.leftId,right_case_id:section.dataset.rightId,updated_at:new Date().toISOString()}};section.querySelectorAll('[data-annotation-field]').forEach(el=>{{row[el.dataset.annotationField]=el.type==='checkbox'?el.checked:el.value;}});return row;}}
function restoreAnnotation(section){{const row=annotations[section.dataset.patternId];if(!row)return;section.querySelectorAll('[data-annotation-field]').forEach(el=>{{if(!(el.dataset.annotationField in row))return;if(el.type==='checkbox')el.checked=Boolean(row[el.dataset.annotationField]);else el.value=row[el.dataset.annotationField];}});}}
function updateProgress(){{const completed=Object.values(annotations).filter(row=>row.completed).length;document.getElementById('completed-count').textContent=String(completed);applyFilters();}}
document.querySelectorAll('.expert-review').forEach(section=>{{restoreAnnotation(section);const save=()=>{{annotations[section.dataset.patternId]=collectAnnotation(section);const saved=persistAnnotations();section.querySelector('.save-status').textContent=saved?'已自动保存':'浏览器禁止本地保存，请及时导出 JSON';updateProgress();}};section.addEventListener('input',save);section.addEventListener('change',save);}});
function applyFilters(){{const query=document.getElementById('review-search').value.trim().toLowerCase();const priority=document.getElementById('priority-filter').value;const completion=document.getElementById('completion-filter').value;document.querySelectorAll('.pattern').forEach(card=>{{const done=Boolean(annotations[card.dataset.patternId]?.completed);const priorityMatch=priority==='all'||card.dataset.priority===priority||(priority==='priority'&&['critical','high'].includes(card.dataset.priority));const show=(!query||card.dataset.search.toLowerCase().includes(query))&&priorityMatch&&(completion==='all'||(completion==='completed'&&done)||(completion==='pending'&&!done));card.hidden=!show;}});}}
['review-search','priority-filter','completion-filter'].forEach(id=>document.getElementById(id).addEventListener('input',applyFilters));
document.getElementById('export-annotations').addEventListener('click',()=>{{const payload={{schema_version:'rca-expert-label-review-v1',experiment_id:'20260816_expanded-pattern-conflict',exported_at:new Date().toISOString(),annotations:Object.values(annotations)}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='expert_label_annotations.json';document.body.appendChild(link);link.click();link.remove();URL.revokeObjectURL(url);}});
document.getElementById('show-all-raw-diffs').addEventListener('change',event=>document.getElementById('diff-body').classList.toggle('show-all',event.target.checked));
document.querySelectorAll('[data-left][data-right]').forEach(b=>b.addEventListener('click',()=>showCompare(b.dataset.left,b.dataset.right)));const closeButton=document.querySelector('.dialog-close');if(closeButton&&dlg&&typeof closeButton.addEventListener==='function')closeButton.addEventListener('click',()=>dlg.close());if(dlg&&typeof dlg.addEventListener==='function')dlg.addEventListener('click',e=>{{if(e.target===dlg)dlg.close();}});updateProgress();
</script></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data-dir", type=Path, required=True)
    parser.add_argument("--new-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--graph-similarity-threshold", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--focus-pair", action="append", default=[], metavar="TRAIN_ID,CASE_ID")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--expert-annotations", type=Path)
    parser.add_argument("--remote-run-dir", type=Path)
    parser.add_argument("--deterministic-run-dir", type=Path)
    parser.add_argument(
        "--clean-missing-old", action=argparse.BooleanOptionalAction, default=False,
        help="remove all old cases absent from the new dataset before fitting/evaluation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_cases, new_cases = load_cases(args.old_data_dir), load_cases(args.new_data_dir)
    old_fps = defaultdict(list)
    for case in old_cases:
        old_fps[physical_fingerprint(case)].append(case)
    retained_fps = {physical_fingerprint(case) for case in new_cases} & set(old_fps)
    added = [case for case in new_cases if physical_fingerprint(case) not in old_fps]
    missing_old = [case for case in old_cases if physical_fingerprint(case) not in {physical_fingerprint(item) for item in new_cases}]
    if args.clean_missing_old:
        non_blackout_missing = [str(case.get("case_id")) for case in missing_old if not case_quality_state(case)["optical_blackout"]]
        if non_blackout_missing:
            raise ValueError(
                "refusing to remove missing old cases that are not verified optical blackout: "
                + ", ".join(non_blackout_missing)
            )
    old_train, old_test = old_cases[: args.train_size], old_cases[args.train_size :]
    missing_ids = {str(case["case_id"]) for case in missing_old}
    clean_train_raw = [case for case in old_train if not args.clean_missing_old or str(case["case_id"]) not in missing_ids]
    clean_test_raw = [case for case in old_test if not args.clean_missing_old or str(case["case_id"]) not in missing_ids]
    expanded_test_raw = clean_test_raw + added
    all_source_cases = {str(case["case_id"]): case for case in old_cases + new_cases}
    original_labels = {case_id: str(case.get("label")) for case_id, case in all_source_cases.items()}
    adjudication = load_expert_annotations(args.expert_annotations, original_labels)
    clean_train = [adjudicated_copy(case, adjudication) for case in clean_train_raw]
    expanded_test = [adjudicated_copy(case, adjudication) for case in expanded_test_raw]

    added_manifest = []
    new_by_id = {str(case["case_id"]): index + 1 for index, case in enumerate(new_cases)}
    for case in added:
        number = new_by_id[str(case["case_id"])]
        added_manifest.append({
            "case_id": case["case_id"], "label": case.get("label"),
            "source_file": f"case_{number:06d}.json", "source_path": str(args.new_data_dir / f"case_{number:06d}.json"),
            "alarm_summary": normalized_alarm(case.get("alarm_name")), "physical_fingerprint_sha256": physical_fingerprint(case),
        })
    (args.output_dir / "added_case_ids.txt").write_text("".join(f"{row['case_id']}\n" for row in added_manifest), encoding="utf-8")
    (args.output_dir / "added_cases_manifest.json").write_text(json.dumps({
        "schema_version": "expanded-added-cases-v1", "matching_policy": "physical-content-sha256-v1",
        "volatile_fields_removed": sorted(VOLATILE_FIELDS), "alarm_normalization": "remove leading 数通设备syslog告警:",
        "added_count": len(added_manifest), "cases": added_manifest,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "expanded_test.jsonl").open("w", encoding="utf-8") as handle:
        for case in expanded_test:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (args.output_dir / "clean_train.jsonl").open("w", encoding="utf-8") as handle:
        for case in clean_train:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (args.output_dir / "clean_expanded_test.jsonl").open("w", encoding="utf-8") as handle:
        for case in expanded_test:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")

    dictionary = dictionary_for("v1")
    train_packs = build_packs(clean_train, source_dataset="expanded-expert-clean-v1:train")
    thresholds = fit_thresholds(clean_train)
    feature_model = fit_feature_model(train_packs, dictionary=dictionary)
    learned_predicates = fit_learned_predicate_model(clean_train)
    comparison_model = fit_comparison_model(clean_train)
    train_features = [
        augment_explainable_features(extract_features(pack, thresholds, feature_model, dictionary=dictionary), case)
        for pack, case in zip(train_packs, clean_train)
    ]
    test_packs = build_packs(expanded_test, source_dataset="expanded-expert-clean-v1:test")
    test_features = [
        augment_explainable_features(extract_features(pack, thresholds, feature_model, dictionary=dictionary), case)
        for pack, case in zip(test_packs, expanded_test)
    ]
    graph = EvidenceGraph.build(
        train_features, labels_of(clean_train), feature_model=feature_model, dictionary=dictionary,
        source_dataset="expanded-expert-clean-v1", confirmed_by="expert-or-original-unreviewed",
    )
    # Keep all 126 historical candidates so the two dimensions can be ranked
    # independently.  ``top_k`` is applied after feature and graph scoring.
    matches = match_many(graph, test_features, top_k=0)
    predictions_path = args.predictions
    if predictions_path is None and args.remote_run_dir is not None:
        predictions_path = args.remote_run_dir / "predictions_expanded.json"
    predictions = prediction_map(predictions_path)
    prediction_rows: list[dict[str, Any]] = []
    if predictions_path is not None and predictions_path.exists():
        prediction_value = json.loads(predictions_path.read_text(encoding="utf-8"))
        prediction_rows = prediction_value if isinstance(prediction_value, list) else prediction_value.get("predictions", [])
    train_features_by_id = {item.case_id: item for item in train_features}
    train_labels = {str(case["case_id"]): str(case.get("label")) for case in clean_train}
    train_cases_by_id = {str(case["case_id"]): case for case in clean_train}
    train_graphs = {
        item.case_id: observable_graph(
            item.tokens, thresholds, feature_model,
            telemetry=case, learned_predicates=learned_predicates,
        )
        for item, case in zip(train_features, clean_train)
    }
    graph_edge_idf = fit_edge_idf(train_graphs.values())

    patterns = []
    case_analysis = []
    train_conflict_ids, test_conflict_ids = set(), set()
    feature_top1_correct = graph_top1_correct = 0
    feature_exact_count = feature_high_count = graph_high_count = 0
    graph_feature_coverages: list[float] = []
    learned_path_counts: list[int] = []
    quadrants = Counter()
    conflict_pair_scores: list[tuple[float, float]] = []
    for case, features, result in zip(expanded_test, test_features, matches):
        query_graph = observable_graph(
            features.tokens, thresholds, feature_model,
            telemetry=case, learned_predicates=learned_predicates,
        )
        graph_feature_coverages.append(query_graph["feature_coverage"])
        learned_path_counts.append(query_graph["learned_range_path_count"])
        feature_ranked = list(result.candidates[: args.top_k])
        graph_rows = []
        candidate_by_id = {item.case_id: item for item in result.candidates}
        for item in result.candidates:
            graph_rows.append((item.case_id, graph_match(query_graph, train_graphs[item.case_id], graph_edge_idf)))
        graph_rows.sort(key=lambda row: (-row[1]["similarity"], row[0]))
        graph_ranked = graph_rows[: args.top_k]
        best_feature = feature_ranked[0] if feature_ranked else None
        best_graph_id, best_graph = graph_ranked[0] if graph_ranked else ("", {"similarity": 0.0})
        if best_feature is not None:
            feature_top1_correct += int(best_feature.label == case.get("label"))
            feature_exact_count += int(best_feature.similarity == 1.0)
            feature_high_count += int(best_feature.similarity >= args.similarity_threshold)
        if best_graph_id:
            graph_top1_correct += int(train_labels[best_graph_id] == case.get("label"))
            graph_high_count += int(best_graph["similarity"] >= args.graph_similarity_threshold)

        pool_ids = {item.case_id for item in feature_ranked} | {case_id for case_id, _ in graph_ranked}
        scored_pool = []
        for case_id in pool_ids:
            feature_candidate = candidate_by_id[case_id]
            graph_score = graph_match(query_graph, train_graphs[case_id], graph_edge_idf)
            feature_high = feature_candidate.similarity >= args.similarity_threshold
            graph_high = graph_score["similarity"] >= args.graph_similarity_threshold
            quadrant = (
                "both_high" if feature_high and graph_high else
                "feature_only" if feature_high else
                "graph_only" if graph_high else "both_low"
            )
            scored_pool.append((feature_candidate, graph_score, quadrant))
        conflicting = [
            row for row in scored_pool
            if row[0].label != case.get("label") and row[2] != "both_low"
        ]
        conflict_pair_scores.extend(
            (row[0].similarity, row[1]["similarity"])
            for row in scored_pool if row[0].label != case.get("label")
        )
        if not conflicting:
            continue
        conflicting.sort(key=lambda row: (
            row[2] != "both_high",
            -min(row[0].similarity, row[1]["similarity"]),
            -(row[0].similarity + row[1]["similarity"]),
            row[0].case_id,
        ))
        best, best_graph_match, best_quadrant = conflicting[0]
        selected = [row for row in conflicting if row[2] == "both_high"][: args.top_k]
        quadrants[best_quadrant] += 1
        if best_quadrant != "both_high":
            # Keep single-dimension conflicts in aggregate diagnostics, but do not
            # expand hundreds of low-priority cards in the human-facing report.
            continue
        # The highest-ranked two-dimensional conflict is the representative pair used
        # by both formula blocks and the raw-data comparison.
        shared = list(best.shared_evidence)
        pattern_id = f"PAT-{len(patterns) + 1:03d}"
        prediction = predictions.get(str(case["case_id"]), {}).get("prediction")
        rows = [{
            "case_id": item.case_id, "split": "train", "label": item.label,
            "original_label": original_labels[item.case_id],
            "label_status": adjudication["cases"].get(item.case_id, {}).get("label_status", "unreviewed"),
            "prediction": None, "feature_similarity": item.similarity,
            "graph_similarity": graph_score["similarity"], "quadrant": quadrant,
            "tokens": list(train_features_by_id[item.case_id].tokens),
        } for item, graph_score, quadrant in selected]
        rows.append({
            "case_id": case["case_id"], "split": "test", "label": case.get("label"),
            "original_label": original_labels[str(case["case_id"])],
            "label_status": adjudication["cases"].get(str(case["case_id"]), {}).get("label_status", "unreviewed"),
            "prediction": prediction, "feature_similarity": best.similarity,
            "graph_similarity": best_graph_match["similarity"], "quadrant": best_quadrant,
            "tokens": list(features.tokens),
        })
        train_conflict_ids.update(item.case_id for item, _, _ in selected)
        test_conflict_ids.add(str(case["case_id"]))
        train_tokens = set(train_features_by_id[best.case_id].tokens)
        query_tokens = set(features.tokens)
        shared_weight = sum(graph.idf.get(token, 1.0) for token in sorted(query_tokens & train_tokens))
        union_weight = sum(graph.idf.get(token, 1.0) for token in sorted(query_tokens | train_tokens))
        physical_logic = [{"token": token, "logic": token_logic(token)} for token in shared]
        shared_relation_criteria = [{
            "path": tuple(item[key] for key in ("side", "measurement", "predicate", "symptom", "layer")),
            "source_tokens": (item["token"],),
            "criteria": (item["criterion"],),
            "predicate_type": item.get("predicate_type", "learned_range" if item.get("learned") else "feature_projection"),
            "provenance": item.get("provenance", "clean-train-supervised" if item.get("learned") else "feature-token"),
            "quantifier": item.get("quantifier", "range branch" if item.get("learned") else "token-defined"),
        } for item in best_graph_match["shared_predicate_paths"]]
        receiving_ambiguity = any(
            token.startswith(("drop:", "status:", "level:"))
            and any(name in token for name in ("rxpower", "media_snr", "RxLOS", "RxLOL"))
            for token in shared
        )
        physical_summary = "；".join(item["logic"] for item in physical_logic[:2]) or "当前共享 token 为空，不能形成物理归因链。"
        boundary = (
            "共享证据主要描述接收方向症状：可确认信号到达/解调异常，但无法仅凭快照区分对端发射、fiber 与本端接收器。"
            if receiving_ambiguity else
            "共享证据能确认异常层次或症状端，但没有形成排他性的 L1/L2/fiber 归因条件。"
        )
        case_differences = compare_case_features(train_cases_by_id[best.case_id], case, comparison_model)
        exact_quality_compatible = quality_compatible(train_cases_by_id[best.case_id], case)
        critical_conflicts = tuple(sorted(set(query_graph["quality_state"]["missing_measurements"]) ^ set(train_graphs[best.case_id]["quality_state"]["missing_measurements"])))
        exact_two_dimensional = (
            best.similarity == 1.0 and best_graph_match["similarity"] == 1.0
            and exact_quality_compatible and not critical_conflicts
        )
        if exact_two_dimensional and case_differences["counts"]["large"] == 0:
            review_priority = "critical"
            review_reason = "两个相似度都为 1，且没有大幅测量差异：优先核查标签。"
        elif case_differences["counts"]["large"] == 0 and case_differences["counts"]["medium"] == 0:
            review_priority = "high"
            review_reason = "没有达到中/大差异门槛的测量特征；现有差别不足以直接解释标签冲突。"
        elif min(best.similarity, best_graph_match["similarity"]) >= 0.9 and case_differences["counts"]["large"] <= 1:
            review_priority = "high"
            review_reason = "两个维度高度相似，显著差异很少：标签冲突较难由现有特征解释。"
        else:
            review_priority = "standard"
            review_reason = "存在可见测量差异；专家需判断这些差异是否足以支持不同标签。"
        pattern = {
            "pattern_id": pattern_id, "summary": f"{case['case_id']} 与历史 {','.join(item.case_id for item, _, _ in selected)} 高相似但标签冲突",
            "feature_similarity": best.similarity, "graph_similarity": best_graph_match["similarity"],
            "quadrant": best_quadrant, "shared_evidence": shared,
            "query_only_evidence": list(best.extra_evidence), "train_only_evidence": list(best.missing_evidence),
            "shared_weight": round(shared_weight, 8), "union_weight": round(union_weight, 8),
            "weighted_shared_terms": [{"token": token, "idf": graph.idf.get(token, 1.0), "meaning": token_logic(token)} for token in shared],
            "physical_logic": physical_logic, "physical_summary": physical_summary,
            "query_graph": query_graph, "train_graph": train_graphs[best.case_id], "graph_match": best_graph_match,
            "shared_relation_criteria": shared_relation_criteria,
            "feature_differences": case_differences,
            "exact_two_dimensional_match": exact_two_dimensional,
            "quality_compatible": exact_quality_compatible,
            "critical_evidence_conflicts": critical_conflicts,
            "review_priority": review_priority,
            "exact_two_dimensional_match": exact_two_dimensional,
            "quality_compatible": exact_quality_compatible,
            "critical_evidence_conflicts": list(critical_conflicts),
            "review_reason": review_reason,
            "confirmed_path_status": "unavailable",
            "identifiability_boundary": boundary, "train_labels": sorted({item.label for item, _, _ in selected}), "test_label": case.get("label"),
            "why_same_pattern": f"特征 IDF-Jaccard={best.similarity:.3f}；可观测证据子图相似度={best_graph_match['similarity']:.3f}。两者独立展示，不再把 token coverage 冒充图结构相似度。",
            "label_conflict_analysis": f"训练标签={sorted({item.label for item, _, _ in selected})}，测试标签={case.get('label')}。现有快照不能证明哪一方标注正确。",
            "impact": "历史复用可能把训练标签直接传播到测试 case；若模型预测与历史标签一致但与测试真值不同，则属于被相似历史模式误导的强候选。",
            "cases": rows,
        }
        patterns.append(pattern)
        case_analysis.append({
            "case_id": case["case_id"], "actual_label": case.get("label"), "prediction": prediction,
            "pattern_id": pattern_id, "feature_similarity": best.similarity,
            "graph_similarity": best_graph_match["similarity"], "quadrant": best_quadrant,
            "review_priority": review_priority,
            "feature_difference_counts": case_differences["counts"],
            "largest_feature_difference": case_differences["largest"],
            "feature_match": result.to_dict(), "observable_graph_match": best_graph_match,
            "confirmed_path_status": "unavailable",
            "failure_step": "N3 two-dimensional historical pattern label conflict",
            "cause": "data_non_identifiable_or_label_suspect" if exact_two_dimensional else "two_dimensional_pattern_conflict",
        })

    patterns.sort(key=lambda item: (
        {"critical": 0, "high": 1, "standard": 2}[item["review_priority"]],
        item["feature_differences"]["counts"]["large"],
        -min(item["feature_similarity"], item["graph_similarity"]),
        item["pattern_id"],
    ))
    annotation_template = {
        "schema_version": "rca-expert-label-review-v1",
        "experiment_id": "20260816_expanded-pattern-conflict",
        "annotations": [{
            "pattern_id": pattern["pattern_id"],
            "left_case_id": next(row["case_id"] for row in pattern["cases"] if row["split"] == "train"),
            "right_case_id": next(row["case_id"] for row in pattern["cases"] if row["split"] == "test"),
            "review_priority": pattern["review_priority"],
            "decision": "", "left_label": "keep", "right_label": "keep",
            "evidence_status": "", "notes": "", "completed": False,
        } for pattern in patterns],
    }
    label_suspects = [row for row in case_analysis if row.get("exact_two_dimensional_match")]
    missing_rows = [{"case_id": case["case_id"], "label": case.get("label"), "source_file": next((p.name for p in args.old_data_dir.glob("case_*.json") if json.loads(p.read_text(encoding="utf-8")).get("case_id") == case["case_id"]), "unknown")} for case in missing_old]
    remote_loaded = bool(predictions)
    original_full_test = old_test + added
    original_full_labels = {str(case["case_id"]): str(case.get("label")) for case in original_full_test}
    original_clean_labels = {str(case["case_id"]): str(case.get("label")) for case in expanded_test_raw}
    adjudicated_clean_labels = {str(case["case_id"]): str(case.get("label")) for case in expanded_test}
    reviewed_clean_ids = set(adjudication["cases"]) & set(adjudicated_clean_labels)
    remote_metrics = {
        "original_343": evaluate_prediction_rows(prediction_rows, original_full_labels),
        "clean_original": evaluate_prediction_rows(prediction_rows, original_clean_labels),
        "clean_partially_adjudicated": evaluate_prediction_rows(prediction_rows, adjudicated_clean_labels),
        "expert_reviewed_clean_subset": evaluate_prediction_rows(
            prediction_rows, adjudicated_clean_labels, include=reviewed_clean_ids,
        ),
    } if prediction_rows else {}
    if args.deterministic_run_dir is not None:
        deterministic_path = args.deterministic_run_dir / "predictions.json"
        if deterministic_path.exists():
            deterministic_value = json.loads(deterministic_path.read_text(encoding="utf-8"))
            deterministic_rows = deterministic_value if isinstance(deterministic_value, list) else deterministic_value.get("predictions", [])
            remote_metrics["clean_deterministic_rebuilt"] = evaluate_prediction_rows(
                deterministic_rows, adjudicated_clean_labels,
            )

    split_by_case = {
        **{str(case["case_id"]): "train" for case in clean_train_raw},
        **{str(case["case_id"]): "test" for case in expanded_test_raw},
    }
    case_records: list[dict[str, Any]] = []
    for case in clean_train_raw + expanded_test_raw:
        case_id = str(case["case_id"])
        review = adjudication["cases"].get(case_id)
        case_records.append({
            "case_id": case_id,
            "split": split_by_case[case_id],
            "included": True,
            "original_label": str(case.get("label")),
            "adjudicated_label": review["adjudicated_label"] if review else str(case.get("label")),
            "label_status": review["label_status"] if review else "unreviewed",
            "expert_decisions": review["expert_decisions"] if review else [],
            "evidence_statuses": review["evidence_statuses"] if review else [],
            "notes": review["notes"] if review else [],
            "reviewed_at": review["reviewed_at"] if review else None,
            "review_source": review["review_source"] if review else None,
            "requires_secondary_physics_review": bool(review and review["requires_secondary_physics_review"]),
        })
    for case in missing_old:
        case_id = str(case["case_id"])
        review = adjudication["cases"].get(case_id)
        case_records.append({
            "case_id": case_id,
            "split": "excluded_train" if case in old_train else "excluded_test",
            "included": False,
            "exclusion_reason": "absent_from_new_dataset_and_optical_blackout",
            "original_label": str(case.get("label")),
            "adjudicated_label": review["adjudicated_label"] if review else str(case.get("label")),
            "label_status": "excluded_low_quality",
            "expert_decisions": review["expert_decisions"] if review else [],
            "evidence_statuses": review["evidence_statuses"] if review else [],
            "notes": review["notes"] if review else [],
            "reviewed_at": review["reviewed_at"] if review else None,
            "review_source": review["review_source"] if review else None,
            "requires_secondary_physics_review": bool(review and review["requires_secondary_physics_review"]),
        })
    data_contract = {
        "schema_version": "expanded-expert-clean-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_old_dataset": str(args.old_data_dir),
        "source_new_dataset": str(args.new_data_dir),
        "matching_policy": "physical-content-sha256-v1",
        "cleanup_policy": "remove all old cases absent from new dataset" if args.clean_missing_old else "retain missing old cases",
        "train_size": len(clean_train),
        "test_size": len(expanded_test),
        "excluded_case_count": len(missing_old) if args.clean_missing_old else 0,
        "expert_annotation_source": str(args.expert_annotations) if args.expert_annotations else None,
        "expert_reviewed_case_count": adjudication["case_count"],
        "expert_changed_case_count": adjudication["changed_count"],
        "reviewed_test_case_count": len(reviewed_clean_ids),
        "unreviewed_test_case_count": len(expanded_test) - len(reviewed_clean_ids),
        "secondary_physics_review_pair_count": adjudication["secondary_physics_review_pair_count"],
        "label_policy": "expert label when reviewed; original label retained but marked unreviewed otherwise",
        "n8_feedback_update": False,
        "cases": sorted(case_records, key=lambda item: (item["split"], item["case_id"])),
    }
    threshold_sensitivity = [
        {
            "threshold": threshold,
            "both_high_different_label_pairs": sum(
                feature >= threshold and graph_score >= threshold
                for feature, graph_score in conflict_pair_scores
            ),
        }
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ]
    summary = {
        "schema_version": "expanded-pattern-summary-v5-expert-clean", "old_data_dir": str(args.old_data_dir), "new_data_dir": str(args.new_data_dir),
        "old_train_size": len(old_train), "old_test_size": len(old_test), "clean_train_size": len(clean_train),
        "retained_old_count": sum(len(old_fps[fp]) for fp in retained_fps),
        "missing_old_count": len(missing_old), "excluded_old_count": len(missing_old) if args.clean_missing_old else 0,
        "added_count": len(added), "expanded_test_size": len(expanded_test),
        "data_contract_version": data_contract["schema_version"],
        "expert_reviewed_case_count": adjudication["case_count"],
        "expert_changed_case_count": adjudication["changed_count"],
        "reviewed_clean_test_count": len(reviewed_clean_ids),
        "unreviewed_clean_test_count": len(expanded_test) - len(reviewed_clean_ids),
        "secondary_physics_review_pair_count": adjudication["secondary_physics_review_pair_count"],
        "feature_similarity_threshold": args.similarity_threshold,
        "graph_similarity_threshold": args.graph_similarity_threshold,
        "top_k_per_dimension": args.top_k, "pattern_count": len(patterns),
        "all_high_in_at_least_one_dimension_count": sum(quadrants.values()),
        "listed_pattern_policy": "both_high_only",
        "train_cases_in_conflicts": len(train_conflict_ids), "test_cases_in_conflicts": len(test_conflict_ids),
        "dual_exact_label_conflicts": len(label_suspects), "evidence_graph_version": graph.version,
        "observable_graph_version": "observable-evidence-subgraph-v4-quality-physical-relation-learned",
        "evidence_state_version": EVIDENCE_STATE_VERSION,
        "learned_predicate_model_version": learned_predicates["version"],
        "learned_predicate_candidates": learned_predicates["candidate_count"],
        "learned_predicate_accepted": learned_predicates["accepted_count"],
        "review_priority_counts": dict(Counter(pattern["review_priority"] for pattern in patterns)),
        "difference_ranking": "absolute delta / clean-adjudicated-train IQR scale; large>=2.0, medium>=0.75, small<0.75",
        "feature_dictionary_version": dictionary.version, "feature_dictionary_hash": dictionary.content_hash(),
        "feature_evaluation": {
            "top1_label_agreement": round(feature_top1_correct / len(expanded_test), 6),
            "exact_top1_count": feature_exact_count, "high_top1_count": feature_high_count,
            "formula": "IDF-weighted Jaccard over v1 tokens plus label-free Q0/P/R explainable tokens",
            "feature_space_version": "expanded-explainable-features-v1",
        },
        "graph_evaluation": {
            "top1_label_agreement": round(graph_top1_correct / len(expanded_test), 6),
            "high_top1_count": graph_high_count,
            "average_feature_to_graph_coverage": round(sum(graph_feature_coverages) / len(graph_feature_coverages), 6),
            "average_learned_range_paths": round(sum(learned_path_counts) / len(learned_path_counts), 6),
            "formula": "IDF-weighted Jaccard over typed edges; nodes are explanation-only",
            "schema": "side -> measurement -> predicate -> symptom -> physical-layer",
            "predicate_policy": "Q0 data quality + P physical boundaries + R side-level relations + L supervised stable ranges learned on clean adjudicated train",
            "confirmed_training_paths_available": len(graph.case_diagnoses),
            "confirmed_training_paths_total": len(graph.cases),
        },
        "conflict_quadrants": {key: quadrants[key] for key in ("both_high", "feature_only", "graph_only", "both_low")},
        "threshold_sensitivity": threshold_sensitivity,
        "remote_metrics": remote_metrics,
        "remote_predictions_loaded": remote_loaded, "remote_status": "已加载远端预测结果。" if remote_loaded else "远端实验尚未运行或结果尚未回传；预测字段明确留空。",
    }
    focus_pairs = []
    all_source_cases = {str(case["case_id"]): case for case in old_cases + new_cases}
    for spec in args.focus_pair:
        try:
            left_id, right_id = (item.strip() for item in spec.split(",", 1))
        except ValueError as exc:
            raise ValueError(f"invalid --focus-pair {spec!r}; expected TRAIN_ID,CASE_ID") from exc
        if left_id not in train_features_by_id or right_id not in all_source_cases:
            raise ValueError(f"focus pair cannot be resolved: {left_id}, {right_id}")
        right_case = all_source_cases[right_id]
        right_pack = build_packs([right_case], source_dataset="focus-pair-v1")[0]
        right_features = extract_features(right_pack, thresholds, feature_model, dictionary=dictionary)
        left_features = train_features_by_id[left_id]
        left_graph, right_graph = train_graphs[left_id], observable_graph(
            right_features.tokens, thresholds, feature_model,
            telemetry=right_case, learned_predicates=learned_predicates,
        )
        graph_detail = graph_match(right_graph, left_graph, graph_edge_idf)
        focus_pairs.append({
            "left_case_id": left_id, "right_case_id": right_id,
            "feature_similarity": weighted_jaccard(set(left_features.tokens), set(right_features.tokens), graph.idf),
            "graph_similarity": graph_detail["similarity"],
            "left_graph_coverage": left_graph["feature_coverage"],
            "right_graph_coverage": right_graph["feature_coverage"],
            "interpretation": (
                "S_feature 比较 v1 token；S_graph 另行比较物理离散谓词与训练监督学习范围。"
                "即使 Q25/Q75 token 相同，学习范围或离散证据不同，图分数也不能按特征分数推断。"
            ),
        })
    record_by_case = {row["case_id"]: row for row in case_records}
    all_raw_cases = {
        str(case["case_id"]): {
            "label": case.get("label"), "split": "train", "label_audit": record_by_case[str(case["case_id"])], "raw": case,
        }
        for case in clean_train
    }
    all_raw_cases.update({
        str(case["case_id"]): {
            "label": case.get("label"), "split": "test", "label_audit": record_by_case[str(case["case_id"])], "raw": case,
        }
        for case in expanded_test
    })
    for case in new_cases:
        all_raw_cases.setdefault(str(case["case_id"]), {"label": case.get("label"), "split": "new-dataset", "raw": case})
    referenced_ids = {str(row["case_id"]) for pattern in patterns for row in pattern["cases"]}
    referenced_ids.update(item[key] for item in focus_pairs for key in ("left_case_id", "right_case_id"))
    raw_cases = {case_id: all_raw_cases[case_id] for case_id in sorted(referenced_ids)}
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(), "summary": summary,
        "learned_predicates": learned_predicates, "added_cases": added_manifest,
        "missing_old_cases": missing_rows, "patterns": patterns, "focus_pairs": focus_pairs,
        "raw_cases": raw_cases, "data_contract": data_contract, "adjudication": adjudication,
    }
    for name, value in (
        ("summary.json", summary), ("data_contract.json", data_contract),
        ("expert_adjudications.json", adjudication),
        ("secondary_physics_review_pairs.json", [row for row in adjudication["pairs"] if row["requires_secondary_physics_review"]]),
        ("learned_predicate_model.json", learned_predicates),
        ("expert_annotation_template.json", annotation_template), ("case_analysis.json", case_analysis),
        ("bad_cases.json", case_analysis), ("label_suspects.json", label_suspects),
        ("irreducible_cases.json", label_suspects),
    ):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render_html(payload)
    (args.output_dir / "expanded_rca_pattern_analysis.html").write_text(report, encoding="utf-8")
    (args.output_dir / "report.html").write_text(report, encoding="utf-8")
    dataset_note = {
        "old_train_path": str(args.old_data_dir), "old_train_range": f"first {len(old_train)} positional cases; cleaned to {len(clean_train)}",
        "old_test_path": str(args.old_data_dir), "old_test_range": f"remaining {len(old_test)} positional cases",
        "new_dataset_path": str(args.new_data_dir), "added_count": len(added), "expanded_test_path": str(args.output_dir / "clean_expanded_test.jsonl"),
        "clean_train_path": str(args.output_dir / "clean_train.jsonl"),
        "data_contract_path": str(args.output_dir / "data_contract.json"),
        "deduplication_basis": "physical-content-sha256-v1",
        "cleanup": f"removed {len(missing_old) if args.clean_missing_old else 0} old cases absent from new dataset",
        "warning": f"new dataset is not a strict superset: {len(missing_old)} old cases were not found",
    }
    (args.output_dir / "DATASET_EXPANSION.md").write_text("# Expanded dataset contract\n\n```json\n" + json.dumps(dataset_note, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
