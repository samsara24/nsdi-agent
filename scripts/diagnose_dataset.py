#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rca_framework.data import canonical_label, side_mapping  # noqa: E402


ROOT_CAUSES = ("L1", "L2", "fiber")
METRICS = ("rxpower", "txpower", "media_snr", "host_snr", "serdes_snr", "bias")


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_lanes(case: dict[str, Any], metric: str, raw_side: str) -> dict[str, float]:
    block = case.get(metric) or {}
    side_block = block.get(raw_side) if isinstance(block, dict) else None
    if not isinstance(side_block, dict):
        return {}
    values: dict[str, float] = {}
    for lane, value in side_block.items():
        parsed = safe_float(value)
        if parsed is not None:
            values[str(lane)] = parsed
    return values


def scalar_side(case: dict[str, Any], key: str, raw_side: str) -> float | None:
    block = case.get(key) or {}
    if not isinstance(block, dict):
        return None
    return safe_float(block.get(raw_side))


def bad_rx_lanes(case: dict[str, Any], raw_side: str) -> set[str]:
    rx = metric_lanes(case, "rxpower", raw_side)
    snr = metric_lanes(case, "media_snr", raw_side)
    return {lane for lane, value in rx.items() if value <= -39.0} | {
        lane for lane, value in snr.items() if value <= 0.0
    }


def load_raw_cases(organized_data: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(organized_data.glob("*/*.json"), key=lambda item: item.relative_to(organized_data).as_posix()):
        case = read_json(path)
        mapping = side_mapping(case)
        if not mapping:
            continue
        label = canonical_label(case.get("label"), mapping)
        if label not in ROOT_CAUSES:
            continue
        rows.append({"path": path, "case": case, "mapping": mapping, "label": label})
    return rows


def median(values: Iterable[float | None]) -> float | None:
    clean = sorted(value for value in values if value is not None)
    return stats.median(clean) if clean else None


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def print_label_distribution(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["label"] for row in rows)
    print("## Label Distribution")
    print(f"valid_cases={len(rows)}")
    for label in ROOT_CAUSES:
        print(f"{label}={counts[label]}")
    print()


def analyze_predictions(predictions_path: Path) -> None:
    if not predictions_path.exists():
        print(f"## Prediction Artifact\nmissing={predictions_path}\n")
        return
    predictions = read_json(predictions_path)
    print("## Prediction Artifact")
    print(f"path={predictions_path.relative_to(ROOT) if predictions_path.is_relative_to(ROOT) else predictions_path}")
    print(f"case_count={len(predictions)}")
    actual = Counter(row.get("actual_label") for row in predictions)
    majority = max(actual.values()) if actual else 0
    print(f"majority_baseline={majority}/{len(predictions)} ({pct(majority / len(predictions) if predictions else None)})")
    correct = sum(1 for row in predictions if row.get("correct"))
    print(f"artifact_accuracy={correct}/{len(predictions)} ({pct(correct / len(predictions) if predictions else None)})")

    zero = [row for row in predictions if not row.get("extracted_anomalies")]
    print(f"zero_anomaly_cases={len(zero)}/{len(predictions)} ({pct(len(zero) / len(predictions) if predictions else None)})")
    print(f"zero_anomaly_actual={dict(Counter(row.get('actual_label') for row in zero))}")
    print(f"zero_anomaly_prediction={dict(Counter(row.get('prediction') for row in zero))}")
    print(f"zero_anomaly_correct={sum(1 for row in zero if row.get('correct'))}")

    anomaly_counts: Counter[str] = Counter()
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for row in predictions:
        label = row.get("actual_label")
        for item in row.get("extracted_anomalies", []):
            anomaly_id = item.get("anomaly_id")
            if anomaly_id:
                anomaly_counts[anomaly_id] += 1
                by_label[label][anomaly_id] += 1
    print("top_anomalies=")
    for anomaly_id, count in anomaly_counts.most_common(20):
        print(f"  {anomaly_id}: {count}")
    print()


def analyze_unused_fields(rows: list[dict[str, Any]]) -> None:
    print("## Unused Field Separability")
    numeric_keys = (
        "L1_bias_max",
        "L2_bias_max",
        "L1_bias_spread",
        "L2_bias_spread",
        "L1_temp",
        "L2_temp",
        "L1_volt",
        "L2_volt",
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        case, mapping, label = row["case"], row["mapping"], row["label"]
        record: dict[str, Any] = {"label": label}
        for side in ("L1", "L2"):
            raw_side = mapping[side]
            bias = list(metric_lanes(case, "bias", raw_side).values())
            record[f"{side}_bias_max"] = max(bias) if bias else None
            record[f"{side}_bias_spread"] = max(bias) - min(bias) if len(bias) >= 2 else None
            record[f"{side}_temp"] = scalar_side(case, "Temperature", raw_side)
            record[f"{side}_volt"] = scalar_side(case, "Voltage", raw_side)
        records.append(record)

    for key in numeric_keys:
        cells = []
        for label in ROOT_CAUSES:
            sub = [record[key] for record in records if record["label"] == label]
            cells.append(f"{label}:median={median(sub)}")
        print(f"{key}: " + " | ".join(cells))
    print()


def analyze_directional_signatures(rows: list[dict[str, Any]]) -> None:
    print("## Directional and Lane Signatures")
    lane_pattern_counts: dict[str, Counter[str]] = defaultdict(Counter)
    txok_rxdown_counts: Counter[str] = Counter()
    txok_rxdown_total: Counter[str] = Counter()

    for row in rows:
        case, mapping, label = row["case"], row["mapping"], row["label"]
        bad_l1 = bad_rx_lanes(case, mapping["L1"])
        bad_l2 = bad_rx_lanes(case, mapping["L2"])
        if not bad_l1 and not bad_l2:
            pattern = "none"
        elif bad_l1 and not bad_l2:
            pattern = "onlyL1rx"
        elif bad_l2 and not bad_l1:
            pattern = "onlyL2rx"
        elif bad_l1 == bad_l2:
            pattern = "both_same_lane"
        elif bad_l1 & bad_l2:
            pattern = "both_partial"
        else:
            pattern = "both_disjoint"
        lane_pattern_counts[pattern][label] += 1

        any_txok_rxdown = False
        for src, dst in (("L1", "L2"), ("L2", "L1")):
            tx = metric_lanes(case, "txpower", mapping[src])
            rx = metric_lanes(case, "rxpower", mapping[dst])
            common = set(tx) & set(rx)
            if any(tx[lane] > -39.0 and rx[lane] <= -39.0 for lane in common):
                any_txok_rxdown = True
        txok_rxdown_total[label] += 1
        if any_txok_rxdown:
            txok_rxdown_counts[label] += 1

    print("lane_pattern_distribution=")
    for pattern, counts in sorted(lane_pattern_counts.items(), key=lambda item: -sum(item[1].values())):
        total = sum(counts.values())
        fiber_rate = counts["fiber"] / total if total else 0.0
        print(f"  {pattern}: n={total}, L1={counts['L1']}, L2={counts['L2']}, fiber={counts['fiber']}, fiber_rate={fiber_rate:.3f}")
    print("txok_rxdown_signature=")
    for label in ROOT_CAUSES:
        total = txok_rxdown_total[label]
        hit = txok_rxdown_counts[label]
        print(f"  {label}: {hit}/{total} ({pct(hit / total if total else None)})")
    print()


def supervised_feature_vector(row: dict[str, Any]) -> list[float]:
    case, mapping = row["case"], row["mapping"]
    features: list[float] = []
    for side in ("L1", "L2"):
        raw_side = mapping[side]
        for metric in METRICS:
            values = list(metric_lanes(case, metric, raw_side).values())
            if values:
                features.extend(
                    [
                        min(values),
                        max(values),
                        sum(values) / len(values),
                        max(values) - min(values),
                        float(sum(1 for value in values if value <= -39.0 or (metric.endswith("snr") and value <= 0.0))),
                    ]
                )
            else:
                features.extend([-999.0] * 5)
        for status in ("RxLOS", "RxLOL", "TxLOS", "TxLOL"):
            block = case.get(status) or {}
            value = str(block.get(raw_side) if isinstance(block, dict) else "").strip().lower()
            features.append(1.0 if value in {"abnormal", "down", "los", "lol", "true", "1"} else 0.0)
        for scalar in ("Temperature", "Voltage"):
            features.append(scalar_side(case, scalar, raw_side) or -999.0)
    for source, target in (("L2", "L1"), ("L1", "L2")):
        tx = metric_lanes(case, "txpower", mapping[source])
        rx = metric_lanes(case, "rxpower", mapping[target])
        losses = [tx[lane] - rx[lane] for lane in sorted(set(tx) & set(rx))]
        features.extend([max(losses), sum(losses) / len(losses)] if losses else [-999.0, -999.0])
    return features


def analyze_supervised_ceiling(rows: list[dict[str, Any]], skip: bool) -> None:
    print("## Supervised Feature-Space Ceiling")
    if skip:
        print("skipped=true\n")
        return
    try:
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    except ImportError as exc:
        print(f"skipped=missing_dependency ({exc})\n")
        return

    x = np.array([supervised_feature_vector(row) for row in rows], dtype=float)
    y = np.array([row["label"] for row in rows])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    dummy = cross_val_score(DummyClassifier(strategy="most_frequent"), x, y, cv=cv, scoring="accuracy")
    forest = RandomForestClassifier(n_estimators=300, random_state=42)
    scores = cross_val_score(forest, x, y, cv=cv, scoring="accuracy")
    pred = cross_val_predict(forest, x, y, cv=cv)
    print(f"feature_shape={x.shape}")
    print(f"majority_cv_accuracy={dummy.mean():.4f} (+/- {dummy.std():.4f})")
    print(f"random_forest_cv_accuracy={scores.mean():.4f} (+/- {scores.std():.4f})")
    print("random_forest_report=")
    print(classification_report(y, pred, digits=3, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the RCA v2 dataset and artifact failure modes.")
    parser.add_argument("--organized-data", type=Path, default=ROOT / "organized_data")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "artifacts/organized_rca_v2_60_40_seed42_deepseek32b_vllm/predictions.json",
    )
    parser.add_argument("--skip-supervised", action="store_true")
    args = parser.parse_args()

    organized_data = args.organized_data if args.organized_data.is_absolute() else ROOT / args.organized_data
    predictions = args.predictions if args.predictions.is_absolute() else ROOT / args.predictions
    rows = load_raw_cases(organized_data)

    print_label_distribution(rows)
    analyze_predictions(predictions)
    analyze_unused_fields(rows)
    analyze_directional_signatures(rows)
    analyze_supervised_ceiling(rows, args.skip_supervised)


if __name__ == "__main__":
    main()
