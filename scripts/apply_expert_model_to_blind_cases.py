#!/usr/bin/env python3
"""Apply the fixed expert-model document to every label-free blind telemetry packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/current_model_case_review_v1"
BLIND = ARTIFACT / "blind_packets"
PROTOCOL_PATH = ARTIFACT / "expert_knowledge_protocol.json"
OUTPUT = ARTIFACT / "expert_augmented_predictions.json"
FREEZE = ARTIFACT / "expert_augmented_freeze.json"
SPLITS = ("all_data", "rule1_channel_not_4")
LEVEL = {"lane_down": 0, "low_value": 1, "high_value": 1, "lane_diff": 2}


def finite_values(value: Any) -> list[float]:
    if not isinstance(value, dict):
        return []
    return [float(item) for item in value.values() if isinstance(item, (int, float))]


def side_metric(packet: dict, metric: str, side: str) -> Any:
    value = packet.get(metric)
    return value.get(side) if isinstance(value, dict) else None


def detect(values: list[float], threshold: dict[str, float]) -> dict | None:
    if not values:
        return None
    if any(value == threshold["down"] for value in values):
        kind = "lane_down"
    elif any(value < threshold["low"] for value in values):
        kind = "low_value"
    elif any(value > threshold["high"] for value in values):
        kind = "high_value"
    elif max(values) - min(values) > threshold["lane_diff"]:
        kind = "lane_diff"
    else:
        return None
    return {"type": kind, "level": LEVEL[kind], "min": min(values), "max": max(values)}


def opposite(side: str) -> str:
    return "L2" if side == "L1" else "L1"


def side_result(side: str, anomalies: dict[str, dict]) -> dict | None:
    candidates = []
    tx = anomalies.get("txpower")
    if tx and tx["type"] == "lane_down":
        candidates.append({"rule": "txpower_lane_down", "fault_location": side, "priority": 0})
    if all(metric in anomalies for metric in ("serdes_snr", "media_snr", "rxpower")):
        candidates.append({"rule": "serdes_media_rx_combination", "fault_location": opposite(side), "priority": 1})
    settings = (
        ("host_snr", 2, side),
        ("serdes_snr", 3, side),
        ("media_snr", 4, opposite(side)),
        ("rxpower", 5, opposite(side)),
        ("txpower", 6, side),
    )
    for metric, prefix, location in settings:
        if metric in anomalies:
            candidates.append({
                "rule": metric,
                "fault_location": location,
                "priority": int(f"{prefix}{anomalies[metric]['level']}"),
            })
    return min(candidates, key=lambda item: item["priority"]) if candidates else None


def port_up(packet: dict, side: str) -> bool:
    tx = finite_values(side_metric(packet, "txpower", side))
    rx = finite_values(side_metric(packet, "rxpower", side))
    tx_ok = bool(tx) and any(value > -40 for value in tx)
    rx_ok = bool(rx) and any(value > -40 for value in rx)
    return tx_ok or rx_ok


def analyze(packet: dict, dataset: str, thresholds: dict) -> dict:
    statuses = {side: port_up(packet, side) for side in ("L1", "L2")}
    if not statuses["L1"] and statuses["L2"]:
        verdict, priority, rule = "L1", -1, "port_status_L1_down"
        reasoning = "专家入口判定L1收发光均无有效lane，而L2仍可用，直接定界L1。"
        side_payload = {}
    elif statuses["L1"] and not statuses["L2"]:
        verdict, priority, rule = "L2", -1, "port_status_L2_down"
        reasoning = "专家入口判定L2收发光均无有效lane，而L1仍可用，直接定界L2。"
        side_payload = {}
    elif not statuses["L1"] and not statuses["L2"]:
        verdict, priority, rule = "L1", -1, "port_status_both_down_default_L1"
        reasoning = "专家入口判定双端收发光均无有效lane；按文档兜底优先检查本端L1。"
        side_payload = {}
    else:
        side_payload = {}
        selected = []
        for side in ("L1", "L2"):
            anomalies = {}
            for metric, threshold in thresholds.items():
                values = finite_values(side_metric(packet, metric, side))
                anomaly = detect(values, threshold)
                if anomaly:
                    anomalies[metric] = anomaly
            result = side_result(side, anomalies)
            side_payload[side] = {"anomalies": anomalies, "selected_rule": result}
            if result:
                selected.append({"side": side, **result})
        if not selected:
            verdict, priority, rule = "L1", 8, "no_anomaly_default_L1"
            reasoning = "专家阈值未检出两端异常；按文档无异常兜底返回本端L1，此结论带结构性L1偏置。"
        elif len(selected) == 2 and selected[0]["fault_location"] != selected[1]["fault_location"] and selected[0]["priority"] == selected[1]["priority"]:
            verdict, priority, rule = "fiber", 7, "same_priority_opposite_locations"
            reasoning = f"两端最高规则优先级同为{selected[0]['priority']}但定界相反，按专家双端裁决判fiber。"
        else:
            winner = min(selected, key=lambda item: item["priority"])
            verdict, priority, rule = winner["fault_location"], winner["priority"], winner["rule"]
            anomaly_text = ", ".join(f"{k}:{v['type']}" for k, v in side_payload[winner["side"]]["anomalies"].items())
            reasoning = f"{winner['side']}侧命中最高优先级规则{rule}（priority={priority}；{anomaly_text}），按规则定界{verdict}。"
    return {
        "dataset": dataset,
        "case_id": packet["case_id"],
        "verdict": verdict,
        "priority": priority,
        "rule": rule,
        "reasoning": reasoning,
        "port_status": statuses,
        "side_analysis": side_payload,
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    thresholds = protocol["thresholds"]
    rows = []
    for dataset in SPLITS:
        packets = json.loads((BLIND / f"{dataset}.json").read_text(encoding="utf-8"))
        rows.extend(analyze(packet, dataset, thresholds) for packet in packets)
    OUTPUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    freeze = {
        "protocol": protocol["protocol"],
        "source_protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "prediction_file": str(OUTPUT.relative_to(ROOT)),
        "sha256": digest,
        "case_count": len(rows),
        "label_fields_read": False,
        "dataset_counts": {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in SPLITS},
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False))


if __name__ == "__main__":
    main()
