"""Label-free physical evidence states for the expanded RCA audit.

This module deliberately does not participate in the legacy classifier.  It turns
raw telemetry into auditable quality, physical-state, and cross-end relation paths
used by the expanded evidence-graph report.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from rca_framework.anomaly import METRIC_ALIASES, metric_values


EVIDENCE_STATE_VERSION = "expanded-evidence-state-v1"
OPTICAL_SENTINEL = -40.0
OPTICAL_DROP_BOUNDARY = -39.0
SNR_INVALID_BOUNDARY = 0.0
SERDES_INVALID_BOUNDARY = 1.0
SENTINEL_TOLERANCE = 1e-6


def is_exact_sentinel(
    value: float, sentinel: float = OPTICAL_SENTINEL, *, tolerance: float = SENTINEL_TOLERANCE,
) -> bool:
    """Return whether a floating-point reading is the declared exact sentinel."""
    return math.isclose(float(value), float(sentinel), rel_tol=0.0, abs_tol=tolerance)


def lane_bucket(count: int, observed: int) -> str:
    if count <= 0 or observed <= 0:
        return "none"
    if count == observed:
        return "all_lanes"
    if count == 1:
        return "single_lane"
    return "partial_lanes"


def measurement_state(metric: str, values: Iterable[float]) -> dict[str, Any]:
    """Classify one metric without assigning a root cause.

    Exact sentinels are kept separate from engineering abnormal ranges.  In
    particular, SerDes uses its observed failure state ``<=1`` rather than the old
    generic ``<=0`` drop boundary.
    """
    readings = tuple(float(value) for value in values)
    if metric not in METRIC_ALIASES:
        raise KeyError(f"unknown metric: {metric}")
    if not readings:
        return {
            "metric": metric,
            "observed": 0,
            "quality": "missing",
            "boundary_count": 0,
            "sentinel_count": 0,
            "bucket": "none",
        }

    if metric in {"txpower", "rxpower"}:
        sentinel_count = sum(is_exact_sentinel(value, OPTICAL_SENTINEL) for value in readings)
        boundary_count = sum(value <= OPTICAL_DROP_BOUNDARY for value in readings)
        boundary = OPTICAL_DROP_BOUNDARY
        sentinel = OPTICAL_SENTINEL
        predicate_type = "physical_sentinel" if sentinel_count else "engineering_range"
    elif metric in {"media_snr", "host_snr"}:
        sentinel_count = sum(is_exact_sentinel(value, SNR_INVALID_BOUNDARY) for value in readings)
        boundary_count = sum(value <= SNR_INVALID_BOUNDARY for value in readings)
        boundary = SNR_INVALID_BOUNDARY
        sentinel = SNR_INVALID_BOUNDARY
        predicate_type = "physical_sentinel" if sentinel_count else "engineering_range"
    else:
        sentinel_count = sum(value <= SERDES_INVALID_BOUNDARY for value in readings)
        boundary_count = sentinel_count
        boundary = SERDES_INVALID_BOUNDARY
        sentinel = 1.0
        predicate_type = "measurement_contract"

    return {
        "metric": metric,
        "observed": len(readings),
        "quality": "observed",
        "boundary": boundary,
        "sentinel": sentinel,
        "predicate_type": predicate_type,
        "boundary_count": boundary_count,
        "sentinel_count": sentinel_count,
        "bucket": lane_bucket(boundary_count, len(readings)),
        "minimum": min(readings),
        "maximum": max(readings),
    }


def case_quality_state(case: dict[str, Any]) -> dict[str, Any]:
    """Return Q0 quality state; blackout takes precedence over fault symptoms."""
    states: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for side in ("L1", "L2"):
        for metric in sorted(METRIC_ALIASES):
            key = f"{side}:{metric}"
            state = measurement_state(metric, metric_values(case, metric, side))
            states[key] = state
            if state["quality"] == "missing":
                missing.append(key)

    optical_keys = [f"{side}:{metric}" for side in ("L1", "L2") for metric in ("txpower", "rxpower")]
    media_keys = [f"{side}:media_snr" for side in ("L1", "L2")]
    optical_blackout = all(
        states[key]["observed"] > 0 and states[key]["sentinel_count"] == states[key]["observed"]
        for key in optical_keys
    ) and all(
        states[key]["observed"] > 0 and states[key]["boundary_count"] == states[key]["observed"]
        for key in media_keys
    )
    quality = "optical_blackout" if optical_blackout else "partial_telemetry" if missing else "valid"
    return {
        "version": EVIDENCE_STATE_VERSION,
        "quality": quality,
        "optical_blackout": optical_blackout,
        "missing_measurements": tuple(missing),
        "measurements": states,
    }


def _path(
    side: str,
    measurement: str,
    predicate: str,
    symptom: str,
    layer: str,
    *,
    token: str,
    predicate_type: str,
    criterion: str,
    provenance: str,
    quantifier: str,
) -> dict[str, Any]:
    return {
        "side": f"side:{side}",
        "measurement": f"measurement:{side}:{measurement}",
        "predicate": f"predicate:{side}:{measurement}:{predicate}",
        "symptom": f"symptom:{symptom}",
        "layer": f"physical-layer:{layer}",
        "token": token,
        "predicate_type": predicate_type,
        "criterion": criterion,
        "provenance": provenance,
        "quantifier": quantifier,
        "learned": False,
    }


def physical_evidence_paths(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Q0/P/R five-layer paths from raw telemetry, without labels."""
    quality = case_quality_state(case)
    paths: list[dict[str, Any]] = []
    if quality["optical_blackout"]:
        for side in ("L1", "L2"):
            paths.append(_path(
                side, "telemetry_bundle", "optical_blackout", "telemetry_unreliable",
                "measurement_quality", token=f"quality:{side}:optical_blackout",
                predicate_type="data_quality", provenance="measurement-contract:C15",
                quantifier="both ends; all observed optical lanes",
                criterion="双端 TX/RX 全部精确为 -40.0，且双端 media_snr 全部 <= 0；按量测无效处理，不解释为激光关断",
            ))
        return paths

    layer_by_metric = {
        "txpower": "local_tx", "rxpower": "receive_path", "media_snr": "receive_path",
        "host_snr": "local_electrical", "serdes_snr": "local_electrical",
    }
    symptom_by_metric = {
        "txpower": "transmit_power_unavailable", "rxpower": "receive_power_unavailable",
        "media_snr": "media_signal_unusable", "host_snr": "host_signal_unusable",
        "serdes_snr": "serdes_state_invalid",
    }
    for side in ("L1", "L2"):
        for metric in sorted(METRIC_ALIASES):
            state = quality["measurements"][f"{side}:{metric}"]
            if state["quality"] == "missing":
                paths.append(_path(
                    side, metric, "missing", "critical_measurement_missing", "measurement_quality",
                    token=f"quality:{side}:{metric}:missing", predicate_type="data_quality",
                    criterion=f"{side}.{metric} 没有可用读数", provenance="evidence-pack-contract",
                    quantifier="all expected lanes missing",
                ))
                continue
            if not state["boundary_count"]:
                continue
            bucket = state["bucket"]
            if metric in {"txpower", "rxpower"} and state["sentinel_count"]:
                predicate = f"exact_minus_40:{bucket}"
                criterion = (
                    f"{side}.{metric} 有 {state['sentinel_count']}/{state['observed']} lane 精确为 -40.0 dBm；"
                    f"另有工程 drop 边界 <= -39 dBm。单独的 -40 不直接证明物理无光"
                )
                provenance = "physical-sentinel-and-engineering-boundary"
                predicate_type = "physical_sentinel"
            elif metric in {"media_snr", "host_snr"}:
                predicate = f"value_le_0:{bucket}"
                criterion = f"{side}.{metric} 有 {state['boundary_count']}/{state['observed']} lane <= 0"
                provenance = "measurement-contract"
                predicate_type = "physical_sentinel"
            elif metric == "serdes_snr":
                predicate = f"value_le_1:{bucket}"
                criterion = f"{side}.serdes_snr 有 {state['boundary_count']}/{state['observed']} lane <= 1；字段只作有效/失效状态，不按 dB 解释"
                provenance = "measurement-contract:C13"
                predicate_type = "measurement_contract"
            else:
                predicate = f"value_le_minus_39:{bucket}"
                criterion = f"{side}.{metric} 有 {state['boundary_count']}/{state['observed']} lane <= -39 dBm"
                provenance = "engineering-boundary"
                predicate_type = "engineering_range"
            paths.append(_path(
                side, metric, predicate, symptom_by_metric[metric], layer_by_metric[metric],
                token=f"physical:{side}:{metric}:{predicate}", predicate_type=predicate_type,
                criterion=criterion, provenance=provenance, quantifier=bucket,
            ))

    # Side-level relations only.  We intentionally do not pair lanes or subtract
    # TX-RX because endpoint lane correspondence/calibration is unresolved (C12).
    for local, peer in (("L1", "L2"), ("L2", "L1")):
        local_rx = quality["measurements"][f"{local}:rxpower"]
        local_media = quality["measurements"][f"{local}:media_snr"]
        local_serdes = quality["measurements"][f"{local}:serdes_snr"]
        peer_tx = quality["measurements"][f"{peer}:txpower"]
        peer_tx_present = peer_tx["observed"] and peer_tx["boundary_count"] < peer_tx["observed"]
        local_rx_down = local_rx["observed"] and local_rx["boundary_count"] > 0
        if peer_tx_present and local_rx_down:
            paths.append(_path(
                local, f"{peer}_tx_to_{local}_rx", "peer_tx_present_and_local_rx_down",
                "receive_direction_failure", "cross_end_optical_path",
                token=f"relation:{peer}:tx_present:{local}:rx_down", predicate_type="cross_end_relation",
                criterion=f"{peer} 端存在非 drop TX lane，同时 {local} 端至少一条 RX <= -39 dBm；只定位接收方向，不区分 fiber 与本端接收器",
                provenance="side-level-relation:C12-safe", quantifier="side-level; no lane pairing",
            ))
        local_rx_present = local_rx["observed"] and local_rx["boundary_count"] == 0
        decode_invalid = local_media["boundary_count"] > 0 or local_serdes["boundary_count"] > 0
        if local_rx_present and decode_invalid:
            paths.append(_path(
                local, "rx_to_decode", "rx_present_and_decode_invalid",
                "local_decode_or_electrical_failure", "local_electrical",
                token=f"relation:{local}:rx_present:decode_invalid", predicate_type="within_side_relation",
                criterion=f"{local} 端 RX 全部高于 -39 dBm，但 media_snr<=0 或 serdes_snr<=1",
                provenance="side-level-relation", quantifier="side-level",
            ))
    return paths


def quality_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Quality states must agree before a pair can be called an exact match."""
    left_quality, right_quality = case_quality_state(left), case_quality_state(right)
    return (
        left_quality["quality"] == right_quality["quality"]
        and left_quality["missing_measurements"] == right_quality["missing_measurements"]
    )


def fit_edge_idf(graphs: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = [set(graph.get("edges", ())) for graph in graphs]
    document_count = len(rows)
    frequencies = Counter(edge for row in rows for edge in row)
    return {
        edge: math.log((1 + document_count) / (1 + count)) + 1.0
        for edge, count in frequencies.items()
    }


def weighted_edge_jaccard(left: Iterable[str], right: Iterable[str], idf: dict[str, float]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    shared = left_set & right_set
    numerator = sum(idf.get(edge, 1.0) for edge in shared)
    denominator = sum(idf.get(edge, 1.0) for edge in union)
    return round(numerator / denominator, 8) if denominator else 0.0
