from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Anomaly, CaseEvidence, SIDES


METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "rxpower": ("rxpower", "rxPower", "RxPower"),
    "txpower": ("txpower", "txPower", "TxPower"),
    "media_snr": ("media_snr", "mediaSNR", "MediaSNR"),
    "host_snr": ("host_snr", "hostSNR", "HostSNR"),
    "serdes_snr": ("serdes_snr", "serdesSNR", "SerdesSNR"),
}
STATUS_KEYS = ("TxLOS", "TxLOL", "RxLOS", "RxLOL")
DOWN_THRESHOLDS = {"rxpower": -39.0, "txpower": -39.0, "media_snr": 0.0, "host_snr": 0.0, "serdes_snr": 0.0}
METRIC_NOUNS = {
    "rxpower": "接收光功率",
    "txpower": "发送光功率",
    "media_snr": "介质侧信噪比",
    "host_snr": "主机侧信噪比",
    "serdes_snr": "SerDes信噪比",
}


def safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def robust_fence(values: Sequence[float], multiplier: float = 3.0) -> Tuple[Optional[float], Optional[float]]:
    if len(values) < 4:
        return None, None
    q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
    if q1 is None or q3 is None:
        return None, None
    width = max(q3 - q1, 1e-9)
    return q1 - multiplier * width, q3 + multiplier * width


def robust_upper(values: Sequence[float], multiplier: float = 3.0) -> Optional[float]:
    return robust_fence(values, multiplier)[1]


def metric_block(case: Dict[str, Any], metric: str) -> Dict[str, Any]:
    for key in METRIC_ALIASES[metric]:
        value = case.get(key)
        if isinstance(value, dict):
            return value
    return {}


def flatten_numeric(value: Any) -> List[float]:
    if isinstance(value, dict):
        return [number for child in value.values() for number in flatten_numeric(child)]
    if isinstance(value, list):
        return [number for child in value for number in flatten_numeric(child)]
    number = safe_float(value)
    return [] if number is None else [number]


def metric_values(case: Dict[str, Any], metric: str, side: str, *, healthy_only: bool = False) -> List[float]:
    values = flatten_numeric(metric_block(case, metric).get(side))
    if healthy_only:
        threshold = DOWN_THRESHOLDS[metric]
        return [value for value in values if value > threshold]
    return values


def directional_loss(case: Dict[str, Any], source: str, target: str) -> Optional[float]:
    tx = metric_values(case, "txpower", source, healthy_only=True)
    rx = metric_values(case, "rxpower", target, healthy_only=True)
    if not tx or not rx:
        return None
    return abs(mean(tx) - mean(rx))


@dataclass
class ThresholdModel:
    value_fences: Dict[str, Tuple[Optional[float], Optional[float]]]
    spread_upper: Dict[str, Optional[float]]
    loss_upper: Dict[str, Optional[float]]
    fitted_case_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ThresholdModel":
        return cls(
            value_fences={key: tuple(item) for key, item in value["value_fences"].items()},
            spread_upper=dict(value["spread_upper"]),
            loss_upper=dict(value["loss_upper"]),
            fitted_case_count=int(value["fitted_case_count"]),
        )


def fit_thresholds(cases: Sequence[Dict[str, Any]]) -> ThresholdModel:
    values: Dict[str, List[float]] = defaultdict(list)
    spreads: Dict[str, List[float]] = defaultdict(list)
    losses: Dict[str, List[float]] = defaultdict(list)
    for case in cases:
        for side in SIDES:
            for metric in METRIC_ALIASES:
                key = f"{side}:{metric}"
                healthy = metric_values(case, metric, side, healthy_only=True)
                values[key].extend(healthy)
                if len(healthy) >= 2:
                    spreads[key].append(max(healthy) - min(healthy))
        for source, target in (("L1", "L2"), ("L2", "L1")):
            value = directional_loss(case, source, target)
            if value is not None:
                losses[f"{source}_to_{target}"].append(value)
    return ThresholdModel(
        value_fences={key: robust_fence(items) for key, items in values.items()},
        spread_upper={key: robust_upper(items) for key, items in spreads.items()},
        loss_upper={key: robust_upper(items) for key, items in losses.items()},
        fitted_case_count=len(cases),
    )


def anomaly(
    anomaly_id: str,
    node_type: str,
    noun: str,
    relation: str,
    side: str,
    metric: str,
    severity: float,
    evidence: str,
) -> Anomaly:
    return Anomaly(anomaly_id, node_type, noun, relation, side, metric, round(max(0.0, severity), 6), evidence)


def abnormal_status(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"abnormal", "down", "fault", "error", "los", "lol", "true", "1"}


def extract_evidence(case: Dict[str, Any], thresholds: ThresholdModel) -> CaseEvidence:
    found: Dict[str, Anomaly] = {}
    missing: List[str] = []
    observed = 0
    expected = len(SIDES) * (len(METRIC_ALIASES) + len(STATUS_KEYS))

    def add(item: Anomaly) -> None:
        current = found.get(item.anomaly_id)
        if current is None or item.severity > current.severity:
            found[item.anomaly_id] = item

    for side in SIDES:
        for metric in METRIC_ALIASES:
            values = metric_values(case, metric, side)
            field = f"{side}.{metric}"
            if not values:
                missing.append(field)
                continue
            observed += 1
            down = [value for value in values if value <= DOWN_THRESHOLDS[metric]]
            healthy = [value for value in values if value > DOWN_THRESHOLDS[metric]]
            noun = METRIC_NOUNS[metric]
            if down:
                add(anomaly(
                    f"signal_drop:{side}:{metric}", "SignalDrop", f"{side}{noun}中断", "HAS_SIGNAL_DROP",
                    side, metric, len(down) / len(values), f"{field} {len(down)}/{len(values)} lane at down sentinel",
                ))
            low, high = thresholds.value_fences.get(f"{side}:{metric}", (None, None))
            if healthy and low is not None and min(healthy) < low:
                delta = (low - min(healthy)) / (abs(low) + 1.0)
                add(anomaly(
                    f"low_outlier:{side}:{metric}", "LowSignal", f"{side}{noun}偏低", "HAS_LOW_SIGNAL",
                    side, metric, delta, f"{field} min={min(healthy):.4g} below fence={low:.4g}",
                ))
            if healthy and high is not None and max(healthy) > high:
                delta = (max(healthy) - high) / (abs(high) + 1.0)
                add(anomaly(
                    f"high_outlier:{side}:{metric}", "HighSignal", f"{side}{noun}偏高", "HAS_HIGH_SIGNAL",
                    side, metric, delta, f"{field} max={max(healthy):.4g} above fence={high:.4g}",
                ))
            spread_limit = thresholds.spread_upper.get(f"{side}:{metric}")
            if len(healthy) >= 2 and spread_limit is not None and max(healthy) - min(healthy) > spread_limit:
                spread = max(healthy) - min(healthy)
                add(anomaly(
                    f"lane_imbalance:{side}:{metric}", "LaneImbalance", f"{side}{noun}通道不均衡", "HAS_LANE_IMBALANCE",
                    side, metric, spread / (abs(spread_limit) + 1e-6), f"{field} lane spread={spread:.4g} over limit={spread_limit:.4g}",
                ))
        for status in STATUS_KEYS:
            block = case.get(status)
            field = f"{side}.{status}"
            if not isinstance(block, dict) or block.get(side) is None:
                missing.append(field)
                continue
            observed += 1
            if abnormal_status(block.get(side)):
                add(anomaly(
                    f"status_fault:{side}:{status}", "DeviceStatusFault", f"{side}{status}状态异常", "HAS_STATUS_FAULT",
                    side, status, 1.0, f"{field}={block.get(side)}",
                ))

    for source, target in (("L1", "L2"), ("L2", "L1")):
        direction = f"{source}_to_{target}"
        value, limit = directional_loss(case, source, target), thresholds.loss_upper.get(direction)
        if value is not None and limit is not None and value > limit:
            add(anomaly(
                f"directional_loss:{direction}:optical_power", "DirectionalLoss", f"{direction}方向光损耗异常",
                "HAS_DIRECTIONAL_LOSS", "fiber", "optical_power", value / (limit + 1e-6),
                f"{direction} mean optical loss={value:.4g} over limit={limit:.4g}",
            ))

    ids = set(found)
    patterns = (
        ("L1", "L2"),
        ("L2", "L1"),
    )
    for source, target in patterns:
        tx_bad = any(f":{source}:txpower" in item for item in ids) or any(f":{source}:TxLO" in item for item in ids)
        rx_bad = any(f":{target}:rxpower" in item for item in ids) or any(f":{target}:RxLO" in item for item in ids)
        if tx_bad and rx_bad:
            direction = f"{source}_to_{target}"
            add(anomaly(
                f"coupled_fault:{direction}:tx_rx", "CoupledTxRxFault", f"{direction}发送接收耦合异常",
                "HAS_COUPLED_TX_RX_FAULT", "fiber", "tx_rx", 1.0,
                f"{source} TX anomaly co-occurs with {target} RX anomaly",
            ))
    if any(item.startswith("directional_loss:L1_to_L2") for item in found) and any(item.startswith("directional_loss:L2_to_L1") for item in found):
        add(anomaly(
            "bidirectional_loss:fiber:optical_power", "BidirectionalLoss", "双向光损耗异常", "HAS_BIDIRECTIONAL_LOSS",
            "fiber", "optical_power", 1.0, "both optical directions exceed learned loss fences",
        ))

    summary_keys = ("alarm_name", "alarm_time", "link_location", "link_side_ip_interface_map", "Lane number")
    return CaseEvidence(
        case_id=str(case.get("case_id", "unknown")),
        label=str(case.get("label", "")),
        anomalies=sorted(found.values(), key=lambda item: (item.node_type, item.anomaly_id)),
        observed_fields=observed,
        expected_fields=expected,
        missing_fields=sorted(set(missing)),
        summary={key: case.get(key) for key in summary_keys if key in case},
    )
