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


LANE_SIGNATURES: Tuple[str, ...] = (
    "tx_ok_rx_down",           # 本端发光正常而对端该 lane 收无光：最强的介质/对端接收指向
    "tx_down",                 # 本端该 lane 就没发出光：根因在发送端，不在介质
    "bidirectional_same_lane", # 同一条 lane 在两个方向都断：整条 lane 或其光纤对被切断
    "uniform_loss_all_lanes",  # 所有 lane 损耗接近且整体超出学到的 fence：连接器/端面类整体衰减
    "single_lane_outlier",     # 只有一条 lane 的损耗显著偏离其余 lane
)

# 声明式门限，尚未标定。`lane_directional_loss` 会把逐 lane 的原始损耗一并返回，
# 便于后续用 artifacts 里的分布回头校准这两个值。
UNIFORM_LOSS_SPREAD_DB = 1.0
SINGLE_LANE_OUTLIER_DB = 2.0


@dataclass(frozen=True)
class LanePair:
    """一条 lane 在某个方向上的发送与接收观测。

    与 `directional_loss` 的关键差别：这里不做 `healthy_only` 过滤，断光状态被
    保留为 `tx_down` / `rx_down` 布尔量。legacy 实现先把 `<= -39.0` 的断光 lane
    过滤掉，再取均值差，于是"tx 正常但对端 rx 断光"这个最指向介质故障的模式
    被过滤器直接消掉，`directional_loss` 因此从未触发。
    """

    lane: str
    tx: Optional[float]
    rx: Optional[float]
    tx_down: bool
    rx_down: bool
    loss: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def lane_values(case: Dict[str, Any], metric: str, side: str) -> Dict[str, Optional[float]]:
    """按 lane 号返回原始值，不做任何过滤。非数值与嵌套结构记为 None。"""
    block = metric_block(case, metric).get(side)
    if isinstance(block, dict):
        return {str(lane): safe_float(value) for lane, value in block.items()}
    if isinstance(block, list):
        return {str(index): safe_float(value) for index, value in enumerate(block)}
    return {}


def _lane_order(lanes: Iterable[str]) -> List[str]:
    return sorted(lanes, key=lambda lane: (0, int(lane)) if lane.isdigit() else (1, lane))


def _is_down(value: Optional[float], metric: str) -> bool:
    return value is not None and value <= DOWN_THRESHOLDS[metric]


def lane_pairs(case: Dict[str, Any], source: str, target: str) -> List[LanePair]:
    """把 `source` 侧的发送与 `target` 侧的接收按 lane 号配对。

    按 lane 号配对是当前数据唯一可用的对应关系。L1 是 400G、L2 是 200G，两端的
    lane 号未必物理对应，因此 `lane_directional_loss` 会同时报出
    `lane_count_mismatch`，供后续判断该 signature 是否可信。
    """
    tx_values = lane_values(case, "txpower", source)
    rx_values = lane_values(case, "rxpower", target)
    pairs: List[LanePair] = []
    for lane in _lane_order(set(tx_values) | set(rx_values)):
        tx, rx = tx_values.get(lane), rx_values.get(lane)
        tx_down, rx_down = _is_down(tx, "txpower"), _is_down(rx, "rxpower")
        loss = tx - rx if tx is not None and rx is not None and not tx_down and not rx_down else None
        pairs.append(LanePair(lane=lane, tx=tx, rx=rx, tx_down=tx_down, rx_down=rx_down, loss=loss))
    return pairs


def lane_directional_loss(
    case: Dict[str, Any],
    source: str,
    target: str,
    thresholds: ThresholdModel | None = None,
) -> Dict[str, Any]:
    """给出 `source -> target` 方向的 lane 级 signature 与原始损耗。

    只做观测，不产生 `Anomaly`，因此不影响 legacy 的 `anomaly_id` 集合与打分。
    """
    pairs = lane_pairs(case, source, target)
    direction = f"{source}_to_{target}"
    reverse = {pair.lane: pair for pair in lane_pairs(case, target, source)}
    losses = {pair.lane: round(pair.loss, 8) for pair in pairs if pair.loss is not None}
    tx_down_lanes = [pair.lane for pair in pairs if pair.tx_down]
    rx_down_lanes = [pair.lane for pair in pairs if pair.rx_down]
    tx_ok_rx_down_lanes = [
        pair.lane for pair in pairs if pair.rx_down and pair.tx is not None and not pair.tx_down
    ]
    bidirectional_lanes = [
        pair.lane for pair in pairs
        if (pair.tx_down or pair.rx_down)
        and pair.lane in reverse
        and (reverse[pair.lane].tx_down or reverse[pair.lane].rx_down)
    ]

    learned_limit = thresholds.loss_upper.get(direction) if thresholds is not None else None
    values = list(losses.values())
    mean_loss = mean(values) if values else None
    spread = max(values) - min(values) if len(values) >= 2 else None
    over_limit = bool(mean_loss is not None and learned_limit is not None and mean_loss > learned_limit)

    outlier_lanes: List[str] = []
    if len(values) >= 3:
        middle = percentile(values, 0.5)
        if middle is not None:
            outlier_lanes = [lane for lane, value in losses.items() if abs(value - middle) >= SINGLE_LANE_OUTLIER_DB]

    signatures: List[str] = []
    if tx_ok_rx_down_lanes:
        signatures.append("tx_ok_rx_down")
    if tx_down_lanes:
        signatures.append("tx_down")
    if bidirectional_lanes:
        signatures.append("bidirectional_same_lane")
    if over_limit and len(values) >= 2 and spread is not None and spread <= UNIFORM_LOSS_SPREAD_DB:
        signatures.append("uniform_loss_all_lanes")
    if len(outlier_lanes) == 1:
        signatures.append("single_lane_outlier")

    tx_lane_count = len(lane_values(case, "txpower", source))
    rx_lane_count = len(lane_values(case, "rxpower", target))
    return {
        "direction": direction,
        "lane_count": len(pairs),
        "lane_count_mismatch": tx_lane_count != rx_lane_count,
        "paired_lane_count": len(losses),
        "lane_losses": losses,
        "mean_loss": round(mean_loss, 8) if mean_loss is not None else None,
        "max_loss": round(max(values), 8) if values else None,
        "loss_spread": round(spread, 8) if spread is not None else None,
        "learned_loss_upper": learned_limit,
        "mean_loss_over_learned_limit": over_limit,
        "tx_down_lanes": tx_down_lanes,
        "rx_down_lanes": rx_down_lanes,
        "tx_ok_rx_down_lanes": tx_ok_rx_down_lanes,
        "bidirectional_down_lanes": bidirectional_lanes,
        "single_lane_outlier_lanes": outlier_lanes,
        "signatures": signatures,
    }


def lane_loss_report(case: Dict[str, Any], thresholds: ThresholdModel | None = None) -> Dict[str, Any]:
    """两个方向合起来的 lane 级观测，供影子模式统计使用。"""
    directions = {
        f"{source}_to_{target}": lane_directional_loss(case, source, target, thresholds)
        for source, target in (("L1", "L2"), ("L2", "L1"))
    }
    triggered = sorted({name for row in directions.values() for name in row["signatures"]})
    return {
        "directions": directions,
        "signatures": triggered,
        "any_signature": bool(triggered),
    }


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
