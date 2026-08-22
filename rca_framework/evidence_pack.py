"""N1 证据包：告警与多源遥测标准化之后、特征抽取之前的统一输入契约。

为什么要单独立一个类型，而不是继续到处传 `Dict[str, Any]`：

1. **标签隔离必须是结构性的，不能靠自觉。** `EvidencePack.from_case` 是唯一入口，
   它在构造时就把 `label` 摘掉。下游模块拿到的对象里根本没有标签字段，
   因此「特征抽取时忘了摘标签」这类泄漏在类型层面就不可能发生。
2. **「没有异常」和「没有数据」必须可区分。** 证据包显式记录 `observed_fields` 与
   `missing_fields`，而不是让下游从空集合里猜。阶段 1 已经证明这两种情况在 legacy
   里被混成同一个空 anomaly 集合，导致 22 条 case 直接退化成类别先验。
3. **N5c 的 prompt 和 N8 的回灌需要同一份原文。** 三个消费者（M1 特征抽取、
   M8 prompt 构造、M11 回灌）读的必须是同一个快照，否则报告里的证据和推理用的证据会对不上。

本模块不参与 legacy 路径，`rca_framework.cli` 不 import 它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .anomaly import (
    DOWN_THRESHOLDS,
    METRIC_ALIASES,
    STATUS_KEYS,
    abnormal_status,
    metric_block,
    safe_float,
)
from .types import SIDES
from .topology import lane_profile_of, lane_widths_of, source_dataset_of, topology_id_of


EVIDENCE_PACK_SCHEMA = "evidence-pack-v2-topology"

#: 非遥测的上下文字段。它们进 prompt 和报告，但不进 signature——
#: T1 的家族消融已经证明把它们当特征会把每个 case 推成唯一 signature（见 Progress 9.3）。
CONTEXT_FIELDS: Tuple[str, ...] = (
    "alarm_name",
    "alarm_time",
    "alarm_ip_interface",
    "link_location",
    "link_side_ip_interface_map",
    "Lane number",
    "vendor",
    "task_id",
    "chip",
    "port",
    "fec_error",
    "crc_error",
    "port_down_dt",
)

#: 逐侧的标量遥测。它们不是 per-lane 的，因此不参与 lane 级判断，但约束库要用。
SCALAR_FIELDS: Tuple[str, ...] = ("Temperature", "Voltage")

TELEMETRY_STATUSES: Tuple[str, ...] = ("no_telemetry", "partial_telemetry", "full_telemetry")
FIELD_STATES: Tuple[str, ...] = ("observed", "missing", "not_applicable", "invalid")


def _lane_map(case: Dict[str, Any], field_name: str, side: str) -> Dict[str, Optional[float]]:
    block = case.get(field_name)
    block = block.get(side) if isinstance(block, dict) else None
    if isinstance(block, dict):
        return {str(lane): safe_float(value) for lane, value in block.items()}
    if isinstance(block, list):
        return {str(index): safe_float(value) for index, value in enumerate(block)}
    return {}


def _metric_state(case: Dict[str, Any], field_name: str, side: str) -> str:
    block = case.get(field_name)
    if block is None:
        return "missing"
    if not isinstance(block, (dict, list)):
        return "invalid"
    if isinstance(block, dict) and side not in block:
        return "not_applicable"
    values = _lane_map(case, field_name, side)
    if not values:
        return "missing"
    return "observed" if any(value is not None for value in values.values()) else "invalid"


@dataclass(frozen=True)
class MetricReading:
    """某一侧某个 per-lane 指标的读数快照。

    `lanes` 保留原始值，包括断光哨兵，不做任何过滤。legacy 的
    `directional_loss` 之所以从未触发，正是因为它在这一层就把断光 lane 过滤掉了。
    """

    side: str
    metric: str
    lanes: Dict[str, Optional[float]]

    @property
    def observed(self) -> bool:
        return any(value is not None for value in self.lanes.values())

    @property
    def lane_count(self) -> int:
        return sum(1 for value in self.lanes.values() if value is not None)

    def to_dict(self) -> Dict[str, Any]:
        return {"side": self.side, "metric": self.metric, "lanes": dict(self.lanes)}


@dataclass
class EvidencePack:
    """N1 的产出。构造后即为只读快照，下游不得回写。"""

    case_id: str
    telemetry: Dict[str, Any]
    readings: Tuple[MetricReading, ...]
    statuses: Dict[str, Optional[str]]
    scalars: Dict[str, Optional[float]]
    context: Dict[str, Any]
    observed_fields: Tuple[str, ...]
    missing_fields: Tuple[str, ...]
    field_states: Dict[str, str] = field(default_factory=dict)
    source_dataset: str = ""
    topology_id: str = ""
    lane_profile: str = ""
    lane_widths: Dict[str, Dict[str, int]] = field(default_factory=dict)
    schema_version: str = EVIDENCE_PACK_SCHEMA

    @classmethod
    def from_case(cls, case: Dict[str, Any], *, source_dataset: str = "") -> "EvidencePack":
        """从标准化 case 构造证据包。`label` 在这里被摘掉，且不保留任何副本。"""
        telemetry = {key: value for key, value in case.items() if key != "label"}

        readings: List[MetricReading] = []
        observed: List[str] = []
        missing: List[str] = []
        field_states: Dict[str, str] = {}
        for side in SIDES:
            for metric in sorted(METRIC_ALIASES):
                field_key = _metric_key(telemetry, metric)
                state = _metric_state(telemetry, field_key, side)
                reading = MetricReading(side=side, metric=metric, lanes=_lane_map(telemetry, field_key, side))
                readings.append(reading)
                name = f"{side}.{metric}"
                field_states[name] = state
                (observed if state == "observed" else missing).append(name)

        statuses: Dict[str, Optional[str]] = {}
        for side in SIDES:
            for status in STATUS_KEYS:
                block = telemetry.get(status)
                value = block.get(side) if isinstance(block, dict) else None
                statuses[f"{side}.{status}"] = None if value is None else str(value)
                name = f"{side}.{status}"
                if value is not None:
                    state = "observed"
                elif block is None:
                    state = "missing"
                elif isinstance(block, dict) and side not in block:
                    state = "not_applicable"
                else:
                    state = "invalid"
                field_states[name] = state
                (observed if state == "observed" else missing).append(name)

        scalars: Dict[str, Optional[float]] = {}
        for side in SIDES:
            for name in SCALAR_FIELDS:
                block = telemetry.get(name)
                scalars[f"{side}.{name}"] = safe_float(block.get(side)) if isinstance(block, dict) else None

        effective_source = source_dataset_of(case, source_dataset)
        return cls(
            case_id=str(case.get("case_id", "unknown")),
            telemetry=telemetry,
            readings=tuple(readings),
            statuses=statuses,
            scalars=scalars,
            context={name: telemetry.get(name) for name in CONTEXT_FIELDS if name in telemetry},
            observed_fields=tuple(sorted(set(observed))),
            missing_fields=tuple(sorted(set(missing))),
            field_states=dict(sorted(field_states.items())),
            source_dataset=effective_source,
            topology_id=topology_id_of(case, effective_source),
            lane_profile=lane_profile_of(case, effective_source),
            lane_widths=lane_widths_of(case),
        )

    @property
    def expected_field_count(self) -> int:
        return len(SIDES) * (len(METRIC_ALIASES) + len(STATUS_KEYS))

    @property
    def coverage(self) -> float:
        if not self.expected_field_count:
            return 0.0
        return min(1.0, len(self.observed_fields) / self.expected_field_count)

    @property
    def telemetry_status(self) -> str:
        """区分「没采到数」「只采到一部分」「采全了」。

        注意这里回答的**不是**「有没有异常」。零异常 case 在这里可能是
        `full_telemetry`，意思是「采全了并且都正常」，这与 `no_telemetry` 是相反的结论。
        """
        if not self.observed_fields:
            return "no_telemetry"
        return "full_telemetry" if len(self.observed_fields) >= self.expected_field_count else "partial_telemetry"

    def reading(self, side: str, metric: str) -> MetricReading:
        for item in self.readings:
            if item.side == side and item.metric == metric:
                return item
        raise KeyError(f"no reading for {side}.{metric}")

    @property
    def optical_blackout(self) -> bool:
        """实现约束 `C15`：全链路光功率读数同时触底且 TxLOS 仍报 Normal。

        这种状态下断光哨兵的含义会翻转——它表示「读不到数」而不是「没有光」。
        判据里必须带上 TxLOS：模块若真的关断了激光，TxLOS 应当告警；
        它报 Normal 却读到哨兵，说明矛盾出在采集侧而不是器件侧。

        这个属性存在的意义是：这类 case 会产出十几个特征 token 看起来证据充分，
        但它们全都源自同一个失效的采集通道，实际上一条有效证据都没有。
        **token 多不等于证据强**，路由必须能识别这一点。
        """
        saw_reading = False
        for item in self.readings:
            if item.metric not in ("txpower", "rxpower"):
                continue
            values = [value for value in item.lanes.values() if value is not None]
            if not values:
                continue
            saw_reading = True
            if any(value > DOWN_THRESHOLDS[item.metric] for value in values):
                return False
        if not saw_reading:
            return False
        tx_los = [self.statuses.get(f"{side}.TxLOS") for side in SIDES]
        return any(value is not None and not abnormal_status(value) for value in tx_los)

    def has_label_field(self) -> bool:
        """证据包内不应存在任何标签字段。用于测试与运行期自检。"""
        return "label" in self.telemetry or "label" in self.context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "source_dataset": self.source_dataset,
            "topology_id": self.topology_id,
            "lane_profile": self.lane_profile,
            "lane_widths": self.lane_widths,
            "telemetry": self.telemetry,
            "readings": [item.to_dict() for item in self.readings],
            "statuses": dict(self.statuses),
            "scalars": dict(self.scalars),
            "context": dict(self.context),
            "observed_fields": list(self.observed_fields),
            "missing_fields": list(self.missing_fields),
            "field_states": dict(sorted(self.field_states.items())),
            "coverage": round(self.coverage, 8),
            "telemetry_status": self.telemetry_status,
            "optical_blackout": self.optical_blackout,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "EvidencePack":
        return cls(
            case_id=value["case_id"],
            telemetry=dict(value["telemetry"]),
            readings=tuple(
                MetricReading(side=item["side"], metric=item["metric"], lanes=dict(item["lanes"]))
                for item in value.get("readings", [])
            ),
            statuses=dict(value.get("statuses", {})),
            scalars=dict(value.get("scalars", {})),
            context=dict(value.get("context", {})),
            observed_fields=tuple(value.get("observed_fields", [])),
            missing_fields=tuple(value.get("missing_fields", [])),
            field_states=dict(value.get("field_states", {})),
            source_dataset=value.get("source_dataset", ""),
            topology_id=value.get("topology_id", ""),
            lane_profile=value.get("lane_profile", ""),
            lane_widths={
                str(metric): {str(side): int(width) for side, width in widths.items()}
                for metric, widths in value.get("lane_widths", {}).items()
            },
            schema_version=value.get("schema_version", EVIDENCE_PACK_SCHEMA),
        )


def _metric_key(case: Dict[str, Any], metric: str) -> str:
    """返回 case 里实际使用的指标键名，兼容 `METRIC_ALIASES` 的多种写法。"""
    for alias in METRIC_ALIASES[metric]:
        if isinstance(case.get(alias), dict):
            return alias
    return metric


def build_packs(
    cases: Sequence[Dict[str, Any]],
    *,
    source_dataset: str = "",
) -> List[EvidencePack]:
    return [EvidencePack.from_case(case, source_dataset=source_dataset) for case in cases]


def labels_of(cases: Sequence[Dict[str, Any]]) -> List[str]:
    """标签与证据包分开保存。训练时由调用方显式配对，检索时可以整条不传。"""
    return [str(case.get("label", "")) for case in cases]
