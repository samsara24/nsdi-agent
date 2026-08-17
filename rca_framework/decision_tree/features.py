"""Continuous feature extraction for the rebuilt RCA decision tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..anomaly import DOWN_THRESHOLDS, STATUS_KEYS, abnormal_status
from ..evidence_pack import EvidencePack
from ..types import SIDES


NUMERIC_FEATURE_SCHEMA = "numeric-decision-tree-features-v1"
TREE_METRICS: Tuple[str, ...] = ("bias", "txpower", "rxpower", "host_snr", "media_snr", "serdes_snr")


@dataclass(frozen=True)
class NumericFeatureRow:
    """A single case represented by continuous values instead of token presence."""

    case_id: str
    values: Mapping[str, float]
    missing: Tuple[str, ...] = ()
    schema_version: str = NUMERIC_FEATURE_SCHEMA
    source_dataset: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Optional[float]:
        value = self.values.get(name)
        return float(value) if value is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "source_dataset": self.source_dataset,
            "values": {key: float(value) for key, value in sorted(self.values.items())},
            "missing": list(self.missing),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NumericFeatureRow":
        return cls(
            case_id=str(value["case_id"]),
            values={str(key): float(entry) for key, entry in value.get("values", {}).items()},
            missing=tuple(str(item) for item in value.get("missing", ())),
            schema_version=str(value.get("schema_version", NUMERIC_FEATURE_SCHEMA)),
            source_dataset=str(value.get("source_dataset", "")),
            metadata=dict(value.get("metadata", {})),
        )


def numeric_features_from_pack(pack: EvidencePack) -> NumericFeatureRow:
    values: Dict[str, float] = {}
    missing = set()

    for side in SIDES:
        for metric in TREE_METRICS:
            try:
                reading = pack.reading(side, metric)
            except KeyError:
                missing.add(f"{side}.{metric}")
                continue
            lane_values = [value for value in reading.lanes.values() if value is not None]
            prefix = f"{side}.{metric}"
            if not lane_values:
                missing.add(prefix)
                continue
            sentinel = DOWN_THRESHOLDS.get(metric)
            if sentinel is None:
                valid_values = lane_values
                down_values = []
            else:
                down_values = [value for value in lane_values if value <= sentinel]
                valid_values = [value for value in lane_values if value > sentinel]
            values[f"{prefix}.observed_count"] = float(len(lane_values))
            values[f"{prefix}.down_count"] = float(len(down_values))
            if valid_values:
                values[f"{prefix}.min"] = min(valid_values)
                values[f"{prefix}.max"] = max(valid_values)
                values[f"{prefix}.mean"] = sum(valid_values) / len(valid_values)
                values[f"{prefix}.spread"] = max(valid_values) - min(valid_values)
            else:
                # All lanes are down.  Keep the count features and mark the
                # continuous statistics as missing so the tree does not learn
                # fake means over sentinel defaults.
                for statistic in ("min", "max", "mean", "spread"):
                    missing.add(f"{prefix}.{statistic}")

        for scalar in ("Temperature", "Voltage"):
            key = f"{side}.{scalar}"
            value = pack.scalars.get(key)
            if value is None:
                missing.add(key)
            else:
                values[key] = float(value)

        for status in STATUS_KEYS:
            key = f"{side}.{status}"
            state = pack.statuses.get(key)
            if state is None:
                missing.add(key)
            else:
                values[key] = 1.0 if abnormal_status(state) else 0.0

    values["telemetry.coverage"] = float(pack.coverage)
    values["telemetry.optical_blackout"] = 1.0 if pack.optical_blackout else 0.0
    values["telemetry.observed_fields"] = float(len(pack.observed_fields))
    values["telemetry.missing_fields"] = float(len(pack.missing_fields))
    return NumericFeatureRow(
        case_id=pack.case_id,
        values=dict(sorted(values.items())),
        missing=tuple(sorted(missing)),
        source_dataset=pack.source_dataset,
        metadata={
            "telemetry_status": pack.telemetry_status,
            "optical_blackout": pack.optical_blackout,
        },
    )


def numeric_features_from_packs(packs: Sequence[EvidencePack]) -> Tuple[NumericFeatureRow, ...]:
    return tuple(numeric_features_from_pack(pack) for pack in packs)
