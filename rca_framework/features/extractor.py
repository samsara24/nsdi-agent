"""M1 抽取器：把一个 case 的原始遥测按特征字典 v1 变成稀疏可解释 token 集合。

与 legacy `anomaly.extract_evidence` 的关系：本模块只读同一批原始字段，
不调用也不改写 `extract_evidence`，因此 legacy 的 `anomaly_id` 集合与 58/85
回归锚点完全不受影响。

连续量的分档边界不写死在代码里，而是由 `fit_feature_model` 在训练集上拟合成
`FeatureModel` 并随实验产物落盘，这样特征字典版本 + FeatureModel 才能唯一确定
一次实验用的是什么特征。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..anomaly import (
    DOWN_THRESHOLDS,
    METRIC_ALIASES,
    STATUS_KEYS,
    ThresholdModel,
    abnormal_status,
    lane_directional_loss,
    lane_values,
    metric_values,
    percentile,
    safe_float,
)
from ..evidence_pack import EvidencePack
from ..types import SIDES
from .dictionary import FEATURE_DICTIONARY, TAIL_QUANTILES, FeatureDictionary


FEATURE_MODEL_VERSION = "feature-model-v1"

#: 参与分位数分档与两端对称性比较的连续统计量。
#: key 是统计量名，value 是 (metric, 聚合方式)。聚合只用健康 lane。
LEVEL_STATISTICS: Dict[str, Tuple[str, str]] = {
    "rxpower_mean": ("rxpower", "mean"),
    "txpower_mean": ("txpower", "mean"),
    "media_snr_min": ("media_snr", "min"),
}

ASYMMETRY_STATISTICS: Tuple[str, ...] = ("rxpower_mean", "txpower_mean", "media_snr_min")

ALARM_KINDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("lane_flap", ("频繁升降lane", "频繁升降 lane")),
    ("lane_degrade", ("降lane", "降 lane")),
    ("link_flap", ("linkflap", "link flap")),
)


def side_statistic(case: Dict[str, Any], side: str, statistic: str) -> Optional[float]:
    metric, aggregation = LEVEL_STATISTICS[statistic]
    values = metric_values(case, metric, side, healthy_only=True)
    if not values:
        return None
    return mean(values) if aggregation == "mean" else min(values)


@dataclass
class FeatureModel:
    """特征字典 v1 中所有数据驱动边界的拟合结果。"""

    version: str
    dictionary_version: str
    dictionary_hash: str
    fitted_case_count: int
    level_edges: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    asymmetry_edges: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "dictionary_version": self.dictionary_version,
            "dictionary_hash": self.dictionary_hash,
            "fitted_case_count": self.fitted_case_count,
            "tail_quantiles": list(TAIL_QUANTILES),
            "level_edges": {key: list(value) for key, value in sorted(self.level_edges.items())},
            "asymmetry_edges": dict(sorted(self.asymmetry_edges.items())),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "FeatureModel":
        return cls(
            version=value["version"],
            dictionary_version=value["dictionary_version"],
            dictionary_hash=value["dictionary_hash"],
            fitted_case_count=int(value["fitted_case_count"]),
            level_edges={key: tuple(item) for key, item in value.get("level_edges", {}).items()},
            asymmetry_edges=dict(value.get("asymmetry_edges", {})),
        )


def fit_feature_model(
    packs: Sequence[EvidencePack],
    *,
    dictionary: FeatureDictionary = FEATURE_DICTIONARY,
) -> FeatureModel:
    """在训练集证据包上拟合分位数边界。

    入参是 `EvidencePack` 而不是原始 case，因为证据包里根本没有 `label` 字段，
    「拟合时不许看标签」这件事因此由类型保证，而不是靠调用方记得先 pop。
    """
    low_q, high_q = TAIL_QUANTILES
    cases = [pack.telemetry for pack in packs]
    level_edges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    asymmetry_edges: Dict[str, Optional[float]] = {}

    for statistic in sorted(LEVEL_STATISTICS):
        for side in SIDES:
            values = [
                value
                for value in (side_statistic(case, side, statistic) for case in cases)
                if value is not None
            ]
            key = f"{side}:{statistic}"
            if len(values) < 4:
                level_edges[key] = (None, None)
                continue
            level_edges[key] = (
                round(percentile(values, low_q), 8),
                round(percentile(values, high_q), 8),
            )

    for statistic in ASYMMETRY_STATISTICS:
        gaps: List[float] = []
        for case in cases:
            left, right = side_statistic(case, "L1", statistic), side_statistic(case, "L2", statistic)
            if left is not None and right is not None:
                gaps.append(abs(left - right))
        edge = percentile(gaps, high_q) if len(gaps) >= 4 else None
        asymmetry_edges[statistic] = round(edge, 8) if edge is not None else None

    return FeatureModel(
        version=FEATURE_MODEL_VERSION,
        dictionary_version=dictionary.version,
        dictionary_hash=dictionary.content_hash(),
        fitted_case_count=len(packs),
        level_edges=level_edges,
        asymmetry_edges=asymmetry_edges,
    )


# --- 逐家族抽取 -------------------------------------------------------------


def _signal_drop(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    tokens: List[str] = []
    for side in SIDES:
        for metric in sorted(METRIC_ALIASES):
            values = [value for value in lane_values(case, metric, side).values() if value is not None]
            if not values:
                continue
            down = [value for value in values if value <= DOWN_THRESHOLDS[metric]]
            if not down:
                continue
            if len(down) == len(values):
                bucket = "all_lanes"
            elif len(down) == 1:
                bucket = "single_lane"
            else:
                bucket = "partial_lanes"
            tokens.append(f"drop:{side}:{metric}:{bucket}")
    return tokens


def _status_fault(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    tokens: List[str] = []
    for side in SIDES:
        for status in STATUS_KEYS:
            block = case.get(status)
            if not isinstance(block, dict) or block.get(side) is None:
                continue
            if abnormal_status(block.get(side)):
                tokens.append(f"status:{side}:{status}")
    return tokens


def _fence_outlier(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    tokens: List[str] = []
    for side in SIDES:
        for metric in sorted(METRIC_ALIASES):
            healthy = metric_values(case, metric, side, healthy_only=True)
            if not healthy:
                continue
            low, high = thresholds.value_fences.get(f"{side}:{metric}", (None, None))
            if low is not None and min(healthy) < low:
                tokens.append(f"fence:{side}:{metric}:low")
            if high is not None and max(healthy) > high:
                tokens.append(f"fence:{side}:{metric}:high")
    return tokens


def _lane_imbalance(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    tokens: List[str] = []
    for side in SIDES:
        for metric in sorted(METRIC_ALIASES):
            healthy = metric_values(case, metric, side, healthy_only=True)
            limit = thresholds.spread_upper.get(f"{side}:{metric}")
            if len(healthy) >= 2 and limit is not None and max(healthy) - min(healthy) > limit:
                tokens.append(f"imbalance:{side}:{metric}")
    return tokens


def _lane_direction(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    tokens: List[str] = []
    for source, target in (("L1", "L2"), ("L2", "L1")):
        report = lane_directional_loss(case, source, target, thresholds)
        for signature in report["signatures"]:
            tokens.append(f"lane:{report['direction']}:{signature}")
    return tokens


def _level_tail(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    if model is None:
        return []
    tokens: List[str] = []
    for statistic in sorted(LEVEL_STATISTICS):
        for side in SIDES:
            value = side_statistic(case, side, statistic)
            low, high = model.level_edges.get(f"{side}:{statistic}", (None, None))
            if value is None or low is None or high is None:
                continue
            if value < low:
                tokens.append(f"level:{side}:{statistic}:low_tail")
            elif value > high:
                tokens.append(f"level:{side}:{statistic}:high_tail")
    return tokens


def _side_asymmetry(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    if model is None:
        return []
    tokens: List[str] = []
    for statistic in ASYMMETRY_STATISTICS:
        edge = model.asymmetry_edges.get(statistic)
        left, right = side_statistic(case, "L1", statistic), side_statistic(case, "L2", statistic)
        if edge is None or left is None or right is None:
            continue
        if abs(left - right) <= edge:
            continue
        lower = "L1" if left < right else "L2"
        tokens.append(f"asym:{statistic}:{lower}_lower")
    return tokens


def _port_width(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    block = case.get("Lane number")
    block = block if isinstance(block, dict) else {}

    def width(side: str) -> str:
        value = safe_float(block.get(side))
        return "unknown" if value is None else str(int(value))

    return [f"width:{width('L1')}x{width('L2')}"]


def _alarm_kind(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    text = str(case.get("alarm_name") or "").lower()
    for kind, needles in ALARM_KINDS:
        if any(needle.lower() in text for needle in needles):
            return [f"alarm:{kind}"]
    return ["alarm:other"]


def _telemetry_gap(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    observed = 0
    expected = len(SIDES) * (len(METRIC_ALIASES) + len(STATUS_KEYS))
    for side in SIDES:
        for metric in METRIC_ALIASES:
            if metric_values(case, metric, side):
                observed += 1
        for status in STATUS_KEYS:
            block = case.get(status)
            if isinstance(block, dict) and block.get(side) is not None:
                observed += 1
    if observed <= 0:
        state = "no_telemetry"
    elif observed >= expected:
        state = "full_telemetry"
    else:
        state = "partial_telemetry"
    return [f"telemetry:{state}"]


def _serdes_state(case: Dict[str, Any], thresholds: ThresholdModel, model: Optional[FeatureModel]) -> List[str]:
    tokens: List[str] = []
    for side in SIDES:
        values = lane_values(case, "serdes_snr", side)
        numeric = [value for value in values.values() if value is not None]
        if not values or not numeric:
            tokens.append(f"serdes:{side}:missing")
        elif all(value <= 1.0 for value in numeric):
            tokens.append(f"serdes:{side}:invalid")
        else:
            tokens.append(f"serdes:{side}:valid")
    return tokens


FAMILY_EXTRACTORS: Dict[str, Callable[[Dict[str, Any], ThresholdModel, Optional[FeatureModel]], List[str]]] = {
    "signal_drop": _signal_drop,
    "status_fault": _status_fault,
    "fence_outlier": _fence_outlier,
    "lane_imbalance": _lane_imbalance,
    "lane_direction": _lane_direction,
    "level_tail": _level_tail,
    "side_asymmetry": _side_asymmetry,
    "port_width": _port_width,
    "alarm_kind": _alarm_kind,
    "telemetry_gap": _telemetry_gap,
    "serdes_state": _serdes_state,
}


#: 互斥分档规则（T3 新增）。key 是 token 去掉最后一段分档后的前缀，
#: 同一前缀下只允许有一个分档同时成立。这不改动特征字典 v1 的内容指纹：
#: 它描述的是「同一维度的分档互斥」这一结构性事实，不引入任何新门限或新维度。
#:
#: `fence_outlier` 故意不在此列：同一侧同一指标的不同 lane 可以一个偏低一个偏高，
#: `fence:X:Y:low` 与 `fence:X:Y:high` 同时出现是合法的，不是冲突。
MUTUALLY_EXCLUSIVE_PREFIXES: Tuple[str, ...] = ("drop:", "level:")


def detect_token_conflicts(tokens: Sequence[str]) -> List[Tuple[str, ...]]:
    """找出同一维度上互相矛盾的 token 组合。

    正常抽取不应产生冲突；一旦出现，说明抽取规则或输入数据自相矛盾，
    应当作为证据冲突上报给 N6，而不是静默地把两个 token 一起塞进 signature。
    """
    groups: Dict[str, List[str]] = {}
    for token in tokens:
        if not token.startswith(MUTUALLY_EXCLUSIVE_PREFIXES):
            continue
        prefix = token.rsplit(":", 1)[0]
        groups.setdefault(prefix, []).append(token)
    return [tuple(sorted(items)) for _, items in sorted(groups.items()) if len(items) > 1]


@dataclass
class CaseFeatures:
    """一个 case 的特征向量。`tokens` 是 signature，`by_family` 保留可解释归因。"""

    case_id: str
    tokens: Tuple[str, ...]
    by_family: Dict[str, Tuple[str, ...]]
    dictionary_version: str
    dictionary_hash: str
    telemetry_status: str = "full_telemetry"
    missing_fields: Tuple[str, ...] = ()
    conflicts: Tuple[Tuple[str, ...], ...] = ()
    #: 约束 C15：本 case 的 token 全部来自一条失效的采集通道，看似证据充分实则无效。
    optical_blackout: bool = False

    @property
    def signature(self) -> str:
        return "|".join(self.tokens)

    @property
    def is_empty(self) -> bool:
        """空 signature。必须结合 `telemetry_status` 才能解释：

        `full_telemetry` 表示采全了且一切正常，`no_telemetry` 表示什么都没采到。
        这两种情况的处置完全相反，不能一起当成「零证据」。
        """
        return not self.tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tokens": list(self.tokens),
            "by_family": {name: list(items) for name, items in sorted(self.by_family.items())},
            "dictionary_version": self.dictionary_version,
            "dictionary_hash": self.dictionary_hash,
            "telemetry_status": self.telemetry_status,
            "missing_fields": list(self.missing_fields),
            "conflicts": [list(item) for item in self.conflicts],
            "optical_blackout": self.optical_blackout,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CaseFeatures":
        """从知识包恢复训练向量，不重新读取原始训练 case。"""
        return cls(
            case_id=str(value["case_id"]),
            tokens=tuple(str(item) for item in value.get("tokens", ())),
            by_family={
                str(name): tuple(str(item) for item in items)
                for name, items in value.get("by_family", {}).items()
            },
            dictionary_version=str(value.get("dictionary_version", "")),
            dictionary_hash=str(value.get("dictionary_hash", "")),
            telemetry_status=str(value.get("telemetry_status", "full_telemetry")),
            missing_fields=tuple(str(item) for item in value.get("missing_fields", ())),
            conflicts=tuple(
                tuple(str(token) for token in conflict)
                for conflict in value.get("conflicts", ())
            ),
            optical_blackout=bool(value.get("optical_blackout", False)),
        )


def extract_features(
    pack: EvidencePack,
    thresholds: ThresholdModel,
    model: Optional[FeatureModel] = None,
    *,
    dictionary: FeatureDictionary = FEATURE_DICTIONARY,
) -> CaseFeatures:
    """N2：证据包 -> 可解释稀疏特征向量。

    入参是 `EvidencePack`，因此这里拿不到标签，抽取过程在类型上就不可能泄漏。
    """
    by_family: Dict[str, Tuple[str, ...]] = {}
    tokens: List[str] = []
    for family in dictionary.families:
        produced = sorted(set(FAMILY_EXTRACTORS[family.name](pack.telemetry, thresholds, model)))
        if produced:
            by_family[family.name] = tuple(produced)
            tokens.extend(produced)
    ordered = tuple(sorted(set(tokens)))
    return CaseFeatures(
        case_id=pack.case_id,
        tokens=ordered,
        by_family=by_family,
        dictionary_version=dictionary.version,
        dictionary_hash=dictionary.content_hash(),
        telemetry_status=pack.telemetry_status,
        missing_fields=pack.missing_fields,
        conflicts=tuple(detect_token_conflicts(ordered)),
        optical_blackout=pack.optical_blackout,
    )


def extract_feature_tokens(
    pack: EvidencePack,
    thresholds: ThresholdModel,
    model: Optional[FeatureModel] = None,
    *,
    dictionary: FeatureDictionary = FEATURE_DICTIONARY,
) -> List[str]:
    return list(extract_features(pack, thresholds, model, dictionary=dictionary).tokens)
