"""专家决策树：把现网人工排障经验实现为可复现、可标定的第三个候选源。

来源是 `docs/EXPERT_EXPERIENCE.md`，即现网专家规则系统的阈值表、异常级别、
故障模式优先级与两端裁决流程。它与本仓库其余部分有一个本质区别：

**这里的每一个阈值都来自项目之外的人工经验，没有一个是在本数据集上拟合的。**

这一点决定了它的评估口径。SOP、门限、标定表都必须区分 train/test，因为它们
从训练标签里学参数；专家规则不学任何参数，因此在 train 与 test 上的表现差异
只反映数据分布差异，不反映过拟合。也正因如此，它是迄今唯一一个可以拿全部
268 条 case 一起报告而不违反训练边界的判别器。

引入它的直接原因是迭代 2 的两个负面结果（见 Progress §9.32）：

1. 特征字典 + 浅决策树在这份数据上相对多数类只有 +1.4pp，而三条独立路线
   都指向 70~75% 的可辨识上限——说明缺的不是模型容量，是**先验方向知识**。
2. LLM 能引用约束却把归因方向用反（接收侧症状归给接收侧自己）。专家规则的
   `SINGLE_METRIC_DIRECTION` 就是这份方向知识的完整表述：介质侧信噪比与接收
   光功率的异常指向**对端发送链路**，主机侧信噪比、SerDes 信噪比与发送光功率
   的异常指向**本端**。

本模块只做诊断，不做标定。置信度必须由调用方在训练集上按 `group` 分组统计
（见 `ExpertCalibration`），原因与 SOP 相同：规则命中率本身不是可靠性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .evidence_pack import EvidencePack
from .types import SIDES, wilson_lower_bound


EXPERT_RULES_VERSION = "expert-rules-v1"

#: 专家规则只遍历这五个指标。`bias` 出现在预处理与 RF 特征里，但专家阈值表
#: 没有它，因此专家规则不用 bias 定界（`EXPERT_EXPERIENCE.md` §3）。
EXPERT_METRICS: Tuple[str, ...] = ("rxpower", "txpower", "host_snr", "media_snr", "serdes_snr")

#: `EXPERT_EXPERIENCE.md` §3.3 的阈值表，逐字照搬，不在本数据上重新拟合。
#:
#: `down` 一栏在文档里是「任一 lane 值**等于** down 阈值」。本实现用 `<=`，
#: 在 268 条 case 上与等值判定完全等价（实测：哨兵值全部精确等于 -40 / 0，
#: 没有任何读数落在阈值以下），改用 `<=` 只是为了对未来的越界读数保持健壮。
EXPERT_THRESHOLDS: Mapping[str, Mapping[str, float]] = {
    "rxpower": {"down": -40.0, "low": -2.5, "high": 4.6, "diff": 1.0},
    "txpower": {"down": -40.0, "low": -2.5, "high": 2.5, "diff": 1.3},
    "host_snr": {"down": 0.0, "low": 22.8, "high": 27.5, "diff": 2.5},
    "media_snr": {"down": 0.0, "low": 22.4, "high": 28.7, "diff": 3.0},
    "serdes_snr": {"down": 0.0, "low": 458750.0, "high": 947750.0, "diff": 230000.0},
}

#: 异常级别，数值越小优先级越高（§4）。低值与高值同级是文档的原始设定。
ANOMALY_LEVEL: Mapping[str, int] = {
    "lane_down": 0,
    "low_value": 1,
    "high_value": 1,
    "lane_diff": 2,
}

#: 异常检测的短路顺序（§3.1）。命中一个即返回，因此同一指标不可能同时是
#: `low_value` 和 `lane_diff`——这是文档承认的表达能力缺陷，此处如实保留，
#: 因为改掉它就不再是被验证过的那套专家规则了。
ANOMALY_ORDER: Tuple[str, ...] = ("lane_down", "low_value", "high_value", "lane_diff")

#: 单指标模式的基础优先级（§5.3）。
SINGLE_METRIC_BASE: Mapping[str, int] = {
    "host_snr": 2,
    "serdes_snr": 3,
    "media_snr": 4,
    "rxpower": 5,
    "txpower": 6,
}

#: 归因方向：`same` 指向异常所在端，`opposite` 指向异常所在端的对端。
#:
#: 这张表是本模块最有价值的部分，也是迭代 3 要注入 LLM 的核心知识：
#: 接收类观测（`rxpower`、`media_snr`）约束的是**对端的发送链路**，
#: 发送类与本地数字侧观测（`txpower`、`host_snr`、`serdes_snr`）约束的是本端。
SINGLE_METRIC_DIRECTION: Mapping[str, str] = {
    "host_snr": "same",
    "serdes_snr": "same",
    "media_snr": "opposite",
    "rxpower": "opposite",
    "txpower": "same",
}

#: 两条组合模式的方向与优先级（§5.1、§5.2）。
TXPOWER_LANE_DOWN_PRIORITY = "0"
MULTI_METRIC_PRIORITY = "1"
MULTI_METRIC_REQUIRES: Tuple[str, ...] = ("serdes_snr", "media_snr", "rxpower")

#: 兜底裁决（§6.2、§7）。这两个不是匹配规则，是合并阶段的出口。
BOTH_ANOMALY_PRIORITY = "7"
NO_ANOMALY_PRIORITY = "8"


@dataclass(frozen=True)
class ExpertVariant:
    """专家规则的一个可执行变体。存在的唯一理由是消融。

    `docs/EXPERT_EXPERIENCE.md` 给的是一整套规则，而它在本数据上的增益必须能
    拆开归因：是**归因方向**这份知识值钱，还是只要「有异常就报某一端」就够了？
    把方向、告警端解析、兜底出口都做成可开关的字段，就能用同一套代码路径回答
    这个问题，而不是靠复制粘贴出几份实现（那样的差异不可信）。

    默认值逐字对应文档，`DOC_VARIANT` 是正式实验唯一允许使用的配置。
    """

    name: str = "doc"
    single_metric_direction: Mapping[str, str] = None  # type: ignore[assignment]
    multi_metric_direction: str = "opposite"
    txpower_lane_down_direction: str = "same"
    resolve_alarm_side: bool = True
    default_alarm_side: str = "L2"
    #: 关掉后，两端无异常与端口全 down 的 case 返回 `verdict=None`（弃权）
    #: 而不是兜底报本端。用于分离「判别力」与「兜底命中先验」。
    use_fallbacks: bool = True

    def __post_init__(self) -> None:
        if self.single_metric_direction is None:
            object.__setattr__(self, "single_metric_direction", dict(SINGLE_METRIC_DIRECTION))
        unknown = sorted(set(self.single_metric_direction) - set(SINGLE_METRIC_BASE))
        if unknown:
            raise ValueError(f"unknown metrics in direction table: {unknown}")
        for value in (*self.single_metric_direction.values(), self.multi_metric_direction,
                      self.txpower_lane_down_direction):
            if value not in ("same", "opposite"):
                raise ValueError(f"direction must be 'same' or 'opposite': {value}")
        if self.default_alarm_side not in SIDES:
            raise ValueError(f"default_alarm_side must be one of {SIDES}")


DOC_VARIANT = ExpertVariant()


def _opposite(side: str) -> str:
    return SIDES[1] if side == SIDES[0] else SIDES[0]


def alarm_side(pack: EvidencePack, *, default: str = "L2") -> str:
    """把专家规则里的「本端」还原成本数据集的 L1 / L2。

    专家文档的 local / remote 是「告警端 / 对端」，而本数据集已经按端口速率
    归一成 L1（400G）/ L2（200G），两者不是同一套坐标。`alarm_ip_interface`
    与 `link_side_ip_interface_map` 保留了原始告警端，可以逐条还原。

    268 条里 212 条告警在 L2、2 条在 L1、54 条没有可用映射。缺失时退回
    `default`，并在诊断结果里标记 `alarm_side_resolved=False`——这批 case 的
    「本端」兜底等价于报多数告警端，评估时必须单独分桶，不能混进规则命中率。
    """
    interface = pack.context.get("alarm_ip_interface")
    mapping = pack.context.get("link_side_ip_interface_map")
    if isinstance(interface, str) and isinstance(mapping, Mapping):
        for side, value in mapping.items():
            if side in SIDES and value == interface:
                return str(side)
    return default


def _lane_values(pack: EvidencePack, side: str, metric: str) -> List[float]:
    try:
        reading = pack.reading(side, metric)
    except KeyError:
        return []
    return [value for value in reading.lanes.values() if value is not None]


def apply_host_snr_rule(values: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """§2.3：某端 `host_snr` 全部 lane 都没有 `v > 0` 时整项置空。

    这不是「正常」而是「读不到」，约束 C14 已经记过同一件事；专家规则的做法与之一致。
    """
    if not any(value > 0 for value in values.get("host_snr", [])):
        values["host_snr"] = []
    return values


def side_metric_values(pack: EvidencePack, side: str) -> Dict[str, List[float]]:
    """取该端各指标的 lane 读数，并施加 `host_snr` 的特殊后处理（§2.3）。"""
    return apply_host_snr_rule(
        {metric: _lane_values(pack, side, metric) for metric in EXPERT_METRICS}
    )


def detect_side_anomalies(values: Mapping[str, Sequence[float]]) -> Dict[str, str]:
    """对一侧的全部指标跑一遍异常检测，返回 `{指标: 异常类型}`。

    单独抽出来是因为特征抽取层（`features/extractor.py`）拿到的是原始 case 字典
    而不是证据包，两边必须走同一段判定逻辑，否则 token 与规则会各说各话。
    """
    anomalies: Dict[str, str] = {}
    for metric in EXPERT_METRICS:
        kind = detect_anomaly(metric, list(values.get(metric, ())))
        if kind is not None:
            anomalies[metric] = kind
    return anomalies


def detect_anomaly(metric: str, values: Sequence[float]) -> Optional[str]:
    """按 §3.1 / §3.2 的短路顺序返回该指标的唯一异常类型。"""
    if not values:
        return None
    threshold = EXPERT_THRESHOLDS[metric]
    for kind in ANOMALY_ORDER:
        if kind == "lane_down" and any(value <= threshold["down"] for value in values):
            return "lane_down"
        if kind == "low_value" and any(value < threshold["low"] for value in values):
            return "low_value"
        if kind == "high_value" and any(value > threshold["high"] for value in values):
            return "high_value"
        if kind == "lane_diff" and max(values) - min(values) > threshold["diff"]:
            return "lane_diff"
    return None


def port_down_from_values(values: Mapping[str, Sequence[float]]) -> bool:
    """§1 的端口状态：`txpower` 与 `rxpower` 都没有任何 lane 高于 down 阈值。"""
    for metric in ("txpower", "rxpower"):
        lanes = list(values.get(metric, ()))
        if lanes and any(value > EXPERT_THRESHOLDS[metric]["down"] for value in lanes):
            return False
    return True


def port_down(pack: EvidencePack, side: str) -> bool:
    return port_down_from_values(
        {metric: _lane_values(pack, side, metric) for metric in ("txpower", "rxpower")}
    )


@dataclass(frozen=True)
class SideDiagnosis:
    """单端诊断结果：命中的最高优先级模式及其指向。"""

    side: str
    rule: str
    priority: str
    location: str
    anomalies: Tuple[Tuple[str, str], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "rule": self.rule,
            "priority": self.priority,
            "location": self.location,
            "anomalies": {metric: kind for metric, kind in self.anomalies},
        }


@dataclass(frozen=True)
class ExpertDiagnosis:
    """专家规则对一条 case 的裁决。

    `group` 是标定分组键，不是规则名：同一条规则在两端命中应当共享可靠性
    统计，否则 161 条训练 case 会被切成几十个各自不足 10 条的小组，
    Wilson 下界永远过不了门（这正是 MVP 里 M9 全部拒绝的成因）。
    """

    verdict: Optional[str]
    group: str
    priority: str
    reason: str
    alarm_side_resolved: bool
    sides: Tuple[SideDiagnosis, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": EXPERT_RULES_VERSION,
            "verdict": self.verdict,
            "group": self.group,
            "priority": self.priority,
            "reason": self.reason,
            "alarm_side_resolved": self.alarm_side_resolved,
            "sides": [item.to_dict() for item in self.sides],
        }


def side_diagnosis_from_anomalies(
    side: str,
    anomalies: Mapping[str, str],
    *,
    variant: ExpertVariant = DOC_VARIANT,
) -> Optional[SideDiagnosis]:
    """§5 + §6.1：匹配全部模式，取 priority 最小者。

    priority 按文档保持**字符串排序**。当前所有取值都是一位前缀，字符串序与
    数值序等价；文档 §8 已经点出这在优先级涨到两位数时会出错，此处如实保留
    并由测试锁住，改掉它就不再是被现网验证过的那套规则。
    """
    if not anomalies:
        return None

    other = _opposite(side)

    def resolve(direction: str) -> str:
        return side if direction == "same" else other

    candidates: List[Tuple[str, str, str]] = []

    if anomalies.get("txpower") == "lane_down":
        candidates.append(
            (
                TXPOWER_LANE_DOWN_PRIORITY,
                "txpower_lane_down",
                resolve(variant.txpower_lane_down_direction),
            )
        )

    if all(metric in anomalies for metric in MULTI_METRIC_REQUIRES):
        candidates.append(
            (MULTI_METRIC_PRIORITY, "multi_metric", resolve(variant.multi_metric_direction))
        )

    for metric, base in SINGLE_METRIC_BASE.items():
        if metric not in anomalies:
            continue
        priority = f"{base}{ANOMALY_LEVEL[anomalies[metric]]}"
        location = resolve(variant.single_metric_direction[metric])
        candidates.append((priority, f"single:{metric}", location))

    candidates.sort(key=lambda item: item[0])
    priority, rule, location = candidates[0]
    return SideDiagnosis(
        side=side,
        rule=rule,
        priority=priority,
        location=location,
        anomalies=tuple(sorted(anomalies.items())),
    )


def diagnose_side(
    pack: EvidencePack,
    side: str,
    *,
    variant: ExpertVariant = DOC_VARIANT,
) -> Optional[SideDiagnosis]:
    return side_diagnosis_from_anomalies(
        side, detect_side_anomalies(side_metric_values(pack, side)), variant=variant
    )


def diagnose(pack: EvidencePack, *, variant: ExpertVariant = DOC_VARIANT) -> ExpertDiagnosis:
    """§1 + §6.2 的完整流程：端口状态门 -> 两端诊断 -> 合并裁决。"""
    resolved = any(
        isinstance(pack.context.get("link_side_ip_interface_map"), Mapping)
        and pack.context["link_side_ip_interface_map"].get(side) == pack.context.get("alarm_ip_interface")
        for side in SIDES
    )
    local = (
        alarm_side(pack, default=variant.default_alarm_side)
        if variant.resolve_alarm_side
        else variant.default_alarm_side
    )
    remote = _opposite(local)

    local_down, remote_down = port_down(pack, local), port_down(pack, remote)
    if local_down or remote_down:
        # §1：一端已 down 就直接定界到该端；两端都 down 时无法诊断，优先本端。
        verdict = local if local_down else remote
        return ExpertDiagnosis(
            verdict=verdict if variant.use_fallbacks else None,
            group="expert:port_status_gate",
            priority="gate",
            reason=f"端口状态门：{verdict} 侧 txpower 与 rxpower 均无有效发光",
            alarm_side_resolved=resolved,
        )

    diagnoses = tuple(
        item
        for item in (
            diagnose_side(pack, SIDES[0], variant=variant),
            diagnose_side(pack, SIDES[1], variant=variant),
        )
        if item is not None
    )
    ordered = tuple(sorted(diagnoses, key=lambda item: item.priority))

    if not ordered:
        return ExpertDiagnosis(
            verdict=local if variant.use_fallbacks else None,
            group="expert:no_anomaly",
            priority=NO_ANOMALY_PRIORITY,
            reason="两端指标无明显异常，按专家规则兜底反馈本端",
            alarm_side_resolved=resolved,
        )

    if (
        len(ordered) == 2
        and ordered[0].location != ordered[1].location
        and ordered[0].priority == ordered[1].priority
    ):
        return ExpertDiagnosis(
            verdict="fiber",
            group="expert:both_anomaly",
            priority=BOTH_ANOMALY_PRIORITY,
            reason="两端同优先级异常且定界不同，按专家规则判为光纤",
            alarm_side_resolved=resolved,
            sides=ordered,
        )

    best = ordered[0]
    return ExpertDiagnosis(
        verdict=best.location,
        group=f"expert:{best.rule}",
        priority=best.priority,
        reason=f"{best.side} 侧命中 {best.rule}（priority {best.priority}），指向 {best.location}",
        alarm_side_resolved=resolved,
        sides=ordered,
    )


def diagnose_many(
    packs: Sequence[EvidencePack],
    *,
    variant: ExpertVariant = DOC_VARIANT,
) -> Tuple[ExpertDiagnosis, ...]:
    return tuple(diagnose(pack, variant=variant) for pack in packs)


@dataclass(frozen=True)
class ExpertCalibration:
    """专家规则各分组在训练集上的实测可靠性。

    规则给出方向，但不给出「这条方向有多可信」。可靠性必须像分支和 SOP 一样
    从训练标签里统计，否则 M9 无法把三种候选放在同一把尺子上比。

    与 SOP 不同的是，这里**不需要重新拟合规则**：规则本身没有参数，一条 case 的
    诊断结果不受其他 case 影响，唯一的乐观来源是它自己参与了分组频率的统计。
    门限反解要用的折外版本由 `knowledge.out_of_fold_expert_predictions` 给出，
    两者的差值可以直接读出这份乐观有多大。
    """

    counts: Mapping[str, Tuple[int, int]]
    source: str = ""
    version: str = EXPERT_RULES_VERSION

    @classmethod
    def fit(
        cls,
        diagnoses: Sequence[ExpertDiagnosis],
        labels: Sequence[str],
        *,
        source: str = "train",
    ) -> "ExpertCalibration":
        if len(diagnoses) != len(labels):
            raise ValueError("diagnoses and labels must be the same length")
        tally: Dict[str, List[int]] = {}
        for diagnosis, truth in zip(diagnoses, labels):
            if diagnosis.verdict is None:
                continue
            row = tally.setdefault(diagnosis.group, [0, 0])
            row[0] += int(diagnosis.verdict == truth)
            row[1] += 1
        return cls(
            counts={key: (value[0], value[1]) for key, value in sorted(tally.items())},
            source=source,
        )

    def confidence(self, group: str) -> float:
        correct, total = self.counts.get(group, (0, 0))
        return round(correct / total, 6) if total else 0.0

    def lower_bound(self, group: str) -> float:
        correct, total = self.counts.get(group, (0, 0))
        return wilson_lower_bound(correct, total)

    def support(self, group: str) -> int:
        return self.counts.get(group, (0, 0))[1]

    def prediction(self, diagnosis: ExpertDiagnosis) -> Optional[Dict[str, Any]]:
        """产出 M9 候选所需的映射；无标定样本的分组返回下界 0，由门禁自行拒绝。"""
        if diagnosis.verdict is None:
            return None
        group = diagnosis.group
        return {
            "verdict": diagnosis.verdict,
            "confidence": self.confidence(group),
            "confidence_lower_bound": self.lower_bound(group),
            "support": self.support(group),
            "group": group,
            "priority": diagnosis.priority,
            "reason": diagnosis.reason,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "groups": {
                key: {
                    "correct": correct,
                    "total": total,
                    "accuracy": round(correct / total, 6) if total else 0.0,
                    "wilson_lower_bound": wilson_lower_bound(correct, total),
                }
                for key, (correct, total) in sorted(self.counts.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpertCalibration":
        return cls(
            counts={
                str(key): (int(item["correct"]), int(item["total"]))
                for key, item in value.get("groups", {}).items()
            },
            source=str(value.get("source", "")),
            version=str(value.get("version", EXPERT_RULES_VERSION)),
        )
