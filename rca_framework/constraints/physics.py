"""Pure physical constraints for optical-link RCA.

This layer intentionally excludes train-set ranges, label distributions, and
Wilson lower bounds.  Those belong to the numeric decision tree.  The only
numeric parameters allowed here are device or topology constants.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple


PHYSICS_LIBRARY_VERSION = "physics-constraints-v3-causal-direction"
PHYSICS_PROVENANCES: Tuple[str, ...] = ("device_spec", "derived")
PHYSICS_EFFECTS: Tuple[str, ...] = ("support", "exclude", "neutral")
PHYSICS_TARGETS: Tuple[str, ...] = ("L1", "L2", "fiber", "")

# Device/spec constants are acceptable in the pure-physics layer.  Train-set
# quantiles such as 7.2-7.8 mA or Wilson percentages are not.
_ALLOWED_PARAMETER_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"-?39(?:\.0)?\s*dBm", re.IGNORECASE),
    re.compile(r"0\s*-\s*70\s*(?:°C|degC|C)", re.IGNORECASE),
    re.compile(r"3\.3\s*V", re.IGNORECASE),
    re.compile(r"±\s*5%"),
    re.compile(r"bias\s*==\s*0", re.IGNORECASE),
    re.compile(r"0\s*mA", re.IGNORECASE),
)


@dataclass(frozen=True)
class PhysicalConstraint:
    constraint_id: str
    title: str
    statement: str
    formal_expression: str
    diagnostic_use: str
    prompt_text: str
    provenance: str
    parameters: Tuple[Tuple[str, str], ...] = ()
    source_constraint_ids: Tuple[str, ...] = ()
    applies_to_token_prefixes: Tuple[str, ...] = ()
    allowed_effects: Tuple[str, ...] = ("neutral",)
    allowed_targets: Tuple[str, ...] = ("",)
    review_status: str = "pending_expert_review"

    def __post_init__(self) -> None:
        if self.provenance not in PHYSICS_PROVENANCES:
            raise ValueError(f"physical constraint provenance must be device_spec or derived: {self.provenance}")
        unknown_effects = sorted(set(self.allowed_effects) - set(PHYSICS_EFFECTS))
        if unknown_effects:
            raise ValueError(f"unknown physical effects: {unknown_effects}")
        unknown_targets = sorted(set(self.allowed_targets) - set(PHYSICS_TARGETS))
        if unknown_targets:
            raise ValueError(f"unknown physical targets: {unknown_targets}")
        for name, value in self.parameters:
            text = f"{name} {value}"
            if not any(pattern.search(text) for pattern in _ALLOWED_PARAMETER_PATTERNS):
                raise ValueError(
                    "pure physical constraints may not carry train-set fitted parameters: "
                    f"{self.constraint_id} parameter {name}={value!r}"
                )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["parameters"] = [list(item) for item in self.parameters]
        return value


@dataclass(frozen=True)
class PhysicalConstraintLibrary:
    version: str
    constraints: Tuple[PhysicalConstraint, ...]

    def get(self, constraint_id: str) -> PhysicalConstraint:
        for item in self.constraints:
            if item.constraint_id == constraint_id:
                return item
        raise KeyError(f"unknown physical constraint: {constraint_id}")

    def ids(self) -> Tuple[str, ...]:
        return tuple(item.constraint_id for item in self.constraints)

    def by_source(self, old_constraint_id: str) -> Tuple[PhysicalConstraint, ...]:
        return tuple(item for item in self.constraints if old_constraint_id in item.source_constraint_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "constraints": [item.to_dict() for item in self.constraints],
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


PHYSICAL_CONSTRAINTS: Tuple[PhysicalConstraint, ...] = (
    PhysicalConstraint(
        constraint_id="P1_bias_zero_means_laser_off",
        title="偏置电流为零等价于激光器未点亮",
        statement="激光器光输出由偏置电流驱动；偏置电流为 0 mA 时该 lane 必然没有光输出。",
        formal_expression="bias[side][lane] == 0  <=>  txpower[side][lane] <= -39 dBm",
        parameters=(("断光哨兵", "-39 dBm"), ("零电流判定", "bias == 0 mA")),
        provenance="device_spec",
        diagnostic_use="把没有发出光与光发出后丢失分开；前者不能归因于介质衰减。",
        prompt_text="偏置电流为 0 mA 表示激光器未点亮，该 lane 没有光输出。",
        source_constraint_ids=("C1_bias_zero_means_laser_off",),
        applies_to_token_prefixes=("drop:L1:bias:", "drop:L2:bias:"),
    ),
    PhysicalConstraint(
        constraint_id="P2_temperature_spec_band",
        title="模块温度工作规格",
        statement="商用光模块工作温度范围为 0-70 degC；规格内温度不足以单独解释链路中断。",
        formal_expression="0 degC <= Temperature[side] <= 70 degC",
        parameters=(("工作温度范围", "0-70 degC"),),
        provenance="device_spec",
        diagnostic_use="温度越界是数据质量或设备环境告警；规格内不能作为热致根因。",
        prompt_text="模块温度在 0-70 degC 内时，不要把根因归为过温。",
        source_constraint_ids=("C3_temperature_operating_range",),
    ),
    PhysicalConstraint(
        constraint_id="P3_voltage_nominal_band",
        title="模块供电规格",
        statement="光模块标称供电 3.3 V，允许偏差 ±5%；越界时设备侧需确认。",
        formal_expression="3.135 V <= Voltage[side] <= 3.465 V",
        parameters=(("标称电压", "3.3 V"), ("允许偏差", "±5%")),
        provenance="device_spec",
        diagnostic_use="供电越界只说明该侧设备是候选，不能从训练集少量样本推出统计规则。",
        prompt_text="供电电压应在 3.3 V ±5% 内；低于下界时该侧设备需人工确认。",
        source_constraint_ids=("C4_voltage_nominal_band",),
    ),
    PhysicalConstraint(
        constraint_id="P4_tx_has_light_or_no_light",
        title="发送侧只区分有光与无光",
        statement="发送光功率由本端激光器输出决定；断光哨兵表示无光，正常带内高低不描述链路衰减。",
        formal_expression="txpower[side][lane] <= -39 dBm => no_light; otherwise => emitted_light",
        parameters=(("断光哨兵", "-39 dBm"),),
        provenance="derived",
        diagnostic_use="发送侧只判断是否发光，不按正常带内的发送电平高低做归因。",
        prompt_text="发送侧只区分有光与无光；正常带内发送功率高低不是归因证据。",
        source_constraint_ids=("C5_tx_power_range", "C21_healthy_band_tx_level_is_not_attribution_evidence"),
        applies_to_token_prefixes=("drop:L1:txpower:", "drop:L2:txpower:", "level:L1:txpower_mean:", "level:L2:txpower_mean:"),
    ),
    PhysicalConstraint(
        constraint_id="P5_tx_down_excludes_medium",
        title="本端未发光时排除介质根因",
        statement="光纤只能衰减已经进入它的光，不能解释一束从未被发出的光。",
        formal_expression="txpower[near][lane] <= -39 dBm => root_cause != fiber",
        parameters=(("断光哨兵", "-39 dBm"),),
        provenance="derived",
        diagnostic_use="确定性排除 fiber；前提是量测契约未判定全链路哨兵语义翻转。",
        prompt_text="如果某侧 lane 根本没有发出光，该方向故障不可能由光纤引起。",
        source_constraint_ids=("C6_tx_down_excludes_medium",),
        applies_to_token_prefixes=("drop:L1:txpower:", "drop:L2:txpower:", "lane:L1_to_L2:tx_down", "lane:L2_to_L1:tx_down"),
        allowed_effects=("exclude",),
        allowed_targets=("fiber",),
    ),
    PhysicalConstraint(
        constraint_id="P6_rx_has_continuous_degradation",
        title="接收侧存在真实连续劣化区间",
        statement="接收功率等于对端发送功率减去链路与接收链路损伤，因此可讨论偏低程度。",
        formal_expression="rxpower[far] = txpower[near] - path_loss - receiver_effects",
        provenance="derived",
        diagnostic_use="接收侧可以讨论功率偏低程度；这与发送侧的有光/无光二值判断不同。",
        prompt_text="接收光功率存在真实连续劣化区间，可以讨论偏低程度。",
        source_constraint_ids=("C7_rx_power_range",),
        applies_to_token_prefixes=("drop:L1:rxpower:", "drop:L2:rxpower:", "level:L1:rxpower_mean:", "level:L2:rxpower_mean:", "imbalance:L1:rxpower", "imbalance:L2:rxpower"),
    ),
    PhysicalConstraint(
        constraint_id="P7_tx_ok_rx_down_means_path_loss",
        title="已发光但对端收不到表示路径中丢失",
        statement="本端已发出的光没有到达对端，故障候选落在介质、连接器或对端接收链路。",
        formal_expression="txpower[near][lane] > -39 dBm AND rxpower[far][lane] <= -39 dBm => path_loss_or_far_receiver",
        parameters=(("断光哨兵", "-39 dBm"),),
        provenance="derived",
        diagnostic_use="只能缩小候选范围，不能单独把 fiber 定为根因。",
        prompt_text=(
            "本端发光正常而对端同 lane 无光，说明光在路径或对端接收链路中丢失。"
            "候选优先取对端接收链路所在端（L1->L2 方向取 L2，L2->L1 方向取 L1）；"
            "只有在同一 lane 双向同时异常（P8）时才可把 target 写成 fiber。"
        ),
        source_constraint_ids=("C8_tx_ok_rx_down_indicates_medium",),
        applies_to_token_prefixes=("lane:L1_to_L2:tx_ok_rx_down", "lane:L2_to_L1:tx_ok_rx_down"),
        allowed_effects=("support",),
        allowed_targets=("fiber", "L1", "L2"),
    ),
    PhysicalConstraint(
        constraint_id="P8_bidirectional_symmetry_points_shared_path",
        title="双向对称异常指向共享路径",
        statement="同一 lane 双向同时异常说明共享的纤芯、连接器或路径部分被共同影响；单向异常更偏端点链路。",
        formal_expression="abnormal(L1->L2,lane) AND abnormal(L2->L1,lane) => shared_path_candidate",
        provenance="device_spec",
        diagnostic_use="与单向路径丢失一起判断介质/连接器是否应升为候选。",
        prompt_text="同一 lane 双向同时异常时，问题偏向共享的光纤对或连接器。",
        source_constraint_ids=("C9_bidirectional_symmetry",),
        applies_to_token_prefixes=("lane:L1_to_L2:bidirectional_same_lane", "lane:L2_to_L1:bidirectional_same_lane"),
        allowed_effects=("support",),
        allowed_targets=("fiber",),
    ),
    PhysicalConstraint(
        constraint_id="P9_scope_all_lanes_vs_single_lane",
        title="全 lane 与单 lane 指向不同故障层级",
        statement="端口内所有 lane 共享供电、时钟、模块壳体和纤束；单 lane 有独立激光器、探测器和纤芯。",
        formal_expression="down_lane_count == all => shared_layer; down_lane_count == 1 => channel_layer",
        provenance="device_spec",
        diagnostic_use="决定排障粒度：端口级共享部分或通道级独立部分。",
        prompt_text="全 lane 同时异常指向共享层；单 lane 异常指向通道级独立部分。",
        source_constraint_ids=("C10_all_lanes_vs_single_lane",),
        applies_to_token_prefixes=("drop:",),
    ),
    PhysicalConstraint(
        constraint_id="P10_receive_symptom_points_to_far_transmit_chain",
        title="接收侧症状约束根因方向",
        statement="一侧接收类读数度量的是对端发出、穿过介质后到达本端的光，不可能由本端发送器造成。",
        formal_expression="receive_symptom(X) => root_cause_chain in {tx_chain(Y), medium, rx_chain(X)}",
        provenance="derived",
        diagnostic_use="解释 rxpower、media_snr、RxLOS/RxLOL 时先把症状侧翻译成候选根因侧。",
        prompt_text=(
            "接收侧异常度量的是对端发出并经过介质到达本端的光，因此只能把故障范围约束在"
            "对端发送链、链路介质和本端接收链三者内；单独使用本约束时写 neutral/空target。"
            "只有另有对端Tx故障、本端接收链佐证或P8双向介质证据时，才可支持唯一标签。"
        ),
        source_constraint_ids=(
            "C16_receive_symptom_constrains_far_transmit_chain",
            "C23_expert_receive_anomaly_on_l1_supports_l2",
            "C24_expert_receive_anomaly_on_l2_supports_l1",
        ),
        # 只绑定接收类症状。禁止用笼统的 drop:/status:/level: 前缀，否则会把
        # 发送侧断光、TxLOS、正常带内 txpower 分档误判成「接收症状关键证据」。
        applies_to_token_prefixes=(
            "drop:L1:rxpower:",
            "drop:L2:rxpower:",
            "drop:L1:media_snr:",
            "drop:L2:media_snr:",
            "status:L1:RxLOS",
            "status:L1:RxLOL",
            "status:L2:RxLOS",
            "status:L2:RxLOL",
            "level:L1:rxpower_mean:",
            "level:L2:rxpower_mean:",
            "level:L1:media_snr_min:",
            "level:L2:media_snr_min:",
            "expert:L1:rxpower:",
            "expert:L2:rxpower:",
            "expert:L1:media_snr:",
            "expert:L2:media_snr:",
        ),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    PhysicalConstraint(
        constraint_id="P11_single_lane_does_not_exclude_fiber",
        title="单 lane 异常不能排除单纤芯介质问题",
        statement="并行模块每条 lane 走独立纤芯；单芯断裂或单个芯位脏污也只影响一条 lane。",
        formal_expression="down_lane_count == 1 => not port_shared_cause; does_not_imply root_cause != fiber",
        provenance="device_spec",
        diagnostic_use="单 lane 只能排除端口级共享原因，不能用来排除 fiber。",
        prompt_text="只有一条 lane 异常时，不能排除光纤介质；单根纤芯也可能只影响一条 lane。",
        source_constraint_ids=("C18_single_lane_scope_does_not_exclude_fiber",),
        applies_to_token_prefixes=("drop:",),
    ),
    PhysicalConstraint(
        constraint_id="P12_receive_lane_imbalance_removes_common_mode",
        title="同侧接收 lane 极差消掉共模项",
        statement="同侧 lane 共享标定口径和共模损耗，接收功率极差主要反映通道间差异。",
        formal_expression="spread(rxpower[X]) => channel_imbalance_after_common_mode_removed",
        provenance="derived",
        diagnostic_use="把同侧相对量与绝对两端相减区分开，后者由量测契约禁止。",
        prompt_text="同侧接收 lane 间极差可作为通道级不均衡线索，但其统计可靠性由决策树决定。",
        source_constraint_ids=("C22_receive_lane_imbalance_indicates_far_transmit_array",),
        applies_to_token_prefixes=("imbalance:L1:rxpower", "imbalance:L2:rxpower"),
        allowed_effects=("support",),
        allowed_targets=("L1", "L2"),
    ),
    PhysicalConstraint(
        constraint_id="P13_local_signal_metrics_point_local",
        title="发送与电口读数量度本端信号",
        statement="txpower、host_snr 与 serdes_snr 描述本端产生或处理的信号，不是对端发来的光。",
        formal_expression="local_signal_symptom(X) => candidate includes local_device(X)",
        provenance="derived",
        diagnostic_use="解释发送、电口、SerDes 类异常时指向本端；可靠性由规则组统计决定。",
        prompt_text=(
            "发送光功率和SerDes异常度量本端产生或处理的信号，可支持异常所在端。"
            "host_snr只有在实际观测到有效异常时才作为同方向增强证据；缺失不扣分、不要求补采，"
            "且host_snr不得单独支撑最终标签。"
        ),
        source_constraint_ids=(
            "C25_expert_local_chain_anomaly_on_l1_supports_l1",
            "C26_expert_local_chain_anomaly_on_l2_is_not_discriminative",
        ),
        applies_to_token_prefixes=("expert:L1:txpower:", "expert:L2:txpower:", "expert:L1:host_snr:", "expert:L2:host_snr:", "expert:L1:serdes_snr:", "expert:L2:serdes_snr:"),
        allowed_effects=("support",),
        allowed_targets=("L1", "L2"),
    ),
)


PHYSICS_LIBRARY = PhysicalConstraintLibrary(
    version=PHYSICS_LIBRARY_VERSION,
    constraints=PHYSICAL_CONSTRAINTS,
)


def render_physics_prompt_block(
    library: PhysicalConstraintLibrary = PHYSICS_LIBRARY,
    *,
    constraints: Sequence[PhysicalConstraint] | None = None,
) -> str:
    selected = list(library.constraints if constraints is None else constraints)
    lines = [f"# 纯物理约束（{library.version}，hash {library.content_hash()}）", ""]
    for item in selected:
        token_scope = "、".join(item.applies_to_token_prefixes) if item.applies_to_token_prefixes else "无需绑定当前 token"
        targets = "、".join("空字符串" if target == "" else target for target in item.allowed_targets)
        lines.append(
            f"- [{item.constraint_id}] {item.prompt_text}\n"
            f"  结构化引用契约：可用 token 前缀={token_scope}；"
            f"effect 只能为 {'、'.join(item.allowed_effects)}；target 只能为 {targets}。"
        )
    return "\n".join(lines).strip() + "\n"


def iter_physics_constraints(
    library: PhysicalConstraintLibrary = PHYSICS_LIBRARY,
) -> Iterable[PhysicalConstraint]:
    return iter(library.constraints)
