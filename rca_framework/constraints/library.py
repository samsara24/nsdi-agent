"""M5 物理约束库 v1。

每条约束都必须能回答三个问题：它断言的物理关系是什么、这个关系的参数从哪来、
它在诊断里怎么用。第三点由 `kind` 决定：

- `invariant`：器件物理上必然成立的关系。违反它说明数据本身有问题，不是故障证据。
- `exclusion`：命中后可以**排除**某个根因。排除条件比正向指示更可靠，优先用。
- `indicator`：命中后**提高**某个根因的可能性，但不足以定论。
- `caveat`：告诉推理者某个看起来合理的推断在本数据集上不成立，防止 LLM 自由发挥。

`provenance` 记录参数来源，这是约束库能不能被专家审核的前提：

- `device_spec`：来自光模块 / 电气规范的通用值，与本数据集无关。
- `measured`：在本仓库训练集上实测得到的区间，随数据集版本变化。
- `derived`：由其它约束推导，不引入新参数。

`review_status` 目前全部为 `pending_expert_review`，等夏思博审核后逐条改为 `approved`。
未经审核的约束可以进 prompt，但必须在 prompt 里标注为待审。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple


#: v3 保留 v2 的 15 条约束，但把实测口径切到 `rca_v2_l2fixed`，并补充
#: token/effect 契约，供 M7 从结构化约束读取适用范围。
#: v2 相对 v1 只增加了 `C15`。它是 T5 在跑「确定性排除是否排掉过真实标签」这项
#: 全量校验时发现的：`C6` 在 14 次触发里排掉了 2 次真实的 fiber 标签，
#: 追下去发现根因不是 `C6` 的物理表述有错，而是断光哨兵在全链路失效时含义会翻转。
CONSTRAINT_LIBRARY_VERSION = "constraint-library-v3"

CONSTRAINT_KINDS: Tuple[str, ...] = ("invariant", "exclusion", "indicator", "caveat")
PROVENANCES: Tuple[str, ...] = ("device_spec", "measured", "derived")
REVIEW_STATUSES: Tuple[str, ...] = ("pending_expert_review", "approved", "rejected")
CATEGORIES: Tuple[str, ...] = (
    "bias_current",
    "temperature",
    "voltage",
    "tx_power",
    "rx_power",
    "signal_quality",
    "lane_directional_consistency",
    "measurement_validity",
)

#: 实测数据集口径。`measured` 类参数全部来自这一份切分，换数据集必须重测。
MEASURED_ON = "rca_v2_l2fixed manifest train split（seed=42，train_ratio=0.6）"


@dataclass(frozen=True)
class Constraint:
    constraint_id: str
    category: str
    kind: str
    title: str
    physical_statement: str
    formal_expression: str
    parameters: Tuple[Tuple[str, str], ...]
    provenance: str
    measured_evidence: str
    diagnostic_use: str
    prompt_text: str
    review_status: str = "pending_expert_review"
    applies_to_token_prefixes: Tuple[str, ...] = ()
    allowed_effects: Tuple[str, ...] = ("neutral",)
    allowed_targets: Tuple[str, ...] = ("",)

    def __post_init__(self) -> None:
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(f"unknown constraint kind: {self.kind}")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"unknown provenance: {self.provenance}")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"unknown review status: {self.review_status}")
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category: {self.category}")
        unknown_effects = sorted(set(self.allowed_effects) - {"support", "exclude", "neutral"})
        if unknown_effects:
            raise ValueError(f"unknown allowed effects: {unknown_effects}")
        unknown_targets = sorted(set(self.allowed_targets) - {"L1", "L2", "fiber", ""})
        if unknown_targets:
            raise ValueError(f"unknown allowed targets: {unknown_targets}")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["parameters"] = [list(item) for item in self.parameters]
        return value


@dataclass(frozen=True)
class ConstraintLibrary:
    version: str
    constraints: Tuple[Constraint, ...]
    measured_on: str = MEASURED_ON

    def get(self, constraint_id: str) -> Constraint:
        for constraint in self.constraints:
            if constraint.constraint_id == constraint_id:
                return constraint
        raise KeyError(f"unknown constraint: {constraint_id}")

    def by_category(self, category: str) -> Tuple[Constraint, ...]:
        return tuple(item for item in self.constraints if item.category == category)

    def by_kind(self, kind: str) -> Tuple[Constraint, ...]:
        return tuple(item for item in self.constraints if item.kind == kind)

    def ids(self) -> Tuple[str, ...]:
        return tuple(item.constraint_id for item in self.constraints)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "measured_on": self.measured_on,
            "constraints": [item.to_dict() for item in self.constraints],
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


_CONSTRAINTS: Tuple[Constraint, ...] = (
    Constraint(
        constraint_id="C1_bias_zero_means_laser_off",
        category="bias_current",
        kind="invariant",
        title="偏置电流为零等价于该 lane 未发光",
        physical_statement=(
            "激光器的光输出由偏置电流驱动。偏置电流为 0 mA 时器件根本没有被点亮，"
            "该 lane 必然没有光输出，这与光纤链路的状态无关。"
        ),
        formal_expression="bias[side][lane] == 0  <=>  txpower[side][lane] <= -39 dBm",
        parameters=(("断光哨兵", "-39 dBm"), ("零电流判定", "bias == 0 mA")),
        provenance="device_spec",
        measured_evidence=(
            "训练集 1008 个 lane 读数中有 45 个 bias == 0，涉及 8 条 case；"
            "这 45 个 lane 的 txpower 全部同时处于断光哨兵，无一例外，"
            "反向也成立：没有出现 tx 断光而 bias 非零的 lane。"
        ),
        diagnostic_use=(
            "先用它把「没发出光」与「光发出后丢失」分开。前者的根因在发送端设备，"
            "后者才可能是介质。这一步必须在任何功率比较之前做。"
        ),
        prompt_text=(
            "偏置电流为 0 mA 的 lane 表示激光器未点亮，该 lane 没有光输出。"
            "此时该方向的问题在发送端设备，不能归因于光纤。"
        ),
        applies_to_token_prefixes=("drop:L1:bias:", "drop:L2:bias:"),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C2_bias_healthy_band",
        category="bias_current",
        kind="indicator",
        title="健康 lane 的偏置电流落在窄带内",
        physical_statement=(
            "同型号模块在正常工作点上的偏置电流分布很窄。电流显著抬高通常意味着"
            "激光器老化后需要更大驱动才能维持同样光功率，是器件劣化的早期信号。"
        ),
        formal_expression="7.2 mA <= bias[side][lane] <= 7.8 mA  (healthy lane)",
        parameters=(("下界", "7.2 mA"), ("上界", "7.8 mA")),
        provenance="measured",
        measured_evidence=(
            "训练集非零偏置电流的 p25 = 7.22 mA（L1）/ 7.29 mA（L2），"
            "p99 = 7.72 mA（L1）/ 7.78 mA（L2），全部落在 7.2-7.8 mA。"
        ),
        diagnostic_use=(
            "本数据集内没有观察到落在该带之外的非零电流，因此它当前只能作为"
            "「电流正常」的排除依据，不能作为老化判据。合并数据集后重新标定。"
        ),
        prompt_text=(
            "健康 lane 的偏置电流在 7.2-7.8 mA。电流在此范围内说明激光器驱动正常，"
            "不要把根因归到发送端器件老化。"
        ),
        applies_to_token_prefixes=("drop:L1:bias:", "drop:L2:bias:"),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C3_temperature_operating_range",
        category="temperature",
        kind="exclusion",
        title="模块温度在工作范围内则排除热致故障",
        physical_statement=(
            "商用光模块的工作温度范围是 0-70 °C。温度在范围内时，"
            "温漂不足以导致链路中断或降 lane。"
        ),
        formal_expression="0 degC <= Temperature[side] <= 70 degC",
        parameters=(("下界", "0 degC"), ("上界", "70 degC")),
        provenance="device_spec",
        measured_evidence=(
            "训练集 252 个温度读数全部落在 39.35-60.13 °C，无一超出 70 °C。"
            "L2 侧系统性高于 L1 侧约 3.5 °C（中位数 50.30 vs 46.72），"
            "这是 200G 与 400G 端口的形态差异，不是故障。"
        ),
        diagnostic_use=(
            "在本数据集上温度恒为排除条件：不允许把根因写成过温。"
            "L1 / L2 之间的固定温差也不能当作两侧差异证据。"
        ),
        prompt_text=(
            "模块温度在 0-70 °C 内属于正常工作范围，不构成故障原因。"
            "本数据集所有 case 的温度都在此范围内，因此不要把根因归为过温。"
            "L2 侧温度普遍比 L1 侧高约 3.5 °C 是端口形态差异，不是异常。"
        ),
        # v2 没有温度 token；该约束只能作为不绑定 evidence token 的中性上下文引用。
        applies_to_token_prefixes=(),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C4_voltage_nominal_band",
        category="voltage",
        kind="exclusion",
        title="供电电压在 3.3 V ±5% 内则排除供电异常",
        physical_statement=(
            "光模块标称供电 3.3 V，允许偏差 ±5%，即 3.135-3.465 V。"
            "超出下界时激光器驱动与 DSP 都可能不稳定。"
        ),
        formal_expression="3.135 V <= Voltage[side] <= 3.465 V",
        parameters=(("标称", "3.3 V"), ("允许偏差", "±5%")),
        provenance="device_spec",
        measured_evidence=(
            "训练集 252 个电压读数中只有 1 个低于下界（case_aa307cc7c7db 的 L2 侧 3.10 V），"
            "其余全部落在 3.14-3.42 V。"
        ),
        diagnostic_use=(
            "命中越界时把该侧设备列为候选并要求人工确认；"
            "由于全训练集只有 1 例，不允许据此建立统计规则。"
        ),
        prompt_text=(
            "供电电压应在 3.135-3.465 V。落在此范围内则排除供电导致的故障；"
            "低于下界时该侧设备是候选根因，但需要人工确认。"
        ),
        # v2 没有电压 token；不能借任意光学 token 把它提升为 L1/L2 排除结论。
        applies_to_token_prefixes=(),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C5_tx_power_range",
        category="tx_power",
        kind="invariant",
        title="发送光功率的量程与断光哨兵",
        physical_statement=(
            "单 lane 发送光功率由激光器输出决定，正常工作点集中在 0 dBm 附近的窄带；"
            "读数掉到 -39 dBm 及以下是「无光」的哨兵值，不是一个真实的功率测量。"
        ),
        formal_expression="txpower[side][lane] > -39 dBm  =>  -1.8 dBm <= txpower <= 2.1 dBm",
        parameters=(("断光哨兵", "-39 dBm"), ("健康下界", "-1.8 dBm"), ("健康上界", "2.1 dBm")),
        provenance="measured",
        measured_evidence=(
            "训练集健康 txpower 共 963 个读数，L1 侧 -1.70~1.91 dBm，L2 侧 -1.73~2.08 dBm；"
            "没有介于 -39 dBm 与 -1.8 dBm 之间的中间值，说明发送功率是「要么正常要么无光」。"
        ),
        diagnostic_use=(
            "发送功率不存在渐变劣化区间，因此 tx 侧只需判断有光 / 无光，"
            "不要对 tx 做「偏低多少 dB」的推断。"
        ),
        prompt_text=(
            "发送光功率正常时在 -1.8~2.1 dBm，异常时直接掉到 -39 dBm 哨兵值，"
            "两者之间没有中间态。因此发送侧只区分有光与无光，不要讨论发送功率轻微偏低。"
        ),
        applies_to_token_prefixes=(
            "drop:L1:txpower:",
            "drop:L2:txpower:",
            "level:L1:txpower_mean:",
            "level:L2:txpower_mean:",
        ),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C6_tx_down_excludes_medium",
        category="tx_power",
        kind="exclusion",
        title="本端未发光时排除介质根因",
        physical_statement=(
            "光纤只能衰减已经进入它的光，不能解释一束从未被发出的光。"
            "本端某 lane 没有光输出时，该方向的问题必然在发送端。"
        ),
        formal_expression="txpower[near][lane] <= -39 dBm  =>  root_cause != fiber (for that direction)",
        parameters=(("断光哨兵", "-39 dBm"),),
        provenance="derived",
        measured_evidence=(
            "由 C1 与 C5 推出，不引入新参数。训练集中 tx 断光的 lane 与 bias == 0 完全重合。"
        ),
        diagnostic_use="这是排除 fiber 的最强单条依据，应在 N5c 推理的第一步执行。",
        prompt_text=(
            "如果某一侧的 lane 根本没有发出光（txpower 处于 -39 dBm 哨兵），"
            "那么该方向的故障不可能由光纤引起，应归到该发送端设备。"
        ),
        applies_to_token_prefixes=(
            "drop:L1:txpower:",
            "drop:L2:txpower:",
            "lane:L1_to_L2:tx_down",
            "lane:L2_to_L1:tx_down",
        ),
        allowed_effects=("exclude",),
        allowed_targets=("fiber",),
    ),
    Constraint(
        constraint_id="C7_rx_power_range",
        category="rx_power",
        kind="invariant",
        title="接收光功率的量程与断光哨兵",
        physical_statement=(
            "接收功率等于对端发送功率减去链路衰减，因此它有真实的连续劣化区间；"
            "读数掉到 -39 dBm 及以下同样是「无光」哨兵。"
        ),
        formal_expression="rxpower[side][lane] > -39 dBm  =>  -12.3 dBm <= rxpower <= 3.0 dBm",
        parameters=(("断光哨兵", "-39 dBm"), ("实测下界", "-12.3 dBm"), ("实测上界", "3.0 dBm")),
        provenance="measured",
        measured_evidence=(
            "训练集健康 rxpower 共 929 个读数，L1 侧 -12.15~2.95 dBm，L2 侧 -12.25~2.83 dBm；"
            "与 tx 不同，rx 存在连续的低功率区间（p1 为 -8.45 / -4.68 dBm）。"
        ),
        diagnostic_use=(
            "接收侧允许讨论「偏低多少」，这是与发送侧的关键区别，"
            "也是 `level_tail` 特征家族只在接收侧有判别力的物理原因。"
        ),
        prompt_text=(
            "接收光功率有真实的连续劣化区间（本数据集健康值 -12.3~3.0 dBm），"
            "低于 -39 dBm 表示完全无光。接收侧可以讨论功率偏低的程度，发送侧不可以。"
        ),
        applies_to_token_prefixes=(
            "drop:L1:rxpower:",
            "drop:L2:rxpower:",
            "imbalance:L1:rxpower",
            "imbalance:L2:rxpower",
            "level:L1:rxpower_mean:",
            "level:L2:rxpower_mean:",
        ),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C8_tx_ok_rx_down_indicates_medium",
        category="lane_directional_consistency",
        kind="indicator",
        title="本端发光正常而对端同 lane 无光指向介质或对端接收",
        physical_statement=(
            "光已经被发出却没有到达对端，说明它在传输路径上丢失。"
            "路径包含光纤、连接器、以及对端的接收器件。"
        ),
        formal_expression=(
            "txpower[near][lane] > -39 dBm AND rxpower[far][lane] <= -39 dBm"
            "  =>  root_cause in {fiber, far_end_device}"
        ),
        parameters=(("断光哨兵", "-39 dBm"),),
        provenance="derived",
        measured_evidence=(
            "全量 211 条 case 中有 61 条命中该模式，标签分布 L2 40 / L1 13 / fiber 8。"
            "fiber 在命中组中的占比 13.1%，高于全局占比 6.6%，约 2 倍富集；"
            "但它同时命中 53 条非 fiber case，因此只能作为 indicator 而不是判据。"
        ),
        diagnostic_use=(
            "命中后必须继续区分「介质」与「对端接收器件」，"
            "单靠这一条不能判 fiber。区分手段见 C9。"
        ),
        prompt_text=(
            "如果本端某 lane 发光正常而对端同一 lane 完全收不到光，说明光在路径中丢失，"
            "候选根因是光纤介质或对端接收器件。这条线索会提高光纤的可能性，但不足以定论，"
            "必须再结合双向一致性判断。"
        ),
        applies_to_token_prefixes=(
            "lane:L1_to_L2:tx_ok_rx_down",
            "lane:L2_to_L1:tx_ok_rx_down",
        ),
        # 该模式不能区分“对端接收器件”和介质；在三分类契约里只允许作 fiber 弱支持。
        allowed_effects=("support",),
        allowed_targets=("fiber",),
    ),
    Constraint(
        constraint_id="C9_bidirectional_symmetry",
        category="lane_directional_consistency",
        kind="indicator",
        title="双向对称异常指向介质，单向异常指向该方向的端点",
        physical_statement=(
            "一对光纤中的两根分别承载两个方向。同一 lane 双向同时异常说明整条 lane "
            "或其光纤对被共同影响（例如同一根尾纤被拔出、同一个连接器脏污）；"
            "只有单向异常说明问题落在该方向的发送端或接收端，而不是共享的介质。"
        ),
        formal_expression=(
            "abnormal(L1->L2, lane) AND abnormal(L2->L1, lane)  =>  shared cause (fiber pair / connector)"
            "; XOR  =>  endpoint of that direction"
        ),
        parameters=(),
        provenance="device_spec",
        measured_evidence=(
            "全量 211 条中同 lane 双向异常的有 9 条（标签 L2 6 / fiber 2 / L1 1），"
            "样本量太小，不足以支撑统计结论；这条约束的依据是物理拓扑而非数据。"
        ),
        diagnostic_use=(
            "与 C8 串联使用：C8 判断光是否在路径中丢失，C9 判断丢失是否双向对称。"
            "双向对称才把 fiber 提到首位。"
        ),
        prompt_text=(
            "同一条 lane 在两个方向上同时异常，说明问题出在双向共享的部分，"
            "即光纤对或连接器，光纤是首位候选。"
            "如果只有一个方向异常，则问题在该方向的端点设备，光纤应当降位。"
        ),
        applies_to_token_prefixes=(
            "lane:L1_to_L2:bidirectional_same_lane",
            "lane:L2_to_L1:bidirectional_same_lane",
        ),
        allowed_effects=("support",),
        allowed_targets=("fiber",),
    ),
    Constraint(
        constraint_id="C10_all_lanes_vs_single_lane",
        category="lane_directional_consistency",
        kind="indicator",
        title="全 lane 同时异常与单 lane 异常指向不同层级",
        physical_statement=(
            "一个端口的所有 lane 共享供电、时钟、模块壳体和同一束光纤；"
            "单条 lane 则有独立的激光器、探测器和纤芯。"
            "所有 lane 同时异常指向共享层，单 lane 异常指向该通道的独立器件。"
        ),
        formal_expression="down_lane_count == lane_count  =>  port-level; down_lane_count == 1  =>  channel-level",
        parameters=(),
        provenance="device_spec",
        measured_evidence=(
            "这正是特征家族 `signal_drop` 把断 lane 数分成 single_lane / partial_lanes / all_lanes "
            "三档的物理依据；T1 家族消融显示该分档是 v1 中不可替代的一项。"
        ),
        diagnostic_use="决定排障动作的粒度：整端口换模块 / 换整束纤，还是单通道定位。",
        prompt_text=(
            "端口内所有 lane 同时异常，指向端口级共享部分：模块本体、供电、或整束光纤。"
            "只有一条 lane 异常，指向该通道独立的激光器、探测器或单根纤芯。"
        ),
        applies_to_token_prefixes=("drop:",),
        # 波及粒度不能在 L1/L2/fiber 三类中唯一定位，只允许解释 token。
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C11_media_snr_floor",
        category="signal_quality",
        kind="indicator",
        title="介质侧信噪比显著低于正常带且收光正常时指向链路质量",
        physical_statement=(
            "介质侧 SNR 反映解调后的信号质量。收光功率正常而 SNR 偏低，"
            "说明损伤不是功率衰减，而是色散、反射、串扰这类不改变总功率的链路质量问题。"
        ),
        formal_expression="rxpower normal AND media_snr < 22.5 dB  =>  link quality degradation",
        parameters=(("正常带下界", "22.5 dB"), ("正常带中位数", "25.6 dB (L1) / 26.0 dB (L2)")),
        provenance="measured",
        measured_evidence=(
            "训练集健康 media_snr 的 p1 为 22.47 dB（L1）/ 22.95 dB（L2），"
            "低于 20 dB 的读数只有 4 个（16.71 / 17.51 / 17.51 / 17.70）。"
        ),
        diagnostic_use=(
            "触发极少，当前只能作为个别 case 的补充线索；"
            "不要把它写进任何需要统计支撑的规则。"
        ),
        prompt_text=(
            "介质侧信噪比正常范围约 22.5-27 dB。如果收光功率正常但信噪比明显低于该范围，"
            "问题偏向链路质量（色散、反射、连接器端面），而不是功率衰减。"
        ),
        applies_to_token_prefixes=(
            "level:L1:media_snr_min:low_tail",
            "level:L2:media_snr_min:low_tail",
        ),
        allowed_effects=("support",),
        allowed_targets=("fiber",),
    ),
    Constraint(
        constraint_id="C12_no_absolute_link_loss",
        category="measurement_validity",
        kind="caveat",
        title="本数据集不能用两端功率相减计算链路损耗",
        physical_statement=(
            "无源链路的损耗必然非负，即对端收到的功率不可能高于本端发出的功率。"
            "本数据集违反这一点，说明两端 lane 编号不对应，或收发功率的标定口径不同。"
        ),
        formal_expression="mean(txpower[near]) - mean(rxpower[far]) >= 0   # 本数据集不成立",
        parameters=(),
        provenance="measured",
        measured_evidence=(
            "按 lane 号配对后，L1->L2 方向的均值损耗中位数为 -0.285 dB，"
            "L2->L1 为 -0.227 dB，两个方向的中位数都是负值，物理上不可能。"
            "legacy `directional_loss` 学到的上界（3.11 / 3.42 dB）因此也不可信。"
        ),
        diagnostic_use=(
            "禁止在约束、规则或 prompt 中写绝对损耗门限。"
            "只允许使用同侧内部的相对量（lane 间极差）与训练集分位分档。"
        ),
        prompt_text=(
            "本数据集两端的光功率读数不能直接相减求链路损耗：实测结果会出现负损耗，"
            "说明两端 lane 编号不对应或标定口径不同。"
            "不要根据「损耗多少 dB」下结论，只使用同侧 lane 之间的相对差异和有光/无光判断。"
        ),
        applies_to_token_prefixes=("lane:",),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C13_serdes_snr_unit_unknown",
        category="measurement_validity",
        kind="caveat",
        title="serdes_snr 不是 dB 量纲，不得按信噪比解释",
        physical_statement=(
            "`serdes_snr` 字段的健康取值在 6.6e5-8.3e5 量级，断链时为 1。"
            "这不是任何 dB 口径的信噪比，更接近某种原始计数或定点数。"
        ),
        formal_expression="serdes_snr 量纲未知；仅可用作有效 / 无效的二值判断",
        parameters=(("健康区间", "约 6.6e5 - 8.3e5（量纲未知）"), ("失效哨兵", "1")),
        provenance="measured",
        measured_evidence=(
            "训练集 972 个 serdes_snr 读数中，健康值 p25-p99 为 6.6e5-8.2e5，"
            "最小值为 1。legacy 规则里出现频率很高的 `low_outlier:*:serdes_snr` "
            "就建立在这个量纲未知的字段上。"
        ),
        diagnostic_use=(
            "在向厂商确认量纲之前，只允许用它区分「有效」与「失效」，"
            "不允许出现「serdes SNR 低了 x dB」这类表述。"
        ),
        prompt_text=(
            "serdes_snr 字段的量纲未确认，健康值在 6.6e5-8.3e5 量级，失效时为 1。"
            "只能把它当作有效 / 失效的二值信号，不要按 dB 信噪比解释或比较。"
        ),
        applies_to_token_prefixes=("serdes:",),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C14_host_snr_mostly_missing",
        category="measurement_validity",
        kind="caveat",
        title="host_snr 在多数 case 上缺失，缺失不等于正常",
        physical_statement=(
            "主机侧信噪比反映模块与交换芯片之间的电口质量，与光链路无关。"
            "该字段在本数据集大面积缺失。"
        ),
        formal_expression="host_snr 存在率 = 52/161 训练 case",
        parameters=(("训练集存在率", "52/161（32.3%）"),),
        provenance="measured",
        measured_evidence=(
            "rca_v2_l2fixed manifest train split 的 161 条 case 中，"
            "只有 52 条任一侧有非空 host_snr 读数，109 条两侧均无有效读数（存在率 32.3%）。"
        ),
        diagnostic_use=(
            "缺失必须被显式表述为「未采集」，不能当作「正常」。"
            "N6 在判断证据充分性时要把它算作缺失证据而不是通过项。"
        ),
        prompt_text=(
            "主机侧信噪比 host_snr 在多数 case 中没有采集。"
            "看不到该字段时应说明「未采集」，不要推断它正常，也不要用它支持任何结论。"
        ),
        applies_to_token_prefixes=("telemetry:partial_telemetry", "telemetry:no_telemetry"),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C15_blackout_sentinel_is_not_laser_off",
        category="measurement_validity",
        kind="caveat",
        title="全链路读数同时触底时，哨兵表示「未读到数」而不是「无光」",
        physical_statement=(
            "两端的发送、接收、介质侧信噪比同时全部处于断光哨兵，"
            "同时 TxLOS 却报 Normal，这两件事互相矛盾：模块若真的没有发光，TxLOS 应当告警。"
            "更合理的解释是链路整体中断后遥测通道本身失效，所有光学读数回落到哨兵默认值。"
            "此时哨兵是「读不到」而不是「没有光」。"
        ),
        formal_expression=(
            "ALL(txpower, rxpower over both sides) <= -39 dBm AND TxLOS == Normal"
            "  =>  哨兵含义为 no_reading，不得据此推断激光关断"
        ),
        parameters=(("断光哨兵", "-39 dBm"), ("训练集命中", "4/161")),
        provenance="measured",
        measured_evidence=(
            "rca_v2_l2fixed manifest train split 的 161 条中有 4 条命中："
            "两侧 4 个 lane 的 txpower / rxpower 全为 -40.0 dBm，"
            "状态位一律 TxLOS=Normal、TxLOL=Normal、RxLOS=Abnormal、RxLOL=Abnormal。"
            "4 条的标签为 L2 3 条、fiber 1 条：物理观测完全一致而根因不同，"
            "说明该状态下的遥测不足以区分根因。"
        ),
        diagnostic_use=(
            "这是 C6 的前置条件：只有在遥测确实有效（存在任一非哨兵读数）时，"
            "才允许用「本端未发光」去排除介质根因。命中本约束的 case 应直接转人工，"
            "不论它产出了多少个特征 token——token 多不等于证据强。"
        ),
        prompt_text=(
            "如果两端的发送与接收光功率全部处于 -39 dBm 哨兵，而 TxLOS 仍报 Normal，"
            "说明这是遥测整体失效而不是激光关断。此时不要断言任何一端「没有发光」，"
            "也不要据此排除光纤，应当说明证据不足并请求现场确认。"
        ),
        applies_to_token_prefixes=(
            "drop:L1:txpower:all_lanes",
            "drop:L2:txpower:all_lanes",
            "drop:L1:rxpower:all_lanes",
            "drop:L2:rxpower:all_lanes",
        ),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
)


CONSTRAINT_LIBRARY = ConstraintLibrary(
    version=CONSTRAINT_LIBRARY_VERSION,
    constraints=_CONSTRAINTS,
)


def render_prompt_block(
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
    *,
    categories: Sequence[str] | None = None,
    constraints: Sequence["Constraint"] | None = None,
) -> str:
    """把约束库渲染成可直接注入 prompt 的文本块。

    渲染顺序固定为 exclusion -> caveat -> invariant -> indicator：
    先给能排除的，再给不许推的，最后才给提高可能性的，
    避免 LLM 在还没排除掉不可能选项时就开始加权猜测。

    `constraints` 用于只注入与当前 case 相关的那几条（M8 会这么用）；
    不给时按 `categories` 从库里选，都不给则注入全部。
    """
    selected = [
        item for item in (library.constraints if constraints is None else constraints)
        if categories is None or item.category in categories
    ]
    order = {"exclusion": 0, "caveat": 1, "invariant": 2, "indicator": 3}
    selected.sort(key=lambda item: (order[item.kind], item.constraint_id))

    lines = [f"# 光模块物理约束（{library.version}，hash {library.content_hash()}）", ""]
    current_kind = ""
    headings = {
        "exclusion": "## 排除条件：命中后可以直接排除对应根因",
        "caveat": "## 禁止推断：以下推断在本数据集上不成立",
        "invariant": "## 物理恒等关系",
        "indicator": "## 倾向性线索：提高可能性，但不足以定论",
    }
    for item in selected:
        if item.kind != current_kind:
            current_kind = item.kind
            lines += ["", headings[current_kind], ""]
        suffix = "（待专家审核）" if item.review_status == "pending_expert_review" else ""
        token_scope = (
            "、".join(item.applies_to_token_prefixes)
            if item.applies_to_token_prefixes
            else "无需绑定当前 token（仅作中性上下文）"
        )
        targets = "、".join("空字符串" if target == "" else target for target in item.allowed_targets)
        lines.append(
            f"- [{item.constraint_id}] {item.prompt_text}{suffix}\n"
            f"  结构化引用契约：可用 token 前缀={token_scope}；"
            f"effect 只能为 {'、'.join(item.allowed_effects)}；target 只能为 {targets}。"
        )
    return "\n".join(lines).strip() + "\n"


def iter_constraints(library: ConstraintLibrary = CONSTRAINT_LIBRARY) -> Iterable[Constraint]:
    return iter(library.constraints)
