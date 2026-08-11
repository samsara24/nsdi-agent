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
#: v4 在 v3 的 15 条之上补 5 条，全部针对同一个结构性缺口：
#: v3 里没有任何一条约束允许 `support` L1 或 L2。实测中 LLM 两轮累计 599 条违规里
#: 最高频的一类正是「把 neutral 约束当 support」——模型想给设备侧根因找依据，
#: 但库里根本没有可引的正向约束，于是只能违规。补的五条分成两组：
#: C16/C18 提供可用的归因方向与波及范围推理；C17/C19/C20 是把实测到的**负结果**
#: 写成禁止推断，防止模型在没有判别证据时硬猜。
#: v5 补的两条来自一次候选知识审计（`scripts/mine_knowledge_candidates.py` 挖候选，
#: `scripts/audit_candidate_confounding.py` 做共线性与形态偏差审计）。审计的 8 个候选里
#: 7 个被否掉，剩下 1 个（两端发送功率相减）在控制共线 token 后增益消失。
#: C21 把这个陷阱写成禁止推断——它是本数据集上最容易被误当成信号的量；
#: C22 则是同一轮审计里唯一站得住的正向发现：同侧接收 lane 间不均衡，
#: 是训练集上唯一一个 Wilson 下界超过 L1 先验的观测条件。
CONSTRAINT_LIBRARY_VERSION = "constraint-library-v5"

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
    "attribution_direction",
    "identifiability",
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
            "第二个更直观的证据（迭代 1 补测）：只看序不看数值，统计两端**最差 lane 的编号**"
            "是否相同。若两端编号真的对应，故障 case 里这个比例应明显高于随机；"
            "实测 rxpower 为 37/155 = 23.9%、media_snr 为 46/161 = 28.6%，"
            "而 4 lane 下随机一致的概率就是 25%，两者都与随机无法区分。"
            "这个检验只需要遥测本身，可以直接用来判断一份数据的两端 lane 是否对齐。"
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
    Constraint(
        constraint_id="C16_receive_symptom_constrains_far_transmit_chain",
        category="attribution_direction",
        kind="indicator",
        title="接收侧症状把故障约束在对端发送链路、介质与本端接收链路三者之内",
        physical_statement=(
            "一侧的接收类读数（rxpower、media_snr、RxLOS / RxLOL）度量的是**对端发出、"
            "穿过介质之后到达本端**的光。因此接收侧出现症状时，候选根因只能落在"
            "「对端发送链路」「介质」「本端接收链路」这三段里，"
            "在物理上不可能是本端自己的发送链路——本端发出的光根本不经过本端的接收器。"
            "这条方向性是光链路 RCA 里最基本的归因约束：报症状的一端通常不是肇事的一端。"
        ),
        formal_expression=(
            "receive_symptom(X)  =>  root_cause_chain in {tx_chain(Y), medium, rx_chain(X)}"
            "  AND  root_cause_chain != tx_chain(X)"
        ),
        parameters=(("接收侧症状定义", "RxLOS / RxLOL 告警，或 rxpower / media_snr 存在断 lane"),),
        provenance="derived",
        measured_evidence=(
            "rca_v2_l2fixed manifest train split（161 条）按「只有哪一侧出现接收侧症状」分组："
            "只有 L1 侧 63 条，其中根因为对端 L2 的 43 条 = 68.3%（Wilson 下界 56.0%，"
            "L2 先验 62.1%）；只有 L2 侧 25 条，其中根因为对端 L1 的 12 条 = 48.0%"
            "（Wilson 下界 30.0%，L1 先验 30.4%）。"
            "**两个方向不对称：只有 L1 受害方向的下界超过其预测类别的先验。**"
            "更细的口径同样如此：L1 侧 rx 只有单 lane 断（其余 lane 健康）37 条，"
            "根因 L2 30 条 = 81.1%（下界 65.8%）；镜像条件 L2 侧 rx 单 lane 断只有 13 条，"
            "根因 L1 仅 4 条 = 30.8%，与 L1 先验无法区分。"
        ),
        diagnostic_use=(
            "只允许在 L1 侧为接收受害方时用它支持 L2；"
            "L2 侧为受害方时按 C17 处理，不得镜像套用。"
            "这条不对称本身是实测结果，不要为了对称性把它写成双向规则。"
        ),
        prompt_text=(
            "接收侧读数描述的是对端发出、穿过光纤后到达本端的光，"
            "因此接收侧异常不可能由本端自己的发送器造成，候选只能是对端发送链路、光纤介质或本端接收链路。"
            "在本数据集上，当 L1（400G）侧是接收受害方时，根因落在对端 L2 的实测比例为 81.1%"
            "（L1 侧 rx 单 lane 断，37 条支持），可以据此支持 L2。"
            "反方向不成立，见 C17。"
        ),
        applies_to_token_prefixes=(
            "drop:L1:rxpower:",
            "drop:L1:media_snr:",
            "status:L1:RxLOS",
            "status:L1:RxLOL",
            "level:L1:rxpower_mean:low_tail",
            "level:L1:media_snr_min:low_tail",
            "lane:L2_to_L1:tx_ok_rx_down",
        ),
        allowed_effects=("support",),
        allowed_targets=("L2",),
    ),
    Constraint(
        constraint_id="C17_l2_side_receive_symptom_is_not_discriminative",
        category="attribution_direction",
        kind="caveat",
        title="L2 侧接收症状不足以支持 L1 根因",
        physical_statement=(
            "C16 的方向性在物理上是对称的，但在本数据集上只有一个方向具备统计判别力。"
            "L2（200G）侧作为接收受害方时，对端归因的实测正确率与 L1 的类别先验没有区别，"
            "说明现有遥测无法把「L1 发送链路劣化」与「L2 自身接收链路劣化」分开——"
            "两者在 L2 侧看到的现象一样。"
        ),
        formal_expression=(
            "receive_symptom(L2) AND NOT receive_symptom(L1)"
            "  =>  P(L1) 与先验不可区分；不得据此断言 L1"
        ),
        parameters=(
            ("训练集触发", "25/161"),
            ("对端归因正确率", "12/25 = 48.0%"),
            ("Wilson 95% 下界", "30.0%"),
            ("L1 类别先验", "30.4%"),
        ),
        provenance="measured",
        measured_evidence=(
            "rca_v2_l2fixed manifest train split：只有 L2 侧出现接收症状的 25 条中，"
            "标签为 L1 的 12 条、L2 的 10 条、fiber 的 3 条。"
            "预测 L1 的 Wilson 下界 30.0% 恰好落在 L1 先验 30.4% 上，没有增益。"
            "单 lane 口径更差：L2 侧 rx 单 lane 断 13 条中 L1 仅 4 条（30.8%）。"
        ),
        diagnostic_use=(
            "命中本约束的 case 应输出「候选 L1，但当前证据不足以定论」并给出补采项"
            "（L1 侧 host_snr / serdes 读数、L1 侧同 lane 的发送功率历史），"
            "而不是给出 L1 结论。这是把一个实测负结果变成明确的补采动作。"
            "唯一的例外是 C22：L2 侧各 lane 之间接收不齐（而不是整体偏低）"
            "仍可作为 L1 的弱支持，两者的适用 token 不重叠。"
        ),
        prompt_text=(
            "当只有 L2（200G）侧出现接收侧异常时，不要据此断定根因在 L1。"
            "实测该条件下归因对端的正确率与 L1 的基础比例没有区别，"
            "因为现有遥测分不开「L1 发送劣化」和「L2 自身接收劣化」。"
            "此时应说明证据不足，并请求补采 L1 侧的电口读数与该 lane 的发送功率历史。"
        ),
        applies_to_token_prefixes=(
            "drop:L2:rxpower:",
            "drop:L2:media_snr:",
            "status:L2:RxLOS",
            "status:L2:RxLOL",
            "level:L2:rxpower_mean:low_tail",
            "level:L2:media_snr_min:low_tail",
            "lane:L1_to_L2:tx_ok_rx_down",
        ),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C18_single_lane_scope_does_not_exclude_fiber",
        category="lane_directional_consistency",
        kind="caveat",
        title="单 lane 异常缩小的是共享层，不是介质本身",
        physical_statement=(
            "单条 lane 异常而同端口其余 lane 健康，可以排除所有 lane 共享的部分："
            "模块供电、壳体温度、整束光纤被拔出、整个连接器脱落。"
            "但**不能排除介质**：并行光模块的每条 lane 走独立纤芯，"
            "单根纤芯断裂或单个 MPO 芯位脏污同样只影响一条 lane。"
            "把「单 lane」直接推成「不是光纤」是一个很自然但错误的推理。"
        ),
        formal_expression=(
            "down_lane_count == 1  =>  排除 port 级共享原因；"
            "不得推出 root_cause != fiber"
        ),
        parameters=(("训练集单 lane 断 case 数", "50（L1 侧 37 + L2 侧 13）"),),
        provenance="measured",
        measured_evidence=(
            "rca_v2_l2fixed manifest train split 中 rx 恰好一条 lane 断的 case 共 50 条，"
            "其中 fiber 标签 6 条 = 12.0%，**高于** fiber 全局先验 7.45%。"
            "如果单 lane 能排除介质，这个比例应当低于先验。"
        ),
        diagnostic_use=(
            "单 lane 结论只能写成「排除端口级共享原因」，"
            "后续仍要在「对端该通道的激光器」「该 lane 的纤芯 / 芯位」「本端该通道的探测器」之间区分。"
        ),
        prompt_text=(
            "只有一条 lane 异常时，可以排除模块供电、温度、整束光纤脱落这类所有 lane 共享的原因，"
            "但不能排除光纤：并行模块每条 lane 走独立纤芯，单芯断裂或单个芯位脏污也只影响一条 lane。"
            "实测单 lane 组里 fiber 占 12.0%，高于全局 7.45%。"
        ),
        applies_to_token_prefixes=("drop:",),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C19_population_prior_is_not_case_evidence",
        category="measurement_validity",
        kind="caveat",
        title="类别先验与 SOP 叶节点分布不是本 case 的物理证据",
        physical_statement=(
            "learned SOP 的叶节点标签分布、历史候选的标签投票和类别先验都是**群体统计**。"
            "它们可以决定在没有判别证据时的默认动作，但它们不描述当前这条链路发生了什么。"
            "把它们当成证据会产生一种特别难发现的错误：结论看起来有依据，"
            "实际上整条推理链没有引用任何一个当前 case 的观测。"
        ),
        formal_expression=(
            "SOP_leaf_distribution, class_prior, history_label_vote"
            "  NOT IN cited_evidence  AND  effect != support"
        ),
        parameters=(("训练集 L2 先验", "100/161 = 62.1%"),),
        provenance="derived",
        measured_evidence=(
            "MVP 正式实验中 M9 前的 44 个候选有 23 个正确（52.3%），"
            "而单纯预测多数类 L2 在同一测试集上是 62.6%。"
            "也就是说，一条大量引用群体先验的推理链的表现低于直接报多数类，"
            "它增加的只是解释的外观。"
        ),
        diagnostic_use=(
            "M7 应拒绝把 SOP 路径、叶节点分布或历史标签投票写进 `cited_evidence` 的回答；"
            "允许在自然语言里提到它是默认动作的来源，但不允许作为 support 步骤。"
        ),
        prompt_text=(
            "learned SOP 的路径与叶节点标签分布、历史 case 的标签投票、类别先验都属于群体统计，"
            "不是当前 case 的物理证据。不要把它们写进 cited_evidence，也不要用它们作为 support 步骤的依据。"
            "每一个 support 步骤都必须引用当前证据包里真实存在的观测 token。"
        ),
        applies_to_token_prefixes=(),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C20_fiber_not_identifiable_from_current_telemetry",
        category="identifiability",
        kind="caveat",
        title="现有遥测无法识别 fiber 根因",
        physical_statement=(
            "介质根因需要的证据是链路损耗、反射事件位置、端面污染或弯曲损耗，"
            "这些都要靠 OTDR、端面镜检或双向功率标定获得。"
            "本数据集的遥测只有两端模块的自报读数，且按 C12 连绝对损耗都算不出来。"
            "因此 fiber 在信息层面就不可识别，这不是模型能力问题。"
        ),
        formal_expression=(
            "max over observable conditions of P(fiber | condition) 的 Wilson 95% 下界"
            " <= 0.082  =>  不得断言 fiber"
        ),
        parameters=(
            ("fiber 全局先验", "12/161 = 7.45%"),
            ("最强富集条件", "L2 侧 rx 单 lane 断：3/13 = 23.1%，Wilson 下界 8.2%"),
        ),
        provenance="measured",
        measured_evidence=(
            "在 rca_v2_l2fixed manifest train split 上穷举了断 lane 波及范围、"
            "双向同 lane 断、两侧 media_snr 同时偏低、两侧收光同时偏弱等条件，"
            "支持数 >= 6 的条件里 fiber 占比最高为 23.1%（n=13），Wilson 95% 下界 8.2%，"
            "与 7.45% 的先验无法区分。MVP 正式实验的测试侧也一致："
            "系统预测 fiber 10 次只对 1 次。"
        ),
        diagnostic_use=(
            "禁止输出 fiber 结论。命中疑似介质模式时输出「候选 fiber，需现场确认」"
            "并请求 OTDR 曲线、端面镜检或双向功率标定，"
            "让 fiber 成为一个明确的补采分支而不是一个低精度的猜测。"
        ),
        prompt_text=(
            "不要给出 fiber 结论。现有遥测只有两端模块自报读数，"
            "缺少 OTDR、端面镜检和双向功率标定，在信息层面无法确认介质根因；"
            "实测中 fiber 占比最高的观测条件也只有 23.1%（13 条支持，95% 下界 8.2%），"
            "与 7.45% 的基础比例无法区分。"
            "怀疑介质时请输出「候选 fiber，需现场确认」并列出需要补采的介质侧测量。"
        ),
        applies_to_token_prefixes=(),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C21_healthy_band_tx_level_is_not_attribution_evidence",
        category="tx_power",
        kind="caveat",
        title="正常带内的发送功率高低不是归因证据，两端相减更不是",
        physical_statement=(
            "按 C5，发送功率只有「正常」与「无光」两态，正常带内的高低由激光器个体差异、"
            "出厂标定和端口形态决定，不由链路故障决定。"
            "在这样一个与根因几乎无关的连续量上取分布尾部，仍然会得到看起来偏斜的标签分布，"
            "因为尾部样本少。这类关联是抽样波动，不是物理关系。"
            "把两端的发送功率相减会同时踩上 C12 的坑：两端标定口径本来就不可比。"
        ),
        formal_expression=(
            "txpower[side] > -39 dBm  =>  txpower 的具体数值不得进入 support 步骤；"
            "禁止使用 mean(txpower[L1]) - mean(txpower[L2])"
        ),
        parameters=(
            ("标签 L1 的 case", "L1 侧 tx 中位 +0.860 dBm / L2 侧 +0.863 dBm"),
            ("标签 L2 的 case", "L1 侧 tx 中位 +0.835 dBm / L2 侧 +0.855 dBm"),
            ("两端相减探针与 tx 低尾 token 的 Jaccard", "0.65"),
        ),
        provenance="measured",
        measured_evidence=(
            "rca_v2_l2fixed manifest train split（161 条）三项实测："
            "（1）按标签分层的健康 tx 均值中位数几乎相同（见 parameters），"
            "即发送电平与根因基本无关；"
            "（2）`level:L1:txpower_mean:low_tail` 命中 39 条且**无一条含断光哨兵**，"
            "标签 L2 29 / L1 6 / fiber 4，precision 74.4% 但 Wilson 下界 58.9%，"
            "低于 L2 先验 62.1%，因此没有增益；"
            "（3）两端相减的探针 `probe:txpower_side_gap:L1_worse` 是唯一下界（65.8%）"
            "超过 L2 先验的 tx 类信号，但它与上面那个低尾 token 的 Jaccard 达 0.65，"
            "控制该 token 后剩余支持只有 7 条，增益消失。"
        ),
        diagnostic_use=(
            "发送侧一律只做有光 / 无光判断（C5、C6）。"
            "不允许出现「L1 侧发送功率偏低所以……」这类步骤，也不允许两端功率相减。"
            "这条约束的作用是拦掉一条统计上很诱人、物理上站不住的捷径。"
        ),
        prompt_text=(
            "发送光功率在正常带（-1.8~2.1 dBm）内的高低不是故障证据："
            "实测按根因分层的发送功率中位数几乎相同，正常带内的差异来自器件个体与标定。"
            "不要写「某侧发送功率偏低」，也不要把两端发送功率相减，"
            "发送侧只判断有光还是无光。"
        ),
        applies_to_token_prefixes=(
            "level:L1:txpower_mean:",
            "level:L2:txpower_mean:",
        ),
        allowed_effects=("neutral",),
        allowed_targets=("",),
    ),
    Constraint(
        constraint_id="C22_receive_lane_imbalance_indicates_far_transmit_array",
        category="lane_directional_consistency",
        kind="indicator",
        title="同侧接收 lane 间不均衡指向对端发送阵列的通道差异",
        physical_statement=(
            "并行光模块的每条 lane 有独立的激光器与探测器，但同一端口内所有 lane 共享"
            "标定口径、整束光纤的共模损耗和接收侧的 AGC 配置。"
            "因此同侧各 lane 接收功率之间的**极差**天然消掉了这些共模项，"
            "剩下的差异只能来自对端各发送通道之间的不一致，即对端发送阵列的通道级劣化。"
            "这使它比两端绝对电平相减可靠得多——后者按 C12 在本数据集上根本不成立。"
            "「用同侧相对量做跨端归因」是这份数据里唯一站得住的跨端推理方式。"
        ),
        formal_expression=(
            "spread(rxpower[X]) 显著大于同侧正常波动  =>  support tx_array(Y)，Y 为对端"
        ),
        parameters=(
            ("L2 侧接收不均衡命中", "7/161"),
            ("其中根因为对端 L1", "6/7 = 85.7%"),
            ("Wilson 95% 下界", "48.7%"),
            ("L1 类别先验", "30.4%"),
        ),
        provenance="measured",
        measured_evidence=(
            "rca_v2_l2fixed manifest train split：`imbalance:L2:rxpower` 命中 7 条，"
            "标签为 L1 的 6 条、fiber 1 条，无一条含断光哨兵（即不均衡不是断 lane 造成的）。"
            "Wilson 下界 48.7% 超过 L1 先验 30.4%，这是全训练集上**唯一**一个"
            "下界超过 L1 先验的观测条件。"
            "镜像方向 `imbalance:L1:rxpower` 命中 10 条、8 条为 L2（80.0%，下界 49.0%），"
            "但 L2 先验是 62.1%，所以镜像方向不成立。"
            "这个不对称主要来自两类先验相差一倍（30.4% vs 62.1%）——"
            "支持少数类需要的证据强度本来就更低，不要读成物理上的不对称。"
        ),
        diagnostic_use=(
            "只允许用它支持 L1，且必须标注为弱证据：支持数只有 7 条，下界 48.7% 意味着"
            "「比先验强，但远达不到可以定论」。命中后应输出 L1 候选并请人工确认，"
            "同时建议补采 L1 侧各发送通道的功率与偏置电流历史，用来确认是哪一路通道。"
            "本条是 C17 的细化，不是推翻：C17 否掉的是「L2 侧整体收光低或告警」，"
            "本条针对的是「L2 侧各 lane 之间不齐」，两者不可混用。"
        ),
        prompt_text=(
            "同一侧各 lane 之间的接收功率不齐（极差偏大），指向对端发送阵列中某几路通道的差异，"
            "因为同侧 lane 共享标定与整束光纤的共模损耗，极差把这些共模项消掉了。"
            "实测中 L2 侧接收不均衡的 7 条里有 6 条根因在对端 L1（95% 下界 48.7%，L1 基础比例 30.4%），"
            "可以据此支持 L1，但只能作为弱证据并请人工确认；"
            "反方向（L1 侧不均衡支持 L2）不成立。"
        ),
        applies_to_token_prefixes=("imbalance:L2:rxpower",),
        allowed_effects=("support",),
        allowed_targets=("L1",),
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
