"""M1 特征字典 v1。

本文件只做声明，不做抽取。每个 `FeatureFamily` 描述一族特征的物理含义、单位、
取值域和抽取规则；`rca_framework/features/extractor.py` 负责按这份声明产出
token。两者用 `content_hash()` 绑定：字典内容一改，hash 就变，实验产物里记录
的版本号因此可以真正区分不同特征集合。

设计约束（来自 AGENTS.md 第 5.1 / 5.3 节）：

- 特征必须是稀疏、可解释、带物理含义的离散 token，不允许黑盒 embedding。
- 每个 token 的字符串形式就是它的解释，可以直接进 prompt 与报告。
- 连续量不直接进 signature；只有"偏离训练集中心区域"这一事实进 signature，
  分位edge 由训练集拟合并写入 `FeatureModel`，避免手写魔法门限。
- 不改写 legacy `anomaly_id`。`legacy_anomaly` 家族是对 legacy 口径的只读复用。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple


FEATURE_DICTIONARY_VERSION = "feature-dictionary-v1"
FEATURE_DICTIONARY_V2_VERSION = "feature-dictionary-v2"

#: 连续量分档采用训练集分位数，只把两侧尾部编码为 token，中间区间不产出 token。
#: 0.25 / 0.75 是默认边界；改这两个数会改变 `content_hash()`。
TAIL_QUANTILES: Tuple[float, float] = (0.25, 0.75)

#: 断光哨兵。与 `anomaly.DOWN_THRESHOLDS` 同源，重复声明是为了让字典自洽可读。
DOWN_SENTINEL_DBM = -39.0


#: 家族的准入状态。`v1` 是冻结进特征字典 v1 的家族，`candidate` 是已实现但未通过
#: T1 选型的家族：它们保留在代码里供消融与后续数据集重测，但不进 v1 signature。
FAMILY_STATUSES: Tuple[str, ...] = ("v1", "v2", "candidate")


@dataclass(frozen=True)
class FeatureFamily:
    """一族特征的完整声明。

    `token_template` 用 `{}` 占位符描述 token 的字面结构，抽取器必须严格按它拼串，
    否则同一份字典会产出两种不兼容的 signature。
    """

    name: str
    dimension: str
    physical_meaning: str
    unit: str
    value_domain: Tuple[str, ...]
    extraction_rule: str
    token_template: str
    sparsity: str
    tier: str
    status: str = "v1"
    selection_note: str = ""
    reuses_legacy: bool = False

    def __post_init__(self) -> None:
        if self.status not in FAMILY_STATUSES:
            raise ValueError(f"family status must be one of {FAMILY_STATUSES}: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["value_domain"] = list(self.value_domain)
        return value


@dataclass(frozen=True)
class FeatureDictionary:
    version: str
    families: Tuple[FeatureFamily, ...]
    notes: Tuple[str, ...] = ()

    def family_names(self) -> Tuple[str, ...]:
        return tuple(family.name for family in self.families)

    def get(self, name: str) -> FeatureFamily:
        for family in self.families:
            if family.name == name:
                return family
        raise KeyError(f"unknown feature family: {name}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "tail_quantiles": list(TAIL_QUANTILES),
            "down_sentinel_dbm": DOWN_SENTINEL_DBM,
            "families": [family.to_dict() for family in self.families],
            "notes": list(self.notes),
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def subset(self, names: Sequence[str], *, version_suffix: str = "") -> "FeatureDictionary":
        keep = [self.get(name) for name in names]
        version = f"{self.version}{version_suffix}" if version_suffix else f"{self.version}+{'-'.join(names)}"
        return FeatureDictionary(version=version, families=tuple(keep), notes=self.notes)


_FAMILIES: Tuple[FeatureFamily, ...] = (
    FeatureFamily(
        name="signal_drop",
        dimension="每侧每指标的断光/断链 lane 数分档",
        physical_meaning=(
            "光模块某一侧的某个指标有多少条 lane 掉到断光哨兵值。"
            "1 条 lane 断通常是单通道器件或单芯问题，全部 lane 断通常是整端口或整束光纤问题，"
            "两者的排障动作完全不同，legacy 把它们压成同一个 token 是分辨率损失的主要来源之一。"
        ),
        unit="lane 计数（无量纲）",
        value_domain=("single_lane", "partial_lanes", "all_lanes"),
        extraction_rule=(
            f"取该侧该指标全部 lane 原始值，统计 <= 断光哨兵（{DOWN_SENTINEL_DBM} dBm / SNR 0）的 lane 数 d 与总 lane 数 n。"
            "d==0 不产出 token；d==1 产出 single_lane；0<d<n 且 d>1 产出 partial_lanes；d==n 产出 all_lanes。"
        ),
        token_template="drop:{side}:{metric}:{bucket}",
        sparsity="只在存在断光 lane 时产出",
        tier="core",
    ),
    FeatureFamily(
        name="status_fault",
        dimension="每侧 LOS / LOL 告警位",
        physical_meaning=(
            "TxLOS / TxLOL / RxLOS / RxLOL 是光模块自身上报的失锁与失光标志位。"
            "它是设备侧唯一不依赖阈值拟合的一等硬证据，指明哪一端、发送还是接收出了问题。"
        ),
        unit="布尔",
        value_domain=("TxLOS", "TxLOL", "RxLOS", "RxLOL"),
        extraction_rule="该侧状态字段落在 {abnormal, down, fault, error, los, lol, true, 1} 集合内则产出 token。",
        token_template="status:{side}:{status_key}",
        sparsity="只在异常时产出",
        tier="core",
        reuses_legacy=True,
    ),
    FeatureFamily(
        name="fence_outlier",
        dimension="每侧每指标相对训练集稳健围栏的越界方向",
        physical_meaning=(
            "健康 lane 的取值是否越过训练集 IQR 稳健围栏。偏低指向衰减或器件老化，"
            "偏高指向发送功率过冲或接收端未加衰减。"
        ),
        unit="dBm（光功率）/ dB（SNR）",
        value_domain=("low", "high"),
        extraction_rule=(
            "只用健康 lane（> 断光哨兵）。min < Q1-3*IQR 产出 low；max > Q3+3*IQR 产出 high。"
            "围栏由 `anomaly.fit_thresholds` 在训练集上拟合。"
        ),
        token_template="fence:{side}:{metric}:{direction}",
        sparsity="只在越界时产出",
        tier="core",
        status="candidate",
        selection_note=(
            "3 倍 IQR 稳健围栏在本数据集上几乎不触发，且触发时与 level_tail 高度重合。"
            "T1 家族消融中它没有进入任何一个满足 train 侧约束的子集，因此不进 v1。"
        ),
        reuses_legacy=True,
    ),
    FeatureFamily(
        name="lane_imbalance",
        dimension="每侧每指标健康 lane 的极差是否超限",
        physical_meaning=(
            "同一端口内不同 lane 之间的差异。整端口共用同一光源与同一段光纤，"
            "lane 间极差偏大说明问题定位在单通道而不是整链路。"
        ),
        unit="dB",
        value_domain=("over_learned_spread",),
        extraction_rule="健康 lane 极差 > 训练集拟合的 spread 上界时产出。",
        token_template="imbalance:{side}:{metric}",
        sparsity="只在超限时产出",
        tier="core",
        reuses_legacy=True,
    ),
    FeatureFamily(
        name="lane_direction",
        dimension="同 lane 收发配对的方向性 signature",
        physical_meaning=(
            "把本端发送与对端接收按 lane 号配对后的方向性判断。"
            "tx_ok_rx_down 表示本端发得出去而对端收不到，是介质侧最强的指向；"
            "tx_down 表示光根本没发出来，根因在发送端而不在光纤；"
            "bidirectional_same_lane 表示同一 lane 双向都断，指向整条 lane 或其光纤对。"
            "legacy 的 `directional_loss` 先过滤断光 lane 再取均值，把这三种模式全部抹平，因此从未触发。"
        ),
        unit="布尔 signature",
        value_domain=(
            "tx_ok_rx_down",
            "tx_down",
            "bidirectional_same_lane",
            "uniform_loss_all_lanes",
            "single_lane_outlier",
        ),
        extraction_rule="`anomaly.lane_directional_loss` 在 L1->L2 与 L2->L1 两个方向各算一次，命中即产出 token。",
        token_template="lane:{direction}:{signature}",
        sparsity="只在命中时产出",
        tier="core",
        status="candidate",
        selection_note=(
            "触发率不低（211 条中 102 条命中至少一个 signature），但在 T1 家族消融里，"
            "把它加进 v1 组合会同时降低 train LOO 的 N5a 桶内准确率与全集准确率。"
            "结论是它在当前 organized 数据集上不具备增量判别力，保留为 candidate，"
            "等合并数据集到位后按同一套 train-LOO 规则重测，不据此下最终结论。"
        ),
    ),
    FeatureFamily(
        name="level_tail",
        dimension="每侧关键连续量相对训练集分位数的尾部档位",
        physical_meaning=(
            "收/发光功率与介质侧 SNR 的绝对水平。稳健围栏只抓极端离群，"
            "但 fiber 类 case 的收光功率是系统性偏低而非离群，因此需要按训练集分位数分档。"
            "低档指向链路衰减，高档指向功率过冲。"
        ),
        unit="dBm（rxpower / txpower）/ dB（media_snr）",
        value_domain=("low_tail", "high_tail"),
        extraction_rule=(
            f"取健康 lane 的均值（SNR 取最小值），与训练集该统计量的 {TAIL_QUANTILES[0]} / {TAIL_QUANTILES[1]} "
            "分位数比较。低于下分位产出 low_tail，高于上分位产出 high_tail，落在中间不产出 token。"
            "分位边界在训练集上拟合并写入 FeatureModel，不使用手写常数。"
        ),
        token_template="level:{side}:{statistic}:{bucket}",
        sparsity="约 50% 的 case 在每个统计量上产出 token",
        tier="core",
    ),
    FeatureFamily(
        name="side_asymmetry",
        dimension="L1 与 L2 之间同名统计量的差异方向",
        physical_meaning=(
            "两端同一物理量的落差。链路两端共享同一段光纤，正常情况下收光水平应当接近；"
            "某一端系统性更低说明问题偏向该端的接收链路或该方向的光纤，而不是全链路。"
        ),
        unit="dB",
        value_domain=("L1_lower", "L2_lower"),
        extraction_rule=(
            "计算 L1 与 L2 的同名统计量之差，绝对值超过训练集该差值的上分位时，"
            "按较低的一侧产出 token。"
        ),
        token_template="asym:{statistic}:{lower_side}_lower",
        sparsity="约 25% 的 case 产出",
        tier="core",
        status="candidate",
        selection_note=(
            "与 level_tail 由同一批统计量派生，信息高度冗余；加入后 signature 迅速唯一化，"
            "N5a 桶塌到 10 条以下，无法支撑分支结论。"
        ),
    ),
    FeatureFamily(
        name="port_width",
        dimension="两端当前工作 lane 数组合",
        physical_meaning=(
            "`Lane number` 是端口当前实际工作的 lane 数。本数据集的主力告警就是"
            "「接口降 lane」，因此降到几条 lane、两端是否同步降，直接反映故障波及范围。"
            "0x0 表示两端都已完全不可用。"
        ),
        unit="lane 计数（无量纲）",
        value_domain=("0", "1", "2", "4", "unknown"),
        extraction_rule="直接读 `Lane number` 的 L1 / L2 值，缺失记 unknown，始终产出一个 token。",
        token_template="width:{L1_width}x{L2_width}",
        sparsity="每个 case 恒定产出 1 个 token",
        tier="context",
        status="candidate",
        selection_note=(
            "恒定产出使它把每个 case 都推向唯一 signature，同时自身判别力接近类别先验。"
            "作为 N5c 的 prompt 上下文比作为 signature 维度更合适。"
        ),
    ),
    FeatureFamily(
        name="alarm_kind",
        dimension="触发告警的类型",
        physical_meaning=(
            "告警名区分「接口降 lane」「频繁升降 lane」「linkflap」。"
            "持续降 lane 与反复抖动对应不同的物理机制：前者偏向稳定劣化，后者偏向接触不良或间歇性失锁。"
        ),
        unit="类别",
        value_domain=("lane_degrade", "lane_flap", "link_flap", "other"),
        extraction_rule="按 `alarm_name` 关键字归一到四个类别，始终产出一个 token。",
        token_template="alarm:{kind}",
        sparsity="每个 case 恒定产出 1 个 token",
        tier="context",
        status="candidate",
        selection_note=(
            "211 条 case 只有 3 个取值，且每个取值的标签分布都接近全局先验，"
            "单独使用时 N5a 桶准确率就是多数类基线。同 port_width，留作 prompt 上下文。"
        ),
    ),
    FeatureFamily(
        name="telemetry_gap",
        dimension="遥测缺失范围",
        physical_meaning=(
            "哪些侧 / 指标根本没有采到数。零证据既可能是「一切正常」也可能是「没数据」，"
            "把缺失编码成显式 token 才能让 N6 区分这两种相反的情况，而不是一起降级成先验。"
        ),
        unit="布尔",
        value_domain=("no_telemetry", "partial_telemetry", "full_telemetry"),
        extraction_rule="按已观测字段数与期望字段数的比较产出一个 token。",
        token_template="telemetry:{state}",
        sparsity="每个 case 恒定产出 1 个 token",
        tier="context",
        status="candidate",
        selection_note=(
            "恒定产出且只有三个取值，进 signature 会稀释相似度；"
            "它的真正用途是 N6 区分「一切正常」与「没采到数」，由 decision 层直接读原字段。"
        ),
    ),
    FeatureFamily(
        name="serdes_state",
        dimension="SerDes SNR 是否只有有效 / 失效二值信息",
        physical_meaning=(
            "`serdes_snr` 的量纲未确认，不能按 dB 信噪比解释。"
            "在 v2 中只把它编码成有效 / 失效 / 缺失，避免让模型对未知量纲做连续推断。"
        ),
        unit="二值状态",
        value_domain=("valid", "invalid", "missing"),
        extraction_rule="每侧只统计 serdes_snr 是否存在、是否全部处于失效哨兵 <= 1。",
        token_template="serdes:{side}:{state}",
        sparsity="每侧最多产出 1 个 token",
        tier="measurement_validity",
        status="v2",
        selection_note=(
            "只用于 l2fixed 的 learned SOP 与证据充分性，不替代 C13；"
            "在确认量纲前不得把它解释为低 SNR 的 dB 数值。"
        ),
    ),
)


#: T1 家族消融的选型结果。选型规则在看测试集之前就已固定，且只用训练集留一法：
#:
#: 1. 训练集混合标签 signature 覆盖率 <= 10%（基线 65.87%）。
#: 2. 训练集留一法下 N5a 桶至少 20 条，否则分支结论没有统计意义。
#: 3. 训练集留一法下 N5a 桶内多数投票准确率 > 64.71%（L2 多数类基线）。
#: 4. 训练集留一法下全集多数投票准确率 >= 65.87%（训练集 L2 先验）。
#:
#: 1023 个非空子集中只有 4 个同时满足，按 N5a 桶内准确率排序后取第一名。
V1_FAMILIES: Tuple[str, ...] = ("signal_drop", "status_fault", "lane_imbalance", "level_tail")
V2_FAMILIES: Tuple[str, ...] = (
    "signal_drop",
    "status_fault",
    "lane_imbalance",
    "level_tail",
    "lane_direction",
    "telemetry_gap",
    "serdes_state",
)

ALL_FAMILIES: Tuple[FeatureFamily, ...] = _FAMILIES

#: 声明了全部家族的完整字典，供 `scripts/sweep_feature_families.py` 做消融。
FULL_DICTIONARY = FeatureDictionary(
    version="feature-dictionary-v1-all-families",
    families=_FAMILIES,
    notes=("包含未通过 T1 选型的 candidate 家族，只用于消融，不用于报告主结果。",),
)

#: 冻结的特征字典 v1。实验产物必须记录它的 `version` 与 `content_hash()`。
FEATURE_DICTIONARY = FeatureDictionary(
    version=FEATURE_DICTIONARY_VERSION,
    families=tuple(family for family in _FAMILIES if family.name in V1_FAMILIES),
    notes=(
        "所有 token 都是 ASCII 字面量，可直接进 prompt 与报告，无需再做一层映射。",
        "只在观测偏离训练集中心区域时产出 token；没有恒定产出的 context 家族，"
        "因此零证据 case 仍然是空 signature，由 N6 显式降级而不是靠上下文硬凑相似度。",
        "本字典不产生也不修改 legacy `anomaly_id`；legacy 路径与本字典完全解耦。",
        "candidate 家族仍保留在 `FULL_DICTIONARY` 中，合并数据集到位后按同一套规则重跑选型。",
    ),
)


#: `rca_v2_l2fixed` 专用 v2 字典。它不是默认字典，调用方必须显式选择，
#: 以免影响 organized 数据集上的 legacy / v1 回归锚点。
FEATURE_DICTIONARY_V2 = FeatureDictionary(
    version=FEATURE_DICTIONARY_V2_VERSION,
    families=tuple(family for family in _FAMILIES if family.name in V2_FAMILIES),
    notes=(
        "面向 rca_v2_l2fixed 的显式 profile；默认 FEATURE_DICTIONARY 仍为 v1。",
        "包含 lane_direction、telemetry_gap 与 serdes_state，用于 learned SOP 和证据充分性，"
        "不使用跨端绝对链路损耗，也不把 serdes_snr 当作 dB 连续量。",
    ),
)


#: 命名 profile，用于家族消融与 T10 的实验配置。
PROFILES: Dict[str, Tuple[str, ...]] = {
    "v1": V1_FAMILIES,
    "v2": V2_FAMILIES,
    "legacy_equivalent": ("status_fault", "fence_outlier", "lane_imbalance"),
    "v1_no_level": ("signal_drop", "status_fault", "lane_imbalance"),
    "v1_plus_lane_direction": V1_FAMILIES + ("lane_direction",),
    "v1_plus_context": V1_FAMILIES + ("port_width", "alarm_kind", "telemetry_gap"),
    "all_families": tuple(family.name for family in _FAMILIES),
    "context_only": ("port_width", "alarm_kind", "telemetry_gap"),
    # 迭代 1 的验证用 profile。`side_asymmetry` 在 T1 被否的理由是它让 signature 过度唯一化、
    # 打塌 N5a 桶，而不是它没有判别力。迭代 1 在 media_snr 上测到两端对比确有分层差异
    # （L1 根因下「L2 侧 SNR 更差」命中 32.7%，L2 根因下只有 9.0%），
    # 因此需要单独一个 profile 来回答「加回它能不能提升 LOO 泛化」，而不改动 v1/v2 字典本身。
    "v2_plus_side_asymmetry": V2_FAMILIES + ("side_asymmetry",),
}


def dictionary_for(profile: str) -> FeatureDictionary:
    if profile == "v1":
        return FEATURE_DICTIONARY
    if profile == "v2":
        return FEATURE_DICTIONARY_V2
    if profile not in PROFILES:
        raise KeyError(f"unknown feature profile: {profile}; available: {sorted(PROFILES)}")
    return FULL_DICTIONARY.subset(PROFILES[profile], version_suffix=f"::{profile}")


def iter_families(dictionary: FeatureDictionary) -> Iterable[FeatureFamily]:
    return iter(dictionary.families)
