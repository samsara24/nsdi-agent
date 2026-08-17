"""T2 锁定测试：约束库声明完整性、覆盖面、prompt 渲染与 SKILL.md 同步。

关键一条是 `test_skill_file_is_generated_from_library`：SKILL.md 必须能由约束库
重新渲染得到。这样门限只有一处定义，不会出现「代码改了但 prompt 还是老数字」。
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from rca_framework.anomaly import fit_thresholds, lane_values
from rca_framework.constraints.library import (
    CATEGORIES,
    CONSTRAINT_LIBRARY,
    CONSTRAINT_KINDS,
    Constraint,
    PROVENANCES,
    render_prompt_block,
)
from rca_framework.constraints.layers import (
    CONSTRAINT_LAYER_MAP,
    LAYER_DECISION_TREE,
    LAYER_MEASUREMENT,
    LAYER_PHYSICS,
    OLD_CONSTRAINT_IDS,
)
from rca_framework.constraints.measurement import MEASUREMENT_CONTRACT_LIBRARY
from rca_framework.constraints.physics import PHYSICS_LIBRARY, PhysicalConstraint
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs
from rca_framework.features import dictionary_for, extract_features, fit_feature_model


#: 约束库 v6 的内容指纹。改任何一条约束或 schema 契约都会让它变化。
LIBRARY_CONTENT_HASH = "af09f49aba8039ca"

#: T2 验收要求覆盖的物理量。
REQUIRED_CATEGORIES = (
    "bias_current",
    "temperature",
    "tx_power",
    "rx_power",
    "lane_directional_consistency",
)

EXPECTED_STEP_CONTRACTS = {
    "C1_bias_zero_means_laser_off": (
        ("drop:L1:bias:", "drop:L2:bias:"), ("neutral",), ("",),
    ),
    "C2_bias_healthy_band": (
        ("drop:L1:bias:", "drop:L2:bias:"), ("neutral",), ("",),
    ),
    "C3_temperature_operating_range": ((), ("neutral",), ("",)),
    "C4_voltage_nominal_band": ((), ("neutral",), ("",)),
    "C5_tx_power_range": (
        (
            "drop:L1:txpower:", "drop:L2:txpower:",
            "level:L1:txpower_mean:", "level:L2:txpower_mean:",
        ),
        ("neutral",), ("",),
    ),
    "C6_tx_down_excludes_medium": (
        (
            "drop:L1:txpower:", "drop:L2:txpower:",
            "lane:L1_to_L2:tx_down", "lane:L2_to_L1:tx_down",
        ),
        ("exclude",), ("fiber",),
    ),
    "C7_rx_power_range": (
        (
            "drop:L1:rxpower:", "drop:L2:rxpower:",
            "imbalance:L1:rxpower", "imbalance:L2:rxpower",
            "level:L1:rxpower_mean:", "level:L2:rxpower_mean:",
        ),
        ("neutral",), ("",),
    ),
    "C8_tx_ok_rx_down_indicates_medium": (
        ("lane:L1_to_L2:tx_ok_rx_down", "lane:L2_to_L1:tx_ok_rx_down"),
        ("support",), ("fiber",),
    ),
    "C9_bidirectional_symmetry": (
        (
            "lane:L1_to_L2:bidirectional_same_lane",
            "lane:L2_to_L1:bidirectional_same_lane",
        ),
        ("support",), ("fiber",),
    ),
    "C10_all_lanes_vs_single_lane": (("drop:",), ("neutral",), ("",)),
    "C11_media_snr_floor": (
        (
            "level:L1:media_snr_min:low_tail",
            "level:L2:media_snr_min:low_tail",
        ),
        ("support",), ("fiber",),
    ),
    "C12_no_absolute_link_loss": (("lane:",), ("neutral",), ("",)),
    "C13_serdes_snr_unit_unknown": (("serdes:",), ("neutral",), ("",)),
    "C14_host_snr_mostly_missing": (
        ("telemetry:partial_telemetry", "telemetry:no_telemetry"),
        ("neutral",), ("",),
    ),
    "C15_blackout_sentinel_is_not_laser_off": (
        (
            "drop:L1:txpower:all_lanes", "drop:L2:txpower:all_lanes",
            "drop:L1:rxpower:all_lanes", "drop:L2:rxpower:all_lanes",
        ),
        ("neutral",), ("",),
    ),
    "C16_receive_symptom_constrains_far_transmit_chain": (
        (
            "drop:L1:rxpower:", "drop:L1:media_snr:",
            "status:L1:RxLOS", "status:L1:RxLOL",
            "level:L1:rxpower_mean:low_tail", "level:L1:media_snr_min:low_tail",
            "lane:L2_to_L1:tx_ok_rx_down",
        ),
        ("support",), ("L2",),
    ),
    "C17_l2_side_receive_symptom_is_not_discriminative": (
        (
            "drop:L2:rxpower:", "drop:L2:media_snr:",
            "status:L2:RxLOS", "status:L2:RxLOL",
            "level:L2:rxpower_mean:low_tail", "level:L2:media_snr_min:low_tail",
            "lane:L1_to_L2:tx_ok_rx_down",
        ),
        ("neutral",), ("",),
    ),
    "C18_single_lane_scope_does_not_exclude_fiber": (("drop:",), ("neutral",), ("",)),
    "C19_population_prior_is_not_case_evidence": ((), ("neutral",), ("",)),
    "C20_fiber_not_identifiable_from_current_telemetry": ((), ("neutral",), ("",)),
    "C21_healthy_band_tx_level_is_not_attribution_evidence": (
        ("level:L1:txpower_mean:", "level:L2:txpower_mean:"),
        ("neutral",), ("",),
    ),
    "C22_receive_lane_imbalance_indicates_far_transmit_array": (
        ("imbalance:L2:rxpower",),
        ("support",), ("L1",),
    ),
    "C23_expert_receive_anomaly_on_l1_supports_l2": (
        (
            "expert:L1:rxpower:",
            "expert:L1:media_snr:",
            "expert:pattern:L1:multi_metric",
            "expert:points_to:L1:L2",
        ),
        ("support",), ("L2",),
    ),
    "C24_expert_receive_anomaly_on_l2_supports_l1": (
        (
            "expert:L2:rxpower:",
            "expert:L2:media_snr:",
            "expert:pattern:L2:multi_metric",
            "expert:points_to:L2:L1",
        ),
        ("support",), ("L1",),
    ),
    "C25_expert_local_chain_anomaly_on_l1_supports_l1": (
        (
            "expert:L1:txpower:",
            "expert:L1:host_snr:",
            "expert:L1:serdes_snr:",
            "expert:pattern:L1:port_down",
            "expert:points_to:L1:L1",
        ),
        ("support",), ("L1",),
    ),
    "C26_expert_local_chain_anomaly_on_l2_is_not_discriminative": (
        (
            "expert:L2:txpower:",
            "expert:L2:host_snr:",
            "expert:L2:serdes_snr:",
            "expert:pattern:L2:port_down",
            "expert:points_to:L2:L2",
        ),
        ("neutral",), ("",),
    ),
}


def test_library_is_frozen():
    assert CONSTRAINT_LIBRARY.version == "constraint-library-v6"
    assert len(CONSTRAINT_LIBRARY.constraints) == 26
    assert CONSTRAINT_LIBRARY.content_hash() == LIBRARY_CONTENT_HASH
    assert len(set(CONSTRAINT_LIBRARY.ids())) == len(CONSTRAINT_LIBRARY.ids())


def test_layer_migration_map_covers_all_legacy_constraints():
    assert set(CONSTRAINT_LAYER_MAP) == set(OLD_CONSTRAINT_IDS) == set(CONSTRAINT_LIBRARY.ids())
    assert LAYER_PHYSICS in CONSTRAINT_LAYER_MAP["C6_tx_down_excludes_medium"]
    assert CONSTRAINT_LAYER_MAP["C2_bias_healthy_band"] == (LAYER_DECISION_TREE,)
    assert CONSTRAINT_LAYER_MAP["C12_no_absolute_link_loss"] == (LAYER_MEASUREMENT,)
    assert set(CONSTRAINT_LAYER_MAP["C23_expert_receive_anomaly_on_l1_supports_l2"]) == {
        LAYER_PHYSICS,
        LAYER_DECISION_TREE,
    }


def test_physics_layer_rejects_train_set_fitted_parameters():
    with pytest.raises(ValueError, match="train-set fitted"):
        PhysicalConstraint(
            constraint_id="P_bad_bias_band",
            title="bad",
            statement="bad",
            formal_expression="7.2 <= bias <= 7.8",
            diagnostic_use="bad",
            prompt_text="bad",
            provenance="derived",
            parameters=(("训练集偏置范围", "7.2-7.8 mA"),),
        )


def test_new_layers_have_expected_contracts():
    assert "P5_tx_down_excludes_medium" in PHYSICS_LIBRARY.ids()
    assert "M4_blackout_sentinel_is_no_reading" in MEASUREMENT_CONTRACT_LIBRARY.ids()
    assert {item.kind for item in MEASUREMENT_CONTRACT_LIBRARY.contracts} == {"veto"}


def test_v4_supplies_the_missing_device_side_support_constraints():
    """v3 里没有任何约束允许 support L1 / L2，这是 LLM 大量违规的结构性原因。

    v4 必须至少提供一条能正向支持设备侧根因的约束，否则模型想给 L1 / L2 找依据时
    只能违规引用 neutral 约束。
    """
    device_support = [
        item
        for item in CONSTRAINT_LIBRARY.constraints
        if "support" in item.allowed_effects
        and set(item.allowed_targets) & {"L1", "L2"}
    ]
    assert device_support, "约束库必须至少有一条可以支持 L1 或 L2 的约束"


def test_negative_results_are_encoded_as_caveats():
    """实测到的负结果必须进库，而不是只写在报告里。

    C17 记录「L2 侧接收症状对 L1 没有增益」，C20 记录「fiber 在现有遥测下不可识别」。
    它们的价值是把弃权与补采变成有依据的动作，而不是靠模型自觉。
    """
    for constraint_id in (
        "C17_l2_side_receive_symptom_is_not_discriminative",
        "C20_fiber_not_identifiable_from_current_telemetry",
    ):
        item = CONSTRAINT_LIBRARY.get(constraint_id)
        assert item.kind == "caveat"
        assert item.provenance == "measured"
        assert any(character.isdigit() for character in item.measured_evidence)


def test_required_categories_are_covered():
    """T2 验收：至少覆盖电流、温度、发光功率、收光功率、同 lane 方向性一致性。"""
    for category in REQUIRED_CATEGORIES:
        assert CONSTRAINT_LIBRARY.by_category(category), f"缺少 {category} 类约束"


def test_every_constraint_declares_provenance_and_evidence():
    for item in CONSTRAINT_LIBRARY.constraints:
        assert item.kind in CONSTRAINT_KINDS
        assert item.category in CATEGORIES
        assert item.provenance in PROVENANCES
        assert item.physical_statement, item.constraint_id
        assert item.formal_expression, item.constraint_id
        assert item.measured_evidence, item.constraint_id
        assert item.diagnostic_use, item.constraint_id
        assert item.prompt_text, item.constraint_id
        if item.provenance == "measured":
            # 实测参数必须带数字，否则无法被专家核对，也无法在换数据集后重标定。
            assert any(character.isdigit() for character in item.measured_evidence), item.constraint_id


def test_measured_constraints_are_bound_to_a_dataset():
    measured = [item for item in CONSTRAINT_LIBRARY.constraints if item.provenance == "measured"]
    assert measured
    assert "rca_v2_l2fixed" in CONSTRAINT_LIBRARY.measured_on


def test_all_constraints_start_pending_expert_review():
    """约束库尚未经夏思博审核，状态必须如实标注，不能默认 approved。"""
    assert {item.review_status for item in CONSTRAINT_LIBRARY.constraints} == {"pending_expert_review"}


def test_library_is_json_serializable():
    payload = json.dumps(CONSTRAINT_LIBRARY.to_dict(), ensure_ascii=False, sort_keys=True)
    assert json.loads(payload)["version"] == "constraint-library-v6"


def test_every_constraint_has_an_exact_v2_step_contract():
    assert set(EXPECTED_STEP_CONTRACTS) == set(CONSTRAINT_LIBRARY.ids())
    for item in CONSTRAINT_LIBRARY.constraints:
        expected = EXPECTED_STEP_CONTRACTS[item.constraint_id]
        assert (
            item.applies_to_token_prefixes,
            item.allowed_effects,
            item.allowed_targets,
        ) == expected


def test_invalid_constraint_fields_are_rejected():
    base = dict(
        constraint_id="X",
        category="tx_power",
        kind="indicator",
        title="t",
        physical_statement="p",
        formal_expression="f",
        parameters=(),
        provenance="measured",
        measured_evidence="e",
        diagnostic_use="d",
        prompt_text="x",
    )
    for field, value in (
        ("kind", "guess"),
        ("provenance", "vibes"),
        ("category", "color"),
        ("review_status", "ok"),
        ("allowed_effects", ("boost",)),
        ("allowed_targets", ("switch",)),
    ):
        with pytest.raises(ValueError):
            Constraint(**{**base, field: value})


def test_c14_and_c15_statistics_are_from_l2fixed_manifest_train_only():
    data_dir = Path("datasets/rca_v2_l2fixed")
    train = cases_by_manifest_split(data_dir, "train")
    test = cases_by_manifest_split(data_dir, "test")
    assert (len(train), len(test)) == (161, 107)

    host_present = [
        case for case in train
        if any(
            any(value is not None for value in lane_values(case, "host_snr", side).values())
            for side in ("L1", "L2")
        )
    ]
    assert len(host_present) == 52
    c14 = CONSTRAINT_LIBRARY.get("C14_host_snr_mostly_missing")
    assert "52/161" in c14.formal_expression
    assert "52 条" in c14.measured_evidence
    assert "109 条" in c14.measured_evidence

    train_packs = build_packs(train, source_dataset=str(data_dir))
    blackouts = [
        case for case, pack in zip(train, train_packs)
        if pack.optical_blackout
    ]
    assert len(blackouts) == 4
    assert Counter(case["label"] for case in blackouts) == Counter({"L2": 3, "fiber": 1})
    c15 = CONSTRAINT_LIBRARY.get("C15_blackout_sentinel_is_not_laser_off")
    assert ("训练集命中", "4/161") in c15.parameters
    assert "L2 3 条、fiber 1 条" in c15.measured_evidence


def test_skill_file_is_generated_from_library():
    """SKILL.md 必须能由约束库重新渲染得到。

    本文件的 docstring 从 T2 起就声称有这条守护，但测试一直没写，
    于是约束库升到 v5 时 SKILL.md 还停在 v4 也没人拦——注入 prompt 的门限
    和代码里的门限可以静默分叉，而这正是这套设计最不能出的问题。
    """
    from scripts.render_constraint_skill import SKILL_PATH, render

    assert SKILL_PATH.exists(), "SKILL.md 缺失，请运行 python scripts/render_constraint_skill.py"
    assert SKILL_PATH.read_text(encoding="utf-8") == render(), (
        "SKILL.md 已过期，请运行 python scripts/render_constraint_skill.py"
    )


def test_c21_and_c22_statistics_are_recomputed_from_l2fixed_train_split():
    """v5 两条约束的数字必须能由数据重算，不能只写在注释里。

    这两条的方向相反、风险也相反：C21 否掉一个统计上很诱人的信号，
    C22 立起全训练集上唯一一条能支持 L1 的证据。前者写松了会放进伪信号，
    后者写松了会把 7 条样本吹成判据，所以都在这里重算。
    """
    data_dir = Path("datasets/rca_v2_l2fixed")
    train = cases_by_manifest_split(data_dir, "train")
    labels = [str(case["label"]) for case in train]
    packs = build_packs(train, source_dataset=str(data_dir))
    dictionary = dictionary_for("v2")
    model = fit_feature_model(packs, dictionary=dictionary)
    thresholds = fit_thresholds(train)
    tokens = [
        set(extract_features(pack, thresholds, model, dictionary=dictionary).tokens)
        for pack in packs
    ]

    # C21：发送功率低尾命中 39 条，且没有一条是断光哨兵造成的，
    # 即它确实在描述「正常带内偏低」这件没有判别力的事。
    tx_low = [index for index, item in enumerate(tokens) if "level:L1:txpower_mean:low_tail" in item]
    assert len(tx_low) == 39
    for index in tx_low:
        values = [
            value for value in lane_values(train[index], "txpower", "L1").values()
            if value is not None
        ]
        assert values and min(values) > -39.0
    assert Counter(labels[index] for index in tx_low) == Counter({"L2": 29, "L1": 6, "fiber": 4})
    c21 = CONSTRAINT_LIBRARY.get("C21_healthy_band_tx_level_is_not_attribution_evidence")
    assert c21.kind == "caveat"
    assert "39 条" in c21.measured_evidence
    assert "58.9%" in c21.measured_evidence  # Wilson 下界低于 L2 先验 62.1%

    # C22：L2 侧接收不均衡只有 7 条支持，其中 6 条根因在对端 L1。
    imbalance = [index for index, item in enumerate(tokens) if "imbalance:L2:rxpower" in item]
    assert len(imbalance) == 7
    assert Counter(labels[index] for index in imbalance) == Counter({"L1": 6, "fiber": 1})
    c22 = CONSTRAINT_LIBRARY.get("C22_receive_lane_imbalance_indicates_far_transmit_array")
    assert c22.kind == "indicator"
    assert ("L2 侧接收不均衡命中", "7/161") in c22.parameters
    assert ("其中根因为对端 L1", "6/7 = 85.7%") in c22.parameters
    # 唯一一条支持 L1 的约束；如果这里被放宽到 L2，C17 的负结果就被绕过了。
    assert c22.allowed_effects == ("support",)
    assert c22.allowed_targets == ("L1",)


def test_c12_lane_alignment_check_is_reproducible():
    """C12 的第二个证据：两端最差 lane 的编号一致率与随机无法区分。

    这个检验只用序、不用数值，所以它不受标定口径影响，
    是判断「一份遥测的两端 lane 能不能按号配对」最省事的办法。
    数字写进了约束，这里重算一遍防止它随数据集变化而失真。
    """
    data_dir = Path("datasets/rca_v2_l2fixed")
    train = cases_by_manifest_split(data_dir, "train")

    def worst_lane(case, metric, side):
        values = lane_values(case, metric, side)
        healthy = {
            lane: value for lane, value in values.items()
            if value is not None and value > -39.0
        }
        if len(healthy) < 2:
            return None
        return min(healthy, key=lambda lane: healthy[lane])

    tallies = {}
    for metric in ("rxpower", "media_snr"):
        aligned = comparable = 0
        for case in train:
            left = worst_lane(case, metric, "L1")
            right = worst_lane(case, metric, "L2")
            if left is None or right is None:
                continue
            comparable += 1
            aligned += int(left == right)
        tallies[metric] = (aligned, comparable)

    assert tallies["rxpower"] == (37, 155)
    assert tallies["media_snr"] == (46, 161)
    # 4 lane 下随机一致概率是 25%，两个实测值分别为 23.9% 与 28.6%。
    for aligned, comparable in tallies.values():
        assert abs(aligned / comparable - 0.25) < 0.05

    c12 = CONSTRAINT_LIBRARY.get("C12_no_absolute_link_loss")
    assert "37/155 = 23.9%" in c12.measured_evidence
    assert "46/161 = 28.6%" in c12.measured_evidence


def test_prompt_block_orders_exclusions_before_indicators():
    block = render_prompt_block()
    assert "## 排除条件" in block
    assert "## 禁止推断" in block
    assert "## 倾向性线索" in block
    assert block.index("## 排除条件") < block.index("## 禁止推断") < block.index("## 倾向性线索")
    for item in CONSTRAINT_LIBRARY.constraints:
        assert item.constraint_id in block
        assert item.prompt_text in block
    # 待审核状态必须出现在 prompt 里，避免 LLM 把未审约束当作已确认事实。
    assert "（待专家审核）" in block


def test_prompt_block_can_be_filtered_by_category():
    block = render_prompt_block(categories=("tx_power",))
    assert "C5_tx_power_range" in block
    assert "C3_temperature_operating_range" not in block


def test_prompt_block_does_not_contain_absolute_link_loss_threshold():
    """C12 是禁止推断项；prompt 里不能同时出现绝对损耗门限，否则自相矛盾。"""
    block = render_prompt_block()
    assert "C12_no_absolute_link_loss" in block
    for forbidden in ("3.11 dB", "3.42 dB", "损耗上界"):
        assert forbidden not in block


def test_skill_renderer_uses_current_library():
    from scripts.render_constraint_skill import render

    rendered = render()
    assert LIBRARY_CONTENT_HASH in rendered
    for item in CONSTRAINT_LIBRARY.constraints:
        assert item.constraint_id in rendered
        assert item.prompt_text in rendered
