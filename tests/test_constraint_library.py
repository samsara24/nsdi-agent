"""T2 锁定测试：约束库声明完整性、覆盖面、prompt 渲染与 SKILL.md 同步。

关键一条是 `test_skill_file_is_generated_from_library`：SKILL.md 必须能由约束库
重新渲染得到。这样门限只有一处定义，不会出现「代码改了但 prompt 还是老数字」。
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from rca_framework.anomaly import lane_values
from rca_framework.constraints.library import (
    CATEGORIES,
    CONSTRAINT_LIBRARY,
    CONSTRAINT_KINDS,
    Constraint,
    PROVENANCES,
    render_prompt_block,
)
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs


#: 约束库 v3 的内容指纹。改任何一条约束或 schema 契约都会让它变化。
LIBRARY_CONTENT_HASH = "c090f825efe2da67"

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
}


def test_library_is_frozen():
    assert CONSTRAINT_LIBRARY.version == "constraint-library-v3"
    assert len(CONSTRAINT_LIBRARY.constraints) == 15
    assert CONSTRAINT_LIBRARY.content_hash() == LIBRARY_CONTENT_HASH
    assert len(set(CONSTRAINT_LIBRARY.ids())) == len(CONSTRAINT_LIBRARY.ids())


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
    assert json.loads(payload)["version"] == "constraint-library-v3"


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
