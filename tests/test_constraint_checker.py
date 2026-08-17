"""T6 测试：M7 可执行断言校验器。

覆盖四类违规（幻觉证据、结论违反排除约束、说了 caveat 禁止的话、凭空断言）
以及证据包自校验。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rca_framework.constraints.checker import (
    FORBIDDEN_CLAIM_PATTERNS,
    CheckReport,
    Violation,
    check_evidence,
    check_response,
)
from rca_framework.data import load_cases
from rca_framework.evidence_pack import EvidencePack, build_packs
from rca_framework.llm.protocol import DiagnosisResponse, ReasoningStep


DATA_DIR = Path("datasets/organized_rca_v2_stratified_60_40_seed42")


@pytest.fixture(scope="module")
def packs():
    return build_packs(load_cases(DATA_DIR))


@pytest.fixture(scope="module")
def normal_pack(packs):
    for pack in packs:
        if not pack.optical_blackout and pack.observed_fields:
            return pack
    pytest.skip("no normal pack")


@pytest.fixture(scope="module")
def blackout_pack(packs):
    for pack in packs:
        if pack.optical_blackout:
            return pack
    pytest.skip("no blackout pack")


def response(*steps, verdict="L2", confidence=0.8):
    return DiagnosisResponse(steps=tuple(steps), verdict=verdict, confidence=confidence)


def step(claim="观测到 L2 侧接收功率偏低", evidence=("level:L2:rxpower_mean:low_tail",),
         constraints=(), effect="support", target="L2"):
    return ReasoningStep(
        claim=claim, cited_evidence=tuple(evidence), cited_constraints=tuple(constraints),
        effect=effect, target=target,
    )


# --- 违规类型 ---------------------------------------------------------------

def test_fabricated_evidence_is_fatal(normal_pack):
    report = check_response(
        response(step(evidence=("level:L2:rxpower_mean:low_tail", "drop:L9:nonexistent:all_lanes"))),
        normal_pack,
        available_evidence=("level:L2:rxpower_mean:low_tail",),
    )
    assert not report.ok
    kinds = {item.kind for item in report.fatal}
    assert "fabricated_evidence" in kinds
    assert "drop:L9:nonexistent:all_lanes" in report.feedback()


def test_fabricated_constraint_is_fatal(normal_pack):
    report = check_response(
        response(step(constraints=("C99_made_up",))),
        normal_pack,
        available_evidence=("level:L2:rxpower_mean:low_tail",),
    )
    assert "fabricated_constraint" in {item.kind for item in report.fatal}


def test_step_without_any_citation_is_fatal(normal_pack):
    report = check_response(
        response(step(claim="我觉得应该是 L2", evidence=(), constraints=())),
        normal_pack,
        available_evidence=("level:L2:rxpower_mean:low_tail",),
    )
    assert "unsupported_step" in {item.kind for item in report.fatal}


def test_verdict_outside_allowed_candidates_is_fatal(normal_pack):
    report = check_response(
        response(step(), verdict="fiber"),
        normal_pack,
        available_evidence=("level:L2:rxpower_mean:low_tail",),
        allowed_root_causes=("L1", "L2"),
    )
    violation = next(item for item in report.fatal if item.kind == "constraint_violation")
    assert violation.constraint_id == "C6_tx_down_excludes_medium"


def test_verdict_without_steps_is_fatal(normal_pack):
    report = check_response(
        response(verdict="L2"), normal_pack, available_evidence=("level:L2:rxpower_mean:low_tail",)
    )
    assert "unsupported_step" in {item.kind for item in report.fatal}


def test_abstain_is_always_allowed(normal_pack):
    report = check_response(
        DiagnosisResponse(steps=(), verdict=None), normal_pack, available_evidence=()
    )
    assert report.ok, "弃权不该被判为违规，否则模型会被逼着给结论"


def test_clean_response_passes(normal_pack):
    report = check_response(
        response(
            step(
                constraints=("C7_rx_power_range",),
                effect="neutral",
                target="",
            )
        ),
        normal_pack,
        available_evidence=("level:L2:rxpower_mean:low_tail",),
    )
    assert report.ok
    assert report.feedback() == ""


def test_constraint_rejects_wrong_token_family(normal_pack):
    token = "level:L2:txpower_mean:low_tail"
    report = check_response(
        response(
            step(
                evidence=(token,),
                constraints=("C7_rx_power_range",),
                effect="neutral",
                target="",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    violation = next(
        item for item in report.fatal
        if item.constraint_id == "C7_rx_power_range"
    )
    assert "token 家族" in violation.message


def test_constraint_rejects_wrong_effect(normal_pack):
    token = "level:L2:rxpower_mean:low_tail"
    report = check_response(
        response(
            step(
                evidence=(token,),
                constraints=("C7_rx_power_range",),
                effect="support",
                target="",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    violation = next(
        item for item in report.fatal
        if item.constraint_id == "C7_rx_power_range" and "effect=support" in item.message
    )
    assert violation.kind == "constraint_violation"


def test_constraint_rejects_wrong_target(normal_pack):
    token = "level:L2:media_snr_min:low_tail"
    report = check_response(
        response(
            step(
                evidence=(token,),
                constraints=("C11_media_snr_floor",),
                effect="support",
                target="L2",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    violation = next(
        item for item in report.fatal
        if item.constraint_id == "C11_media_snr_floor" and "target=L2" in item.message
    )
    assert violation.kind == "constraint_violation"


def test_indicator_can_only_weakly_support_fiber(normal_pack):
    token = "level:L2:media_snr_min:low_tail"
    report = check_response(
        response(
            step(
                evidence=(token,),
                constraints=("C11_media_snr_floor",),
                effect="support",
                target="fiber",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    assert report.ok


def test_c6_only_excludes_fiber(normal_pack):
    token = "drop:L1:txpower:all_lanes"
    valid = check_response(
        response(
            step(
                evidence=(token,),
                constraints=("C6_tx_down_excludes_medium",),
                effect="exclude",
                target="fiber",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    assert valid.ok

    invalid = check_response(
        response(
            step(
                evidence=(token,),
                constraints=("C6_tx_down_excludes_medium",),
                effect="support",
                target="L1",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    assert not invalid.ok


def test_endpoint_ambiguous_indicator_cannot_support_endpoint(normal_pack):
    token = "lane:L1_to_L2:tx_ok_rx_down"
    report = check_response(
        response(
            step(
                evidence=(token,),
                constraints=("C8_tx_ok_rx_down_indicates_medium",),
                effect="support",
                target="L2",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    assert any(
        item.constraint_id == "C8_tx_ok_rx_down_indicates_medium"
        and "target=L2" in item.message
        for item in report.fatal
    )


def test_constraint_without_v2_token_cannot_borrow_optical_evidence(normal_pack):
    token = "level:L2:rxpower_mean:low_tail"
    invalid = check_response(
        response(
            step(
                evidence=(token,),
                constraints=("C3_temperature_operating_range",),
                effect="neutral",
                target="",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    assert any(
        item.constraint_id == "C3_temperature_operating_range"
        and "没有对应 token" in item.message
        for item in invalid.fatal
    )

    context_only = check_response(
        response(
            step(
                claim="温度规则仅作量测解释",
                evidence=(),
                constraints=("C3_temperature_operating_range",),
                effect="neutral",
                target="",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(token,),
    )
    assert context_only.ok


def test_blackout_preconditions_guard_c6_and_c15(normal_pack, blackout_pack):
    blackout_token = "drop:L1:txpower:all_lanes"
    c6_during_blackout = check_response(
        response(
            step(
                evidence=(blackout_token,),
                constraints=("C6_tx_down_excludes_medium",),
                effect="exclude",
                target="fiber",
            ),
            verdict=None,
        ),
        blackout_pack,
        available_evidence=(blackout_token,),
    )
    assert any(
        item.constraint_id == "C6_tx_down_excludes_medium"
        and "前提不成立" in item.message
        for item in c6_during_blackout.fatal
    )

    c15_without_blackout = check_response(
        response(
            step(
                evidence=(blackout_token,),
                constraints=("C15_blackout_sentinel_is_not_laser_off",),
                effect="neutral",
                target="",
            ),
            verdict=None,
        ),
        normal_pack,
        available_evidence=(blackout_token,),
    )
    assert any(
        item.constraint_id == "C15_blackout_sentinel_is_not_laser_off"
        and "未命中全链路 blackout" in item.message
        for item in c15_without_blackout.fatal
    )

    c15_during_blackout = check_response(
        response(
            step(
                evidence=(blackout_token,),
                constraints=("C15_blackout_sentinel_is_not_laser_off",),
                effect="neutral",
                target="",
            ),
            verdict=None,
        ),
        blackout_pack,
        available_evidence=(blackout_token,),
    )
    assert c15_during_blackout.ok


# --- caveat 类禁止说法 ------------------------------------------------------

@pytest.mark.parametrize("claim,constraint_id", [
    ("链路损耗约 3.2 dB，超出正常范围", "C12_no_absolute_link_loss"),
    ("serdes_snr 只有 12.5 dB，明显偏低", "C13_serdes_snr_unit_unknown"),
    ("host_snr 正常，排除主机侧问题", "C14_host_snr_mostly_missing"),
])
def test_forbidden_claims_are_detected(normal_pack, claim, constraint_id):
    report = check_response(
        response(step(claim=claim)), normal_pack,
        available_evidence=("level:L2:rxpower_mean:low_tail",),
    )
    violation = next(item for item in report.fatal if item.kind == "forbidden_claim")
    assert violation.constraint_id == constraint_id


def test_laser_off_claim_only_forbidden_during_blackout(normal_pack, blackout_pack):
    """正常 case 说「未发光」是合法的；全链路失效时说才是违规。"""
    claim = "L1 侧未发光"
    tokens = ("drop:L1:txpower:all_lanes",)

    normal = check_response(response(step(claim=claim, evidence=tokens)), normal_pack, tokens)
    assert not any(item.kind == "forbidden_claim" for item in normal.fatal)

    during = check_response(response(step(claim=claim, evidence=tokens)), blackout_pack, tokens)
    violation = next(item for item in during.fatal if item.kind == "forbidden_claim")
    assert violation.constraint_id == "C15_blackout_sentinel_is_not_laser_off"


def test_any_verdict_during_blackout_is_vetoed(blackout_pack):
    tokens = ("drop:L1:txpower:all_lanes",)
    report = check_response(response(step(claim="收发全断", evidence=tokens)), blackout_pack, tokens)
    assert any(
        item.constraint_id == "M4_blackout_sentinel_is_no_reading"
        and item.kind == "measurement_veto"
        for item in report.veto
    )
    # 但弃权是允许的。
    assert check_response(DiagnosisResponse(verdict=None), blackout_pack, tokens).ok


def test_forbidden_patterns_do_not_fire_on_ordinary_text(normal_pack):
    """正则宁可漏检也不能误伤：误伤会让合规推理被反复重写。"""
    ordinary = [
        "L2 侧接收功率处于低尾档，低于训练集 25 分位",
        "两侧 lane 之间存在功率不平衡",
        "serdes_snr 失效，按二值信号处理",
        "host_snr 未采集，不用于推断",
    ]
    for claim in ordinary:
        report = check_response(
            response(step(claim=claim)), normal_pack,
            available_evidence=("level:L2:rxpower_mean:low_tail",),
        )
        assert not any(item.kind == "forbidden_claim" for item in report.fatal), claim


# --- 证据包自校验 -----------------------------------------------------------

def test_evidence_check_flags_blackout(blackout_pack):
    report = check_evidence(blackout_pack)
    assert any(item.constraint_id == "C15_blackout_sentinel_is_not_laser_off" for item in report.violations)
    # invariant 违规只是警告，不该阻断推理。
    assert report.ok


def test_evidence_check_finds_the_known_low_voltage_case(packs):
    """Validation.md V6 记录的那一例 3.10 V 必须被 C4 抓到。"""
    flagged = [
        pack.case_id for pack in packs
        if any(item.constraint_id == "C4_voltage_nominal_band" for item in check_evidence(pack).violations)
    ]
    assert "case_aa307cc7c7db" in flagged


def test_evidence_check_is_clean_on_most_cases(packs):
    """绝大多数 case 不该触发量测有效性告警，否则说明断言写得太紧。"""
    clean = sum(1 for pack in packs if not check_evidence(pack).violations)
    assert clean / len(packs) > 0.8


def test_violation_rejects_unknown_kind_and_severity():
    with pytest.raises(ValueError, match="unknown violation kind"):
        Violation(kind="nope", severity="fatal", message="x")
    with pytest.raises(ValueError, match="unknown severity"):
        Violation(kind="unsupported_step", severity="nope", message="x")


def test_report_feedback_lists_only_fatal_violations():
    report = CheckReport(violations=(
        Violation(kind="unsupported_step", severity="fatal", message="严重", step_index=0),
        Violation(kind="invalid_measurement", severity="warning", message="轻微"),
    ))
    feedback = report.feedback()
    assert "严重" in feedback and "轻微" not in feedback
    assert "第 1 步" in feedback


def test_every_forbidden_pattern_maps_to_a_real_constraint():
    from rca_framework.constraints.library import CONSTRAINT_LIBRARY

    for constraint_id in FORBIDDEN_CLAIM_PATTERNS:
        constraint = CONSTRAINT_LIBRARY.get(constraint_id)
        assert constraint.kind == "caveat"
