"""解释器可核对性检查的测试。

这组检查将来要在没有标签的生产环境上跑，因此它必须**只依赖证据与规则输出**，
不能偷偷用到真值。测试锁住每一项的判定边界，特别是两条容易写松的：
`verdict_consistency` 不许模型改判、`direction_consistency` 必须真的读指标类型
而不是只看两端是否不同。
"""

from __future__ import annotations

import json

import pytest

from rca_framework.llm.explain import (
    EXPLAIN_PROMPT_VERSION,
    build_explain_prompt,
    check_explanation,
    parse_explanation,
    summarize_checks,
)

AVAILABLE = (
    "expert:L1:rxpower:lane_down",
    "expert:L1:media_snr:lane_down",
    "expert:L1:txpower:lane_down",
    "level:L1:rxpower_mean:low_tail",
    "expert:points_to:L1:L2",
)
RULE_EVIDENCE = [("L1", "rxpower"), ("L1", "media_snr")]


def _explanation(**overrides):
    base = {
        "summary": "L1 侧收光跌破阈值，问题在对端 L2 的发送链路。",
        "key_evidence": [
            {"token": "expert:L1:rxpower:lane_down", "reads_as": "L1 某 lane 收光跌到阈值以下"}
        ],
        "symptom_side": "L1",
        "root_cause_side": "L2",
        "direction_reason": "收光测的是对端发来的光，所以指向对端。",
        "next_actions": ["查 L2 侧对应 lane 的发送功率与偏置电流"],
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def _check(text, **overrides):
    kwargs = {
        "available_tokens": AVAILABLE,
        "rule_evidence": RULE_EVIDENCE,
        "rule_verdict": "L2",
    }
    kwargs.update(overrides)
    return check_explanation(parse_explanation(text), **kwargs)


def test_well_formed_explanation_passes_all_four_checks():
    report = _check(_explanation())
    assert report["all_pass"]
    assert report["fabricated_tokens"] == []


def test_fabricated_token_fails_existence():
    report = _check(_explanation(key_evidence=[{"token": "expert:L1:bogus:x", "reads_as": "无"}]))
    assert not report["token_existence"]
    assert report["fabricated_tokens"] == ["expert:L1:bogus:x"]
    assert not report["all_pass"]


def test_true_but_irrelevant_evidence_fails_relevance():
    """引用的 token 真实存在，但与规则的依据无关——解释等于答非所问。"""
    report = _check(
        _explanation(
            key_evidence=[{"token": "expert:L1:txpower:lane_down", "reads_as": "L1 发光异常"}],
            root_cause_side="L1",
        ),
        rule_verdict="L1",
    )
    assert report["token_existence"]
    assert not report["token_relevance"]


def test_model_may_not_override_the_rule_verdict():
    report = _check(_explanation(root_cause_side="L1", symptom_side="L1"))
    assert not report["verdict_consistency"]
    assert not report["all_pass"]


def test_direction_check_reads_metric_type_not_just_side_difference():
    """症状端与根因端不同还不够：本端类指标指向本端，说成对端就是错的。"""
    report = _check(
        _explanation(
            key_evidence=[{"token": "expert:L1:txpower:lane_down", "reads_as": "L1 发光异常"}],
            symptom_side="L1",
            root_cause_side="L2",
        ),
        rule_evidence=[("L1", "txpower")],
    )
    assert report["token_relevance"]
    assert not report["direction_consistency"]


def test_local_metric_pointing_local_is_consistent():
    report = _check(
        _explanation(
            key_evidence=[{"token": "expert:L1:txpower:lane_down", "reads_as": "L1 发光异常"}],
            symptom_side="L1",
            root_cause_side="L1",
        ),
        rule_evidence=[("L1", "txpower")],
        rule_verdict="L1",
    )
    assert report["direction_consistency"]
    assert report["all_pass"]


def test_quantile_family_token_also_carries_direction():
    """方向由 (侧, 指标) 决定，与判据族无关——与约束语义等价类同一套坐标。"""
    report = _check(
        _explanation(
            key_evidence=[{"token": "level:L1:rxpower_mean:low_tail", "reads_as": "L1 收光偏低"}]
        )
    )
    assert report["direction_consistency"]


def test_fiber_verdict_skips_direction_check():
    """方向表说不出光纤，它是两端仲裁的产物，不该被方向检查判错。"""
    report = _check(
        _explanation(root_cause_side="fiber"),
        rule_verdict="fiber",
    )
    assert report["direction_consistency"]
    assert report["all_pass"]


def test_rule_without_evidence_does_not_fail_relevance():
    """no_anomaly 兜底本来就没有依据可引，苛求相关性等于惩罚如实解释。"""
    report = _check(_explanation(), rule_evidence=[])
    assert report["token_relevance"]


@pytest.mark.parametrize(
    "text",
    ["", "没有 JSON", _explanation(symptom_side="L3"), _explanation(root_cause_side="unknown")],
)
def test_unparseable_output_fails_every_check(text):
    report = _check(text)
    assert not report["parsed"]
    assert not report["all_pass"]
    for name in ("token_existence", "token_relevance", "direction_consistency"):
        assert not report[name]


def test_empty_evidence_fails_existence():
    report = _check(_explanation(key_evidence=[]))
    assert not report["token_existence"]


def test_prompt_is_deterministic_and_forbids_overriding():
    kwargs = dict(
        case_id="c",
        expert_verdict="L2",
        expert_group="expert:multi_metric",
        expert_reason="L1 侧三项组合异常",
        rule_evidence=[("L1", "rxpower")],
        evidence_tokens=list(AVAILABLE),
    )
    first, second = build_explain_prompt(**kwargs), build_explain_prompt(**kwargs)
    assert first == second
    assert "没有改判的资格" in first
    assert "不可更改" in first


def test_summary_counts_are_rates():
    reports = [
        _check(_explanation()),
        _check(_explanation(root_cause_side="L1", symptom_side="L1")),
    ]
    summary = summarize_checks(reports)
    assert summary["case_count"] == 2
    assert summary["all_pass"] == 0.5
    assert summary["token_existence"] == 1.0


def test_version_is_pinned():
    assert EXPLAIN_PROMPT_VERSION == "rca-explain-v1"
