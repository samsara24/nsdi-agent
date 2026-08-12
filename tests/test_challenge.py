"""质疑器协议与指标的测试。

这一组测试锁住的核心性质是：**解析失败不得被算成质疑**。
质疑命中率是这个岗位唯一的考核指标，而模型输出不合 schema 是它最常见的失败形态；
若把解析失败记成质疑，命中率就会被一批与内容无关的噪声稀释或抬高，
整个评测失去意义。
"""

from __future__ import annotations

import json

import pytest

from rca_framework.llm.challenge import (
    ASSESSMENTS,
    CHALLENGE_PROMPT_VERSION,
    PREMISES,
    PREMISE_TEXT,
    build_challenge_prompt,
    challenge_metrics,
    parse_challenge,
)


def _payload(**overrides):
    base = {
        "assessment": "challenge",
        "premise_at_risk": "other_side_overlooked",
        "cited_evidence": ["expert:L2:rxpower:low_value"],
        "alternative_verdict": "L1",
        "confidence": 0.7,
        "evidence_to_collect": ["L1 侧各 lane 发送功率历史"],
        "explanation": "另一端存在同等强度的接收异常。",
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def test_every_premise_has_operator_facing_text():
    assert set(PREMISE_TEXT) == set(PREMISES)
    for text in PREMISE_TEXT.values():
        assert len(text) > 20


def test_parse_roundtrip():
    parsed = parse_challenge(_payload())
    assert parsed is not None
    assert parsed.challenges
    assert parsed.premise_at_risk == "other_side_overlooked"
    assert parsed.alternative_verdict == "L1"
    assert parsed.cited_evidence == ("expert:L2:rxpower:low_value",)


def test_agree_is_not_a_challenge():
    parsed = parse_challenge(_payload(assessment="agree", premise_at_risk=""))
    assert parsed is not None and not parsed.challenges


@pytest.mark.parametrize(
    "text",
    [
        "",
        "没有 JSON 的一段话",
        _payload(assessment="maybe"),
        _payload(premise_at_risk="made_up_premise"),
        _payload(alternative_verdict="L3"),
        _payload(confidence="很高"),
    ],
)
def test_malformed_output_returns_none(text):
    """返回 None 而不是一个默认的 challenge——调用方据此按 agree 计入。"""
    assert parse_challenge(text) is None


def test_confidence_is_clamped():
    assert parse_challenge(_payload(confidence=3.0)).confidence == 1.0
    assert parse_challenge(_payload(confidence=-1.0)).confidence == 0.0


def test_thinking_block_before_json_is_tolerated():
    """DeepSeek-R1 会先输出 <think> 段，JSON 在末尾。"""
    text = "<think>先看两端读数……</think>\n\n```json\n" + _payload() + "\n```"
    assert parse_challenge(text) is not None


def test_prompt_is_deterministic_and_carries_the_rule_under_review():
    kwargs = dict(
        case_id="case_x",
        expert_verdict="L2",
        expert_group="expert:multi_metric",
        expert_reason="L1 侧三项组合异常，按方向表指向对端",
        evidence_tokens=["expert:L1:rxpower:lane_down", "expert:points_to:L1:L2"],
        missing_fields=["L1.host_snr"],
    )
    first, second = build_challenge_prompt(**kwargs), build_challenge_prompt(**kwargs)
    assert first == second
    # 复核的对象必须出现在 prompt 里，否则模型无从质疑。
    assert "expert:multi_metric" in first
    assert "L1 侧三项组合异常" in first
    for premise in PREMISES:
        assert premise in first


def test_prompt_sorts_evidence_so_token_order_does_not_change_the_prompt():
    a = build_challenge_prompt(
        case_id="c", expert_verdict="L1", expert_group="g", expert_reason="r",
        evidence_tokens=["b:token", "a:token"],
    )
    b = build_challenge_prompt(
        case_id="c", expert_verdict="L1", expert_group="g", expert_reason="r",
        evidence_tokens=["a:token", "b:token"],
    )
    assert a == b


def test_metrics_hit_rate_and_recall():
    # 4 条质疑里 3 条命中；总共 5 个错误。
    rows = [(True, True), (True, True), (True, True), (True, False)]
    rows += [(False, False)] * 14 + [(False, True), (False, True)]
    metrics = challenge_metrics(rows)
    assert metrics["challenged"] == 4
    assert metrics["hits"] == 3
    assert metrics["hit_rate"] == 0.75
    assert metrics["rule_error_count"] == 5
    assert metrics["error_recall"] == 0.6
    assert metrics["lift_over_error_rate"] == pytest.approx(0.75 - 0.25)


def test_metrics_do_not_reward_challenging_everything():
    """全部质疑时命中率必然等于错误率，lift 为 0——指标必须体现这一点。"""
    rows = [(True, index < 5) for index in range(20)]
    metrics = challenge_metrics(rows)
    assert metrics["hit_rate"] == 0.25
    assert metrics["lift_over_error_rate"] == 0.0


def test_version_is_pinned():
    assert CHALLENGE_PROMPT_VERSION == "rca-challenge-v1"
    assert ASSESSMENTS == ("agree", "challenge")
