"""证据 token 语义等价类的测试。

这组测试的重点不是「等价类能匹配上」，而是**它没有顺手放宽方向**：
迭代 3 的全部增益来自「接收类观测指向对端」这条方向知识，
如果等价类允许跨侧或跨指标匹配，方向约束 C23-C25 就会失去意义。
"""

from __future__ import annotations

import pytest

from rca_framework.constraints.library import CONSTRAINT_LIBRARY
from rca_framework.constraints.semantics import matches_scope, prefix_scope, token_scope


@pytest.mark.parametrize(
    "token,expected",
    [
        ("drop:L1:rxpower:single_lane", ("L1", "rxpower")),
        ("expert:L1:rxpower:lane_down", ("L1", "rxpower")),
        ("level:L1:rxpower_mean:low_tail", ("L1", "rxpower")),
        ("level:L2:media_snr_min:high_tail", ("L2", "media_snr")),
        ("imbalance:L2:serdes_snr", ("L2", "serdes_snr")),
        ("serdes:L1:valid", ("L1", "serdes_snr")),
        ("status:L1:RxLOS", ("L1", "rxpower")),
        ("status:L2:TxLOS", ("L2", "txpower")),
        ("expert:pattern:L1:port_down", ("L1", None)),
        ("expert:points_to:L1:L2", ("L1", None)),
    ],
)
def test_token_scope_reduces_families_to_side_and_metric(token, expected):
    assert token_scope(token) == expected


@pytest.mark.parametrize(
    "token",
    ["lane:L1_to_L2:tx_down", "telemetry:partial_telemetry", "nonsense"],
)
def test_unreducible_tokens_return_none(token):
    assert token_scope(token) is None


def test_same_side_same_metric_matches_across_families():
    """本轮改动的目标：同侧同指标的不同判据族互相承认。"""
    prefixes = ("expert:L1:txpower:",)
    for token in (
        "level:L1:txpower_mean:low_tail",
        "drop:L1:txpower:all_lanes",
        "status:L1:TxLOS",
    ):
        assert matches_scope(token, prefixes), token


def test_other_side_never_matches():
    """跨侧匹配会直接摧毁方向约束，必须拒绝。"""
    prefixes = ("expert:L1:rxpower:",)
    for token in (
        "level:L2:rxpower_mean:low_tail",
        "drop:L2:rxpower:single_lane",
        "expert:L2:rxpower:lane_down",
    ):
        assert not matches_scope(token, prefixes), token


def test_other_metric_never_matches():
    prefixes = ("expert:L1:rxpower:",)
    for token in ("expert:L1:txpower:lane_down", "level:L1:media_snr_min:low_tail"):
        assert not matches_scope(token, prefixes), token


def test_side_level_tokens_require_literal_prefix():
    """`expert:pattern:*` 没有指标坐标，只能字面匹配，不得靠等价类蒙混。"""
    assert matches_scope("expert:pattern:L1:port_down", ("expert:pattern:L1:port_down",))
    assert not matches_scope("expert:pattern:L1:multi_metric", ("expert:L1:txpower:",))
    assert not matches_scope("expert:points_to:L1:L1", ("expert:L1:host_snr:",))


def test_direction_constraints_keep_disjoint_scopes():
    """C23 与 C24 是镜像的一对：任何一个 token 不得同时落进两者的适用范围。"""
    c23 = CONSTRAINT_LIBRARY.get("C23_expert_receive_anomaly_on_l1_supports_l2")
    c24 = CONSTRAINT_LIBRARY.get("C24_expert_receive_anomaly_on_l2_supports_l1")
    tokens = [
        f"{family}:{side}:{metric}{suffix}"
        for family, suffix in (("expert", ":lane_down"), ("drop", ":single_lane"))
        for side in ("L1", "L2")
        for metric in ("rxpower", "media_snr", "txpower", "serdes_snr")
    ]
    for token in tokens:
        in_c23 = matches_scope(token, c23.applies_to_token_prefixes)
        in_c24 = matches_scope(token, c24.applies_to_token_prefixes)
        assert not (in_c23 and in_c24), token


def test_local_chain_constraint_does_not_absorb_receive_evidence():
    """C25 支持本端，绝不能被接收类证据触发，否则等于取消方向表。"""
    c25 = CONSTRAINT_LIBRARY.get("C25_expert_local_chain_anomaly_on_l1_supports_l1")
    for token in (
        "level:L1:rxpower_mean:low_tail",
        "drop:L1:media_snr:all_lanes",
        "expert:L1:rxpower:lane_down",
    ):
        assert not matches_scope(token, c25.applies_to_token_prefixes), token


def test_bare_family_prefix_is_side_wildcard():
    """C13 的契约是裸 `serdes:`，两端的同指标证据都应当接受。"""
    assert prefix_scope("serdes:") == (None, "serdes_snr")
    assert matches_scope("expert:L1:serdes_snr:low_value", ("serdes:",))
    assert matches_scope("imbalance:L2:serdes_snr", ("serdes:",))
    assert not matches_scope("expert:L1:rxpower:lane_down", ("serdes:",))
