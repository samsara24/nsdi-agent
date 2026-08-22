"""T5 测试：N4 分流路由（M4）与 N5a / N5b / N5c / N6 三分支处理器。

除了常规的分支行为，这里锁定三条不能退化的性质：

1. 确定性物理排除**永远不能排掉真实标签**。这是约束库正确性的全量校验，
   `C6` 当初就是被这条测试抓出来的（14 次触发排错 2 次），修法是补上前置约束 `C15`。
2. 置信度只能用训练集留一法标定，用留出集标定要报错。
3. 弃权时不给结论，不退回类别先验。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rca_framework.anomaly import fit_thresholds
from rca_framework.branches import fit_calibration, handle, handle_many
from rca_framework.branches.base import wilson_lower_bound
from rca_framework.branches.general import (
    build_request,
    deterministic_exclusions,
    relevant_constraints,
)
from rca_framework.branches.partial import critical_missing
from rca_framework.data import load_cases
from rca_framework.evidence_graph import (
    BOARD_POLICY,
    COVERAGE_POLICY,
    FILTERED_RULE_THREE_CHANNEL_POLICY,
    EvidenceGraph,
    match_many,
    policy_for,
    route,
    route_many,
    routing_summary,
)
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.features.extractor import extract_features, fit_feature_model
from rca_framework.types import ROOT_CAUSES


DATA_DIR = Path("datasets/organized_rca_v2_stratified_60_40_seed42")
TRAIN_SIZE = 126


@pytest.fixture(scope="module")
def world():
    cases = load_cases(DATA_DIR)
    train, test = cases[:TRAIN_SIZE], cases[TRAIN_SIZE:]
    thresholds = fit_thresholds(train)
    train_packs, test_packs = build_packs(train), build_packs(test)
    model = fit_feature_model(train_packs)
    train_features = [extract_features(pack, thresholds, model) for pack in train_packs]
    test_features = [extract_features(pack, thresholds, model) for pack in test_packs]
    graph = EvidenceGraph.build(train_features, labels_of(train), feature_model=model)
    return {
        "graph": graph,
        "train_packs": train_packs,
        "test_packs": test_packs,
        "train_labels": labels_of(train),
        "test_labels": labels_of(test),
        "train_results": match_many(graph, train_features, top_k=0, leave_one_out=True),
        "test_results": match_many(graph, test_features, top_k=0),
        "all_packs": build_packs(cases),
        "all_labels": labels_of(cases),
    }


# --- M4 路由 ---------------------------------------------------------------

def test_policies_are_registered_and_serializable():
    assert policy_for("board-100-70") is BOARD_POLICY
    assert policy_for("coverage-v2") is COVERAGE_POLICY
    assert policy_for("filtered-rule-three-channel-v1") is FILTERED_RULE_THREE_CHANNEL_POLICY
    with pytest.raises(KeyError, match="unknown routing policy"):
        policy_for("nope")
    assert json.loads(json.dumps(COVERAGE_POLICY.to_dict()))["partial_similarity"] is None


def test_board_policy_reproduces_the_whiteboard_split(world):
    decisions = route_many(world["test_results"], BOARD_POLICY)
    summary = routing_summary(decisions)
    assert summary["counts"] == {"N5a": 21, "N5b": 26, "N5c": 38, "N6": 0}
    assert summary["policy"] == "board-100-70"


def test_coverage_policy_split(world):
    decisions = route_many(world["test_results"], COVERAGE_POLICY)
    summary = routing_summary(decisions)
    # C15 把 2 条全链路遥测失效的 case 从 N5a / N5b 拉进 N6。
    assert summary["counts"] == {"N5a": 20, "N5b": 17, "N5c": 46, "N6": 2}


def test_filtered_rule_policy_has_exactly_three_pre_inference_channels(world):
    decisions = route_many(world["test_results"], FILTERED_RULE_THREE_CHANNEL_POLICY)
    summary = routing_summary(decisions)
    assert summary["counts"]["N6"] == 0
    assert sum(summary["counts"][branch] for branch in ("N5a", "N5b", "N5c")) == len(decisions)
    for result in world["test_results"]:
        assert route(result, FILTERED_RULE_THREE_CHANNEL_POLICY).branch in {"N5a", "N5b", "N5c"}


def test_routing_reason_is_human_readable(world):
    for result in world["test_results"][:20]:
        decision = route(result, COVERAGE_POLICY)
        assert decision.reason and not decision.reason.startswith("N5")
        assert decision.branch in {"N5a", "N5b", "N5c", "N6"}


def test_zero_evidence_abstains_only_under_coverage_policy(world):
    empty = [item for item in world["train_results"] if not item.query_tokens]
    assert len(empty) == 2
    for result in empty:
        assert route(result, COVERAGE_POLICY).branch == "N6"
        assert route(result, BOARD_POLICY).branch == "N5c"


def test_optical_blackout_is_routed_to_human(world):
    """C15：十几个 token 全部来自一条失效的采集通道，token 多不等于证据强。"""
    blackout = [item for item in world["test_results"] if item.query_optical_blackout]
    assert blackout, "held-out split should contain blackout cases"
    for result in blackout:
        assert len(result.query_tokens) > 10  # 看起来证据充分
        assert route(result, COVERAGE_POLICY).branch == "N6"
        assert route(result, BOARD_POLICY).branch in {"N5a", "N5b", "N5c"}


# --- 物理约束的正确性全量校验 ----------------------------------------------

def test_deterministic_exclusion_never_excludes_the_true_label(world):
    """全量 211 条上，确定性排除不得排掉真实标签。

    这条测试当初抓出了 `C6` 的缺陷：未加 `C15` 前置条件时它触发 14 次、排错 2 次。
    任何新增的可执行排除约束都必须先过这一关才能上线。
    """
    violations = []
    for pack, label in zip(world["all_packs"], world["all_labels"]):
        for exclusion in deterministic_exclusions(pack):
            if exclusion.root_cause == label:
                violations.append((pack.case_id, label, exclusion.constraint_id))
    assert violations == []


def test_c6_fires_on_the_expected_number_of_cases(world):
    fired = [pack for pack in world["all_packs"] if deterministic_exclusions(pack)]
    assert len(fired) == 8
    blackout = [pack for pack in world["all_packs"] if pack.optical_blackout]
    assert len(blackout) == 6
    # 全黑 case 必须被 C15 挡住，不允许触发 C6。
    for pack in blackout:
        assert deterministic_exclusions(pack) == ()


def test_relevant_constraints_are_filtered_not_dumped(world):
    """不能把 15 条约束无差别塞进 prompt。"""
    thresholds = fit_thresholds(load_cases(DATA_DIR)[:TRAIN_SIZE])
    model = fit_feature_model(world["train_packs"])
    sizes = set()
    for pack in world["all_packs"]:
        features = extract_features(pack, thresholds, model)
        constraints = relevant_constraints(features.tokens)
        sizes.add(len(constraints))
        # 量测有效性类无条件注入：它们的作用是阻止无效推理。
        assert any(item.category == "measurement_validity" for item in constraints)
    assert min(sizes) < 15


def test_diagnosis_request_carries_exclusions_and_candidates(world):
    for result, pack in zip(world["test_results"], world["test_packs"]):
        request = build_request(result, pack)
        excluded = {item.root_cause for item in request.exclusions}
        assert set(request.candidate_root_causes) == set(ROOT_CAUSES) - excluded
        assert json.loads(json.dumps(request.to_dict(), ensure_ascii=False))


# --- 置信度标定 -------------------------------------------------------------

def test_wilson_lower_bound_is_conservative_on_small_samples():
    assert wilson_lower_bound(2, 2) < 1.0  # 正态近似会错误地给出 1.0
    assert wilson_lower_bound(12, 14) == pytest.approx(0.6006, abs=1e-3)
    assert wilson_lower_bound(0, 0) == 0.0
    # 同样的比例，样本越大下界越高。
    assert wilson_lower_bound(80, 100) > wilson_lower_bound(8, 10)


def test_calibration_is_fitted_on_train_loo_only(world):
    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    assert calibration.source == "train-loo:coverage-v2"
    assert calibration.support("N5a_pure") + calibration.support("N5a_mixed") > 0
    with pytest.raises(ValueError, match="same length"):
        fit_calibration(world["train_results"], world["train_packs"], world["test_labels"])


def test_n5a_is_split_by_bucket_purity(world):
    """AGENTS.md 硬要求：N5a 必须先校验 signature 标签纯净度。"""
    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    pure = calibration.confidence("N5a_pure")
    mixed = calibration.confidence("N5a_mixed")
    assert calibration.support("N5a_pure") > 0 and calibration.support("N5a_mixed") > 0
    assert pure > mixed, "纯桶的置信度必须高于混合桶，否则这个拆分没有意义"


def test_confidence_is_measured_not_hardcoded(world):
    """置信度必须来自标定表，换一份标定表就应该跟着变。"""
    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    paired = handle_many(world["test_results"], world["test_packs"], calibration, policy=COVERAGE_POLICY)
    for _, outcome in paired:
        if outcome.verdict is None:
            continue
        assert outcome.confidence == calibration.confidence(outcome.calibration_group)
        assert outcome.confidence_lower_bound <= outcome.confidence
        assert outcome.calibration_support > 0


# --- 分支行为 ---------------------------------------------------------------

def test_n5a_outcome_reuses_history_and_reports_purity(world):
    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    for result, pack in zip(world["test_results"], world["test_packs"]):
        decision, outcome = handle(result, pack, calibration, policy=COVERAGE_POLICY)
        if decision.branch != "N5a":
            continue
        assert outcome.verdict in ROOT_CAUSES
        assert outcome.reused_case_ids
        assert outcome.missing_evidence == ()
        kinds = {link.kind for link in outcome.evidence_chain}
        assert "exact_match" in kinds
        assert kinds & {"purity_check", "purity_warning"}
        # 全覆盖口径下，纯桶也要进入 LLM，历史结论并行保留。
        assert outcome.needs_llm
        assert outcome.history_verdict in ROOT_CAUSES


def test_n5b_flags_critical_missing_evidence(world):
    assert critical_missing(("status:L1:RxLOS", "level:L2:rxpower_mean:low_tail")) == (
        "status:L1:RxLOS",
        "level:L2:rxpower_mean:low_tail",
    )
    assert critical_missing(("fence:L2:rxpower_mean:low_tail",)) == ()
    assert critical_missing(("drop:L1:txpower:all_lanes",)) == ("drop:L1:txpower:all_lanes",)
    # 正常带内发送电平不是归因证据（P4/C21）；缺失不得触发关键仲裁。
    assert critical_missing(("level:L2:txpower_mean:low_tail",)) == ()
    # 接收类 level 仍是关键。
    assert critical_missing(("level:L1:media_snr_min:high_tail",)) == (
        "level:L1:media_snr_min:high_tail",
    )

    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    seen = False
    for result, pack in zip(world["test_results"], world["test_packs"]):
        decision, outcome = handle(result, pack, calibration, policy=COVERAGE_POLICY)
        if decision.branch != "N5b":
            continue
        seen = True
        assert outcome.verdict in ROOT_CAUSES
        if critical_missing(outcome.missing_evidence):
            assert outcome.needs_llm
            assert outcome.caveats
    assert seen


def test_n5c_does_not_guess_without_an_llm(world):
    """N5c 在 T6 之前不给结论，不退回类别先验。"""
    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    count = 0
    for result, pack in zip(world["test_results"], world["test_packs"]):
        decision, outcome = handle(result, pack, calibration, policy=COVERAGE_POLICY)
        if decision.branch != "N5c":
            continue
        count += 1
        assert outcome.verdict is None
        assert outcome.needs_llm
        assert outcome.caveats
        assert any(link.kind == "constraint_context" for link in outcome.evidence_chain)
    assert count == 46


def test_handle_many_executes_all_marked_llm_arbitrations(world):
    from rca_framework.llm import DiagnosisResponse, ReasoningTrace

    class RecordingReasoner:
        def __init__(self):
            self.branches = []

        def reason_many(self, requests, packs):
            assert len(requests) == len(packs)
            self.branches.extend(request.branch for request in requests)
            return [
                ReasoningTrace(
                    case_id=request.case_id,
                    accepted=DiagnosisResponse(
                        verdict=request.candidate_root_causes[0],
                        confidence=0.8,
                    ),
                    backend_name="recording",
                )
                for request in requests
            ]

    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    reasoner = RecordingReasoner()
    traces = {}
    paired = handle_many(
        world["test_results"],
        world["test_packs"],
        calibration,
        policy=COVERAGE_POLICY,
        reasoner=reasoner,
        trace_collector=traces,
    )

    assert {"N5a", "N5b", "N5c"} <= set(reasoner.branches)
    targeted = [
        outcome for _, outcome in paired
        if outcome.case_id in traces
    ]
    assert targeted and all(not outcome.needs_llm for outcome in targeted)
    assert all(outcome.calibration_group.startswith("uncalibrated:llm:") for outcome in targeted)


def test_n6_abstains_and_asks_for_a_human(world):
    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    count = 0
    for result, pack in zip(world["test_results"], world["test_packs"]):
        decision, outcome = handle(result, pack, calibration, policy=COVERAGE_POLICY)
        if decision.branch != "N6":
            continue
        count += 1
        assert outcome.verdict is None
        assert outcome.needs_human
        assert outcome.needs_llm
        assert outcome.confidence == 0.0
    assert count == 2


def test_t5_end_to_end_numbers(world):
    """锁定 T5 的验收数字，防止后续改动悄悄推翻结论。"""
    expectations = {
        BOARD_POLICY.name: (47, 32),
        COVERAGE_POLICY.name: (37, 29),
    }
    for policy in (BOARD_POLICY, COVERAGE_POLICY):
        calibration = fit_calibration(
            world["train_results"], world["train_packs"], world["train_labels"], policy=policy
        )
        paired = handle_many(world["test_results"], world["test_packs"], calibration, policy=policy)
        answered = [
            (outcome, truth)
            for (_, outcome), truth in zip(paired, world["test_labels"])
            if outcome.verdict is not None
        ]
        correct = sum(outcome.verdict == truth for outcome, truth in answered)
        assert (len(answered), correct) == expectations[policy.name], policy.name


def test_outcomes_are_json_serializable(world):
    calibration = fit_calibration(
        world["train_results"], world["train_packs"], world["train_labels"], policy=COVERAGE_POLICY
    )
    paired = handle_many(world["test_results"], world["test_packs"], calibration, policy=COVERAGE_POLICY)
    payload = json.dumps(
        [{"decision": d.to_dict(), "outcome": o.to_dict()} for d, o in paired],
        ensure_ascii=False,
    )
    assert json.loads(payload)
