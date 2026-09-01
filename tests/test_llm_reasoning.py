"""T6 测试：M8 输出协议、prompt 模板与受约束推理循环。

重点锁定验收要求「LLM 每步输出可被约束校验；不合规可回退或重写」，
以及全覆盖口径下「重写用尽后低置信强制产出，而不是无结论退出」。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rca_framework.anomaly import fit_thresholds
from rca_framework.branches import fit_calibration, handle_many
from rca_framework.branches.general import build_request
from rca_framework.data import load_cases
from rca_framework.evidence_graph import COVERAGE_POLICY, EvidenceGraph, match_many, route
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.features.extractor import extract_features, fit_feature_model
from rca_framework.llm import (
    ConstrainedReasoner,
    NoneBackend,
    PathLLMReasoner,
    ScriptedBackend,
    build_prompt,
    parse_response,
)
from rca_framework.llm.prompts import (
    FILTERED_RULE_PROMPT_TEMPLATE_VERSION,
    PROMPT_TEMPLATE_VERSION,
    prompt_template_hash,
    prompt_template_version_for,
)
from rca_framework.llm.protocol import DIAGNOSIS_OUTPUT_SCHEMA


DATA_DIR = Path("datasets/organized_rca_v2_stratified_60_40_seed42")
TRAIN_SIZE = 126


@pytest.fixture(scope="module")
def n5c():
    cases = load_cases(DATA_DIR)
    train, test = cases[:TRAIN_SIZE], cases[TRAIN_SIZE:]
    thresholds = fit_thresholds(train)
    train_packs, test_packs = build_packs(train), build_packs(test)
    model = fit_feature_model(train_packs)
    train_features = [extract_features(pack, thresholds, model) for pack in train_packs]
    test_features = [extract_features(pack, thresholds, model) for pack in test_packs]
    graph = EvidenceGraph.build(train_features, labels_of(train), feature_model=model)
    results = match_many(graph, test_features, top_k=0)
    picked = [
        (result, pack) for result, pack in zip(results, test_packs)
        if route(result, COVERAGE_POLICY).branch == "N5c" and result.query_tokens
    ]
    result, pack = picked[0]
    return {
        "result": result,
        "pack": pack,
        "request": build_request(result, pack),
        "graph": graph,
        "results": results,
        "test_packs": test_packs,
        "train_results": match_many(graph, train_features, top_k=0, leave_one_out=True),
        "train_packs": train_packs,
        "train_labels": labels_of(train),
    }


def answer(tokens, verdict="L2", constraints=()):
    return json.dumps({
        "steps": [{
            "claim": "接收侧功率处于低尾档",
            "cited_evidence": list(tokens),
            "cited_constraints": list(constraints),
            "effect": "support",
            "target": verdict,
        }],
        "verdict": verdict,
        "confidence": 0.7,
        "confidence_breakdown": {
            "evidence_completeness": 0.6,
            "physical_compliance": 0.7,
            "reasoning_completeness": 0.6,
            "history_similarity": 0.5,
        },
        "missing_information": ["对端同 lane 的收光读数"],
    }, ensure_ascii=False)


# --- 协议 -------------------------------------------------------------------

def test_parser_rejects_anything_it_cannot_verify():
    assert parse_response("") is None
    assert parse_response("完全不是 JSON") is None
    assert parse_response('{"verdict": "L2"}') is None  # 缺 steps
    assert parse_response('{"steps": [], "verdict": "外星人"}') is None
    assert parse_response('{"steps": [{"claim":"x","effect":"maybe"}], "verdict": "L2"}') is None
    assert parse_response('{"steps": [{"claim":"x","effect":"support","target":"L9"}], "verdict":"L2"}') is None


def test_parser_handles_reasoning_model_thinking_blocks():
    """DeepSeek-R1 系列会先输出 `<think>` 段，里面常有半成品 JSON。

    必须取思考段之后的最终答案，而不是草稿。贪婪正则在这里会从思考段的第一个
    左括号一路吃到最后一个右括号，抓出一段无效文本。
    """
    raw = (
        "<think>\n先看证据。也许可以写成 {\"verdict\": \"L1\"}，"
        "但再想想 {\"steps\": [] } 好像不对。\n</think>\n\n"
        + answer(("level:L2:rxpower_mean:low_tail",), verdict="L2")
    )
    parsed = parse_response(raw)
    assert parsed is not None
    assert parsed.verdict == "L2"
    assert parsed.cited_evidence == ("level:L2:rxpower_mean:low_tail",)


def test_parser_takes_the_last_object_when_several_are_present():
    raw = "先给个例子 {\"steps\": [], \"verdict\": \"L1\"}\n最终答案：\n" + answer(("t",), verdict="fiber")
    parsed = parse_response(raw)
    assert parsed is not None and parsed.verdict == "fiber"


def test_parser_rejects_truncated_output():
    """思考被 max_tokens 截断时拿不到完整 JSON，必须判为不合规而不是猜。"""
    assert parse_response("<think>思考中... {\"steps\": [{\"claim\": \"未完") is None


def test_parser_is_not_confused_by_braces_inside_strings():
    raw = json.dumps({
        "steps": [{
            "claim": "证据里出现了 } 这个字符 { 也出现了",
            "cited_evidence": ["level:L2:rxpower_mean:low_tail"],
            "cited_constraints": [], "effect": "support", "target": "L2",
        }],
        "verdict": "L2", "confidence": 0.5,
        "confidence_breakdown": {
            "evidence_completeness": 0.5,
            "physical_compliance": 0.5,
            "reasoning_completeness": 0.5,
            "history_similarity": 0.5,
        },
        "missing_information": [],
    }, ensure_ascii=False)
    parsed = parse_response(raw)
    assert parsed is not None and parsed.verdict == "L2"


def test_parser_rejects_abstain_for_forced_coverage():
    parsed = parse_response(
        '{"steps": [], "verdict": "abstain", "confidence": 0.1, '
        '"confidence_breakdown": {"evidence_completeness": 0.1, "physical_compliance": 0.1, '
        '"reasoning_completeness": 0.1, "history_similarity": 0.1}, "missing_information": []}'
    )
    assert parsed is None


def test_parser_clamps_confidence_and_collects_citations():
    parsed = parse_response(answer(("a", "b")).replace('"confidence": 0.7', '"confidence": 5'))
    assert parsed.self_reported_confidence == 1.0
    assert parsed.confidence == pytest.approx(0.61)
    assert parsed.cited_evidence == ("a", "b")
    assert parsed.cited_constraints == ()


def test_excluded_root_causes_are_structured_not_textual():
    parsed = parse_response(json.dumps({
        "steps": [
            {"claim": "本端无光", "cited_evidence": ["drop:L1:txpower:all_lanes"],
             "cited_constraints": ["C6_tx_down_excludes_medium"], "effect": "exclude", "target": "fiber"},
        ],
        "verdict": "L1", "confidence": 0.6,
        "confidence_breakdown": {
            "evidence_completeness": 0.6,
            "physical_compliance": 0.6,
            "reasoning_completeness": 0.6,
            "history_similarity": 0.6,
        },
        "missing_information": [],
    }, ensure_ascii=False))
    assert parsed.excluded_root_causes() == ("fiber",)


def test_output_schema_matches_the_dataclass():
    required = set(DIAGNOSIS_OUTPUT_SCHEMA["required"])
    assert required == {"steps", "verdict", "confidence", "confidence_breakdown", "missing_information"}
    assert "abstain" not in DIAGNOSIS_OUTPUT_SCHEMA["properties"]["verdict"]["enum"]


# --- prompt -----------------------------------------------------------------

def test_prompt_contains_evidence_constraints_and_forced_option(n5c):
    prompt = build_prompt(n5c["request"])
    for token in n5c["request"].evidence_tokens:
        assert token in prompt
    assert "P5_tx_down_excludes_medium" in prompt
    assert "M6_fiber_not_identifiable_without_field_evidence" in prompt
    assert "专家排障 SOP" in prompt
    assert "S1_collect_anomaly_level" in prompt
    assert "禁止输出 abstain" in prompt
    assert "纯物理约束" in prompt
    assert "量测契约" in prompt


def test_prompt_only_injects_relevant_constraints(n5c):
    """低档不再把旧 C 约束库无差别塞进去。"""
    from rca_framework.constraints.library import CONSTRAINT_LIBRARY

    prompt = build_prompt(n5c["request"])
    assert len(n5c["request"].constraint_ids) < len(CONSTRAINT_LIBRARY.constraints)
    assert "C2_bias_healthy_band" not in prompt
    assert "Wilson" not in prompt


def test_prompt_orders_exclusions_before_indicators(n5c):
    prompt = build_prompt(n5c["request"])
    assert prompt.index("# 纯物理约束") < prompt.index("# 量测契约")


def test_retry_feedback_is_labelled_as_a_past_mistake(n5c):
    prompt = build_prompt(n5c["request"], retry_feedback="- 第 1 步：引用了不存在的证据")
    assert "上一次回答未通过物理约束校验" in prompt
    # 反馈必须在证据之前出现，且明确是「上一次」的问题，不能被当成新证据。
    assert prompt.index("上一次回答未通过") < prompt.index("本 case 证据")


def test_prompt_is_deterministic(n5c):
    assert build_prompt(n5c["request"]) == build_prompt(n5c["request"])
    assert PROMPT_TEMPLATE_VERSION == "rca-dual-sop-multidim-v14-full-step-ids"
    assert len(prompt_template_hash()) == 16


def test_filtered_rule_prompt_has_independent_topology_contract(n5c):
    from dataclasses import replace

    request = replace(
        n5c["request"],
        topology_context={
            "contract_version": "filtered-rule-topology-v1",
            "source_dataset": "rule1_channel_not_4",
            "topology_id": "400g-400g-logical8",
        },
    )
    prompt = build_prompt(request)
    assert "当前 case 本端的设备或端口根因" in prompt
    assert '"source_dataset": "rule1_channel_not_4"' in prompt
    assert prompt_template_version_for(request) == FILTERED_RULE_PROMPT_TEMPLATE_VERSION
    assert prompt_template_hash("filtered_rule_v1") != prompt_template_hash()


def test_filtered_rule_prompt_uses_general_optional_reasoning_contract(n5c):
    from dataclasses import replace

    request = replace(
        n5c["request"],
        branch="N5b",
        topology_context={"contract_version": "filtered-rule-topology-v1"},
    )
    prompt = build_prompt(request)
    assert "不要求固定步骤数" in prompt
    assert "sop_step_id 和 cited_predicates 都是可选字段" in prompt
    assert "Q0_validate_measurements → P_apply_physical_boundaries" not in prompt
    assert "current_physical_evidence_paths" in prompt
    assert "historical_evidence_chains" in prompt


def test_filtered_rule_prompt_treats_host_snr_as_optional_enhancement(n5c):
    from dataclasses import replace

    request = replace(
        n5c["request"],
        missing_fields=("L1.host_snr", "L2.host_snr"),
        topology_context={"contract_version": "filtered-rule-topology-v1"},
    )
    prompt = build_prompt(request)
    assert "host_snr 是可选增强证据" in prompt
    assert "缺失时不扣分、不要求补采" in prompt
    assert "只有对端 TxLOS/TxLOL" in prompt
    assert '"missing_fields": []' in prompt


def test_prompt_exposes_checker_effect_target_and_token_contracts(n5c):
    prompt = build_prompt(n5c["request"])
    assert "结构化引用契约" in prompt
    assert "effect 只能为" in prompt
    assert "target 只能为" in prompt
    assert "可用 token 前缀=" in prompt
    assert "严禁写入 `cited_constraints`" in prompt
    assert "违反后该推理步骤或结论作废" in prompt


def test_prompt_injects_numeric_tree_as_advisory_not_evidence(n5c):
    from dataclasses import replace

    request = replace(
        n5c["request"],
        decision_tree_prediction={
            "verdict": "L2",
            "leaf_id": "root.present",
            "path": ["L1.rxpower.min <= -2.5"],
            "support": 17,
            "confidence": 0.7,
        },
    )
    prompt = build_prompt(request)
    assert "numeric_decision_tree_path" in prompt
    assert "量测契约只能否决不可信推理" in prompt
    assert '"leaf_id": "root.present"' in prompt


def test_prompt_is_branch_aware_for_arbitration(n5c):
    from dataclasses import replace

    request = replace(
        n5c["request"],
        branch="N5a",
        routing_reason="完全匹配桶标签不纯，需要仲裁",
        historical_label_distribution=(("L1", 2), ("L2", 3)),
    )
    prompt = build_prompt(request)
    assert "N5a 分支" in prompt
    assert "历史标签不纯" in prompt
    assert '"L1": 2' in prompt and '"L2": 3' in prompt
    assert "专家排障 SOP" not in prompt


def test_prompt_separates_incomplete_telemetry_from_insufficient_evidence(n5c):
    """真机实测暴露的失败模式：模型把「有字段没采集」当成「证据不足」而全部弃权。

    C14 已经写明 host_snr 常态缺失，按那个标准会对所有 case 弃权。
    弃权判据必须落在「可用证据能否区分候选根因」上，这条不能被改回去。
    """
    prompt = build_prompt(n5c["request"])
    assert "不是拒答理由" in prompt
    assert "证据不足必须体现为低 evidence_completeness" in prompt
    assert "禁止输出 abstain" in prompt
    assert "L1/L2/fiber 三选一" in prompt


# --- 推理循环 ---------------------------------------------------------------

def test_none_backend_forces_low_confidence_fallback(n5c):
    reasoner = ConstrainedReasoner(backend=NoneBackend())
    trace = reasoner.reason(n5c["request"], n5c["pack"])
    assert trace.accepted is not None
    assert trace.accepted.verdict in ("L1", "L2", "fiber")
    assert trace.accepted.confidence == 0.0
    assert trace.accepted.fallback_source == "parse_failure"
    assert "强制低置信兜底" in trace.degradation_reason
    assert trace.backend_name == "none"


def test_compliant_answer_is_accepted_on_the_first_try(n5c):
    tokens = n5c["request"].evidence_tokens
    backend = ScriptedBackend(responses=[[answer(tokens[:1])]])
    trace = ConstrainedReasoner(backend=backend).reason(n5c["request"], n5c["pack"])
    assert trace.accepted is not None
    assert trace.accepted.verdict == "L2"
    assert trace.attempt_count == 1
    assert not trace.rewrote
    assert "上一次回答未通过" not in backend.prompts_seen[0][0]
    assert "这是最后一轮" not in backend.prompts_seen[0][0]


def test_violating_answer_triggers_a_rewrite_with_feedback(n5c):
    """第一轮引用虚构证据，第二轮改对，最终接受。"""
    tokens = n5c["request"].evidence_tokens
    backend = ScriptedBackend(responses=[
        [answer(("drop:L9:fake:all_lanes",))],
        [answer(tokens[:1])],
    ])
    trace = ConstrainedReasoner(backend=backend, max_attempts=2).reason(n5c["request"], n5c["pack"])

    assert trace.attempt_count == 2
    assert trace.rewrote
    assert trace.accepted is not None and trace.accepted.verdict == "L2"
    assert not trace.attempts[0].check.ok
    assert trace.attempts[1].check.ok
    # 第二轮的 prompt 必须带上第一轮的违规原因。
    second_prompt = backend.prompts_seen[1][0]
    assert "drop:L9:fake:all_lanes" in second_prompt
    assert "上一次回答未通过物理约束校验" in second_prompt


def test_persistent_violation_ends_in_low_confidence_forced_answer(n5c):
    """重写用尽仍不合规时保留低置信候选，并把 fatal 作为扣分明细落盘。"""
    backend = ScriptedBackend(responses=[
        [answer(("drop:L9:fake:all_lanes",))],
        [answer(("drop:L8:also_fake:all_lanes",))],
    ])
    trace = ConstrainedReasoner(backend=backend, max_attempts=2).reason(n5c["request"], n5c["pack"])
    assert trace.accepted is not None
    assert trace.accepted.verdict == "L2"
    assert trace.accepted.confidence_breakdown.physical_compliance == 0.0
    # history_similarity 不再采信模型自评，改由检索到的最高相似度直接给出。
    similarity = n5c["request"].nearest_similarity
    assert trace.accepted.confidence_breakdown.history_similarity == pytest.approx(similarity)
    assert trace.accepted.confidence == pytest.approx(0.3 + 0.2 * similarity)
    assert trace.accepted.fallback_source == "last_parsed_after_fatal"
    assert trace.attempt_count == 2
    assert "强制低置信兜底" in trace.degradation_reason
    assert all(not item.check.ok for item in trace.attempts)


def test_trace_records_every_attempt_for_the_log(n5c):
    tokens = n5c["request"].evidence_tokens
    backend = ScriptedBackend(responses=[[answer(("fake:token",))], [answer(tokens[:1])]])
    trace = ConstrainedReasoner(backend=backend, max_attempts=2).reason(n5c["request"], n5c["pack"])
    payload = json.loads(json.dumps(trace.to_dict(), ensure_ascii=False))
    assert payload["attempt_count"] == 2
    assert payload["rewrote"] is True
    assert payload["constraint_library_version"] == "constraint-library-v6"
    assert payload["prompt_version"] == PROMPT_TEMPLATE_VERSION
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["check"]["fatal_count"] > 0


def test_batch_reasoning_only_retries_the_failing_cases(n5c):
    """已通过的 case 不该在重写轮里再消耗一次生成。"""
    request, pack = n5c["request"], n5c["pack"]
    tokens = request.evidence_tokens
    backend = ScriptedBackend(responses=[
        [answer(tokens[:1]), answer(("fake:token",))],
        [answer(tokens[:1])],
    ])
    traces = ConstrainedReasoner(backend=backend, max_attempts=2).reason_many(
        [request, request], [pack, pack]
    )
    assert len(backend.prompts_seen[0]) == 2
    assert len(backend.prompts_seen[1]) == 1, "第二轮只该重发未通过的那一条"
    assert traces[0].attempt_count == 1
    assert traces[1].attempt_count == 2
    assert all(item.accepted is not None for item in traces)


def test_reason_many_rejects_mismatched_lengths(n5c):
    with pytest.raises(ValueError, match="same length"):
        ConstrainedReasoner().reason_many([n5c["request"]], [])


# --- 与分支处理器的集成 -----------------------------------------------------

def test_n5c_uses_the_llm_verdict_and_renders_the_steps(n5c):
    calibration = fit_calibration(
        n5c["train_results"], n5c["train_packs"], n5c["train_labels"], policy=COVERAGE_POLICY
    )
    tokens = n5c["request"].evidence_tokens

    class AlwaysCompliant:
        name = "always-compliant"

        def generate(self, prompts):
            return [answer(tokens[:1]) for _ in prompts]

    paired = handle_many(
        n5c["results"], n5c["test_packs"], calibration, policy=COVERAGE_POLICY,
        reasoner=ConstrainedReasoner(backend=AlwaysCompliant()),
    )
    answered = [
        outcome for decision, outcome in paired
        if decision.branch == "N5c" and outcome.verdict is not None
    ]
    assert answered, "接上 LLM 之后 N5c 应当能给出结论"
    sample = answered[0]
    assert any(link.kind == "llm_step" for link in sample.evidence_chain)
    assert not sample.needs_llm  # 已经跑过了


def test_n5c_still_abstains_when_the_llm_is_absent(n5c):
    calibration = fit_calibration(
        n5c["train_results"], n5c["train_packs"], n5c["train_labels"], policy=COVERAGE_POLICY
    )
    paired = handle_many(n5c["results"], n5c["test_packs"], calibration, policy=COVERAGE_POLICY)
    n5c_outcomes = [outcome for decision, outcome in paired if decision.branch == "N5c"]
    assert n5c_outcomes and all(item.verdict is None for item in n5c_outcomes)
    assert all(item.needs_llm for item in n5c_outcomes)


def test_checker_audit_numbers_are_locked(n5c):
    """锁定 `scripts/audit_constraint_checker.py` 的结果。

    注意这些攻击样本是我们自己按已知失效模式构造的，与校验器的检测规则同源，
    因此 100% 拦截率**不能证明**校验器对真实模型的鲁棒性。它的价值是回归保护：
    以后改动校验器或约束库时，这些拦截不能悄悄失效。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit", Path("scripts/audit_constraint_checker.py")
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    from rca_framework.constraints.checker import check_response

    targets = [
        (result, pack) for result, pack in zip(n5c["results"], n5c["test_packs"])
        if route(result, COVERAGE_POLICY).branch == "N5c" and result.query_tokens
    ]
    assert len(targets) == 46

    for mode in audit.MODES:
        blocked = 0
        for result, pack in targets:
            request = build_request(result, pack)
            report = check_response(
                parse_response(audit.make_response(mode, request)),
                pack, request.evidence_tokens,
                allowed_root_causes=request.candidate_root_causes,
            )
            blocked += int(not report.ok)
        if mode == "compliant":
            assert blocked == 0, "合规回答被拦会让系统陷入无限重写"
        else:
            assert blocked == len(targets), mode


# --- legacy 兼容 -------------------------------------------------------------

def test_legacy_llm_imports_still_work():
    """`from rca_framework.llm import PathLLMReasoner` 是 58/85 锚点的一部分。"""
    assert PathLLMReasoner(backend="none").backend == "none"
    from rca_framework.llm import LLM_OUTPUT_SCHEMA, build_path_prompt, parse_llm_json

    assert LLM_OUTPUT_SCHEMA["required"][0] == "prediction"
    assert callable(build_path_prompt) and callable(parse_llm_json)
