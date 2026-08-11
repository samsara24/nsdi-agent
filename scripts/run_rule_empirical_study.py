"""独立评估“大模型是否真的会使用当前物理规则”的三组配对实验。

三个实验臂使用完全相同的 test case、证据 token、模型和解码参数：

1. evidence_only：只给证据，不给规则，也不做 checker。
2. rules_prompt：给当前相关规则，但不拦截模型输出。
3. rules_prompt_checker：复用第二组首轮输出，只对违规输出做一次或多次重写。

历史 case、历史标签、N5a/N5b 投票和 M9 最终门禁均不进入 prompt。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.branches.general import (  # noqa: E402
    DiagnosisRequest,
    deterministic_exclusions,
    relevant_constraints,
)
from rca_framework.constraints.checker import CheckReport, Violation, check_response  # noqa: E402
from rca_framework.constraints.library import CONSTRAINT_LIBRARY  # noqa: E402
from rca_framework.data import load_cases  # noqa: E402
from rca_framework.evidence_graph import (  # noqa: E402
    COVERAGE_POLICY,
    EvidenceGraph,
    match_many,
    route_many,
)
from rca_framework.evidence_pack import build_packs, labels_of  # noqa: E402
from rca_framework.features.extractor import extract_features, fit_feature_model  # noqa: E402
from rca_framework.llm import backend_for, parse_response  # noqa: E402
from rca_framework.llm.empirical import (  # noqa: E402
    EMPIRICAL_PROMPT_VERSION,
    build_empirical_prompt,
    empirical_prompt_hash,
)
from rca_framework.types import ROOT_CAUSES  # noqa: E402


ARMS: Tuple[str, ...] = ("evidence_only", "rules_prompt", "rules_prompt_checker")


def parse_failure(raw: str) -> CheckReport:
    return CheckReport(violations=(
        Violation(
            kind="unsupported_step",
            severity="fatal",
            message="输出不是符合 schema 的 JSON",
            detail=raw[:200],
        ),
    ))


def rule_check(response, pack, request: DiagnosisRequest, raw: str = "") -> CheckReport:
    if response is None:
        return parse_failure(raw)
    return check_response(
        response,
        pack,
        request.evidence_tokens,
        allowed_root_causes=request.candidate_root_causes,
        library=CONSTRAINT_LIBRARY,
    )


def build_request(pack, features, *, include_rules: bool) -> DiagnosisRequest:
    exclusions = deterministic_exclusions(pack) if include_rules else ()
    excluded = {item.root_cause for item in exclusions}
    constraints = relevant_constraints(features.tokens) if include_rules else ()
    return DiagnosisRequest(
        case_id=pack.case_id,
        evidence_tokens=features.tokens,
        missing_fields=pack.missing_fields,
        telemetry_status=pack.telemetry_status,
        candidate_root_causes=tuple(
            label for label in ROOT_CAUSES if label not in excluded
        ),
        exclusions=tuple(exclusions),
        constraint_ids=tuple(item.constraint_id for item in constraints),
        nearest_similarity=0.0,
        branch="empirical",
        routing_reason="独立规则经验研究，不使用历史匹配结果",
        historical_case_ids=(),
        historical_label_distribution=(),
    )


def raw_arm(backend, prompts, packs, rule_requests):
    outputs = backend.generate(prompts)
    responses = [parse_response(raw) for raw in outputs]
    reports = [
        rule_check(response, pack, request, raw)
        for response, pack, request, raw in zip(responses, packs, rule_requests, outputs)
    ]
    return responses, reports, outputs


def checked_arm(
    backend,
    requests: Sequence[DiagnosisRequest],
    packs,
    initial_prompts: Sequence[str],
    initial_outputs: Sequence[str],
    *,
    max_attempts: int,
):
    """复用 rules_prompt 首轮结果，仅重写 checker 未通过的 case。"""
    attempts: List[List[Dict[str, Any]]] = [[] for _ in requests]
    accepted: List[Optional[Any]] = [None for _ in requests]
    last_reports: List[CheckReport] = []
    pending: List[int] = []

    for index, (request, pack, prompt, raw) in enumerate(
        zip(requests, packs, initial_prompts, initial_outputs)
    ):
        response = parse_response(raw)
        report = rule_check(response, pack, request, raw)
        last_reports.append(report)
        attempts[index].append({
            "index": 0,
            "prompt": prompt,
            "raw_output": raw,
            "parsed": response is not None,
            "check": report.to_dict(),
        })
        if response is not None and report.ok:
            accepted[index] = response
        else:
            pending.append(index)

    for attempt_index in range(1, max(1, max_attempts)):
        if not pending:
            break
        prompts = [
            build_empirical_prompt(
                requests[index],
                include_rules=True,
                retry_feedback=last_reports[index].feedback(),
            )
            for index in pending
        ]
        outputs = backend.generate(prompts)
        next_pending: List[int] = []
        for position, index in enumerate(pending):
            raw = outputs[position] if position < len(outputs) else ""
            response = parse_response(raw)
            report = rule_check(response, packs[index], requests[index], raw)
            last_reports[index] = report
            attempts[index].append({
                "index": attempt_index,
                "prompt": prompts[position],
                "raw_output": raw,
                "parsed": response is not None,
                "check": report.to_dict(),
            })
            if response is not None and report.ok:
                accepted[index] = response
            else:
                next_pending.append(index)
        pending = next_pending
    return accepted, last_reports, attempts


def wilson_interval(successes: int, total: int, z: float = 1.96) -> List[float]:
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * total)) / total)
    return [
        round(max(0.0, (centre - margin) / denominator), 6),
        round(min(1.0, (centre + margin) / denominator), 6),
    ]


def arm_metrics(responses, reports, labels: Sequence[str], indices: Sequence[int]) -> Dict[str, Any]:
    parsed = sum(responses[index] is not None for index in indices)
    predictions = [
        responses[index].verdict if responses[index] is not None else None
        for index in indices
    ]
    truths = [labels[index] for index in indices]
    answered = sum(item is not None for item in predictions)
    correct = sum(prediction == truth for prediction, truth in zip(predictions, truths))
    compliant = sum(reports[index].ok for index in indices)
    violation_counts = Counter(
        violation.kind
        for index in indices
        for violation in reports[index].violations
    )
    by_class: Dict[str, Any] = {}
    for label in ROOT_CAUSES:
        support = sum(truth == label for truth in truths)
        tp = sum(prediction == label and truth == label for prediction, truth in zip(predictions, truths))
        fp = sum(prediction == label and truth != label for prediction, truth in zip(predictions, truths))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_class[label] = {
            "support": support,
            "predicted": tp + fp,
            "correct": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "recall_wilson_95": wilson_interval(tp, support),
        }
    total = len(indices)
    return {
        "case_count": total,
        "parsed": parsed,
        "schema_valid_rate": round(parsed / total, 6) if total else 0.0,
        "answered": answered,
        "answer_rate": round(answered / total, 6) if total else 0.0,
        "abstained_or_invalid": total - answered,
        "correct": correct,
        "accuracy_all_cases": round(correct / total, 6) if total else 0.0,
        "precision_when_answered": round(correct / answered, 6) if answered else None,
        "precision_wilson_95": wilson_interval(correct, answered),
        "rule_compliant": compliant,
        "rule_compliance_rate": round(compliant / total, 6) if total else 0.0,
        "violation_counts": dict(sorted(violation_counts.items())),
        "class_metrics": by_class,
    }


def exact_mcnemar_p(improved: int, worsened: int) -> float:
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    smaller = min(improved, worsened)
    probability = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
    return round(min(1.0, 2.0 * probability), 6)


def paired_comparison(left, right, labels: Sequence[str]) -> Dict[str, Any]:
    left_correct = [
        response is not None and response.verdict is not None and response.verdict == truth
        for response, truth in zip(left, labels)
    ]
    right_correct = [
        response is not None and response.verdict is not None and response.verdict == truth
        for response, truth in zip(right, labels)
    ]
    improved = sum(not old and new for old, new in zip(left_correct, right_correct))
    worsened = sum(old and not new for old, new in zip(left_correct, right_correct))
    return {
        "improved_cases": improved,
        "worsened_cases": worsened,
        "net_correct_change": improved - worsened,
        "both_correct": sum(old and new for old, new in zip(left_correct, right_correct)),
        "both_not_correct": sum(not old and not new for old, new in zip(left_correct, right_correct)),
        "mcnemar_exact_p": exact_mcnemar_p(improved, worsened),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/organized_rca_v2_stratified_60_40_seed42"),
    )
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("none", "vllm"), default="none")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = load_cases(args.data_dir)
    train_cases, test_cases = cases[: args.train_size], cases[args.train_size :]
    train_packs = build_packs(train_cases, source_dataset=str(args.data_dir))
    test_packs = build_packs(test_cases, source_dataset=str(args.data_dir))
    thresholds = fit_thresholds(train_cases)
    feature_model = fit_feature_model(train_packs)
    train_features = [extract_features(pack, thresholds, feature_model) for pack in train_packs]
    test_features = [extract_features(pack, thresholds, feature_model) for pack in test_packs]

    graph = EvidenceGraph.build(
        train_features,
        labels_of(train_cases),
        feature_model=feature_model,
        source_dataset=str(args.data_dir),
    )
    route_results = match_many(graph, test_features, top_k=0)
    route_decisions = route_many(route_results, COVERAGE_POLICY)

    evidence_requests = [
        build_request(pack, features, include_rules=False)
        for pack, features in zip(test_packs, test_features)
    ]
    rule_requests = [
        build_request(pack, features, include_rules=True)
        for pack, features in zip(test_packs, test_features)
    ]
    evidence_prompts = [
        build_empirical_prompt(request, include_rules=False)
        for request in evidence_requests
    ]
    rule_prompts = [
        build_empirical_prompt(request, include_rules=True)
        for request in rule_requests
    ]

    backend = backend_for(
        args.backend,
        **({
            "model_path": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "seed": args.seed,
        } if args.backend == "vllm" else {}),
    )

    print(f"cases: train={len(train_cases)} test={len(test_cases)}")
    print(f"backend: {args.backend}")
    print("arm 1/3: evidence_only")
    evidence_responses, evidence_reports, evidence_outputs = raw_arm(
        backend, evidence_prompts, test_packs, rule_requests
    )
    print("arm 2/3: rules_prompt")
    rule_responses, rule_reports, rule_outputs = raw_arm(
        backend, rule_prompts, test_packs, rule_requests
    )
    print("arm 3/3: rules_prompt_checker (reuse arm 2 first pass)")
    checked_responses, checked_reports, checked_attempts = checked_arm(
        backend,
        rule_requests,
        test_packs,
        rule_prompts,
        rule_outputs,
        max_attempts=args.max_attempts,
    )

    labels = labels_of(test_cases)
    responses_by_arm = {
        "evidence_only": evidence_responses,
        "rules_prompt": rule_responses,
        "rules_prompt_checker": checked_responses,
    }
    reports_by_arm = {
        "evidence_only": evidence_reports,
        "rules_prompt": rule_reports,
        "rules_prompt_checker": checked_reports,
    }
    all_indices = list(range(len(test_cases)))
    route_indices = {
        branch: [
            index for index, decision in enumerate(route_decisions)
            if decision.branch == branch
        ]
        for branch in ("N5a", "N5b", "N5c", "N6")
    }
    summary = {
        "study_question": "在相同证据、模型和解码参数下，当前物理规则是否提高大模型 RCA 判断质量？",
        "arms": {
            arm: {
                "overall": arm_metrics(
                    responses_by_arm[arm], reports_by_arm[arm], labels, all_indices
                ),
                "by_route": {
                    branch: arm_metrics(
                        responses_by_arm[arm],
                        reports_by_arm[arm],
                        labels,
                        indices,
                    )
                    for branch, indices in route_indices.items()
                },
            }
            for arm in ARMS
        },
        "paired_comparisons": {
            "evidence_only_to_rules_prompt": paired_comparison(
                evidence_responses, rule_responses, labels
            ),
            "rules_prompt_to_checker": paired_comparison(
                rule_responses, checked_responses, labels
            ),
            "evidence_only_to_full_rules": paired_comparison(
                evidence_responses, checked_responses, labels
            ),
        },
        "interpretation_guardrails": [
            "规则库当前仍为 pending_expert_review；本实验评估经验效果，不证明规则物理上已获专家确认。",
            "fiber 测试样本只有 6 条，必须同时报告绝对数量和 Wilson 区间。",
            "temperature=0 的单次确定性运行用于配对比较，不代表跨模型或跨数据集泛化。",
            "历史 case、历史标签和 M9 最终门禁未进入 prompt。",
        ],
    }

    outcomes = []
    traces = []
    for index, (pack, decision, truth) in enumerate(zip(test_packs, route_decisions, labels)):
        arm_outcomes: Dict[str, Any] = {}
        for arm in ARMS:
            response = responses_by_arm[arm][index]
            report = reports_by_arm[arm][index]
            arm_outcomes[arm] = {
                "verdict": response.verdict if response is not None else None,
                "confidence": response.confidence if response is not None else 0.0,
                "parsed": response is not None,
                "answered": response is not None and response.verdict is not None,
                "correct": response is not None and response.verdict == truth,
                "rule_check": report.to_dict(),
            }
        outcomes.append({
            "case_id": pack.case_id,
            "actual": truth,
            "route_stratum": decision.branch,
            "telemetry_status": pack.telemetry_status,
            "evidence_count": len(test_features[index].tokens),
            "arms": arm_outcomes,
        })
        traces.append({
            "case_id": pack.case_id,
            "route_stratum": decision.branch,
            "evidence_only": {
                "prompt": evidence_prompts[index],
                "raw_output": evidence_outputs[index],
                "parsed": evidence_responses[index].to_dict()
                if evidence_responses[index] is not None else None,
                "offline_rule_check": evidence_reports[index].to_dict(),
            },
            "rules_prompt": {
                "prompt": rule_prompts[index],
                "raw_output": rule_outputs[index],
                "parsed": rule_responses[index].to_dict()
                if rule_responses[index] is not None else None,
                "offline_rule_check": rule_reports[index].to_dict(),
            },
            "rules_prompt_checker": {
                "attempts": checked_attempts[index],
                "accepted": checked_responses[index].to_dict()
                if checked_responses[index] is not None else None,
            },
        })

    manifest = {
        "schema_version": "rule-empirical-study-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "data": {
            "data_dir": str(args.data_dir),
            "train_size": len(train_cases),
            "test_size": len(test_cases),
            "train_case_ids": [pack.case_id for pack in train_packs],
            "test_case_ids": [pack.case_id for pack in test_packs],
        },
        "versions": {
            "evidence_graph": graph.version,
            "feature_dictionary": graph.dictionary_version,
            "feature_dictionary_hash": graph.dictionary_hash,
            "constraint_library": CONSTRAINT_LIBRARY.version,
            "constraint_library_hash": CONSTRAINT_LIBRARY.content_hash(),
            "sop": "not_used_in_isolated_rule_study",
            "empirical_prompt": EMPIRICAL_PROMPT_VERSION,
            "evidence_only_prompt_hash": empirical_prompt_hash(include_rules=False),
            "rules_prompt_hash": empirical_prompt_hash(include_rules=True),
        },
        "model": {
            "backend": args.backend,
            "model_path": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": 0.0,
            "seed": args.seed,
            "max_attempts": args.max_attempts,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
        },
        "experimental_control": {
            "paired_cases": True,
            "same_evidence_tokens": True,
            "same_output_schema": True,
            "history_in_prompt": False,
            "labels_read_after_generation": True,
            "checker_reuses_rules_prompt_first_pass": True,
            "retrieval_top_k_for_stratification_only": 0,
            "routing_policy_for_stratification_only": COVERAGE_POLICY.to_dict(),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "run_manifest.json": manifest,
        "summary.json": summary,
        "outcomes.json": outcomes,
        "traces.json": traces,
    }
    for filename, value in artifacts.items():
        path = args.output_dir / filename
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")

    for arm in ARMS:
        row = summary["arms"][arm]["overall"]
        precision = row["precision_when_answered"]
        precision_text = f"{precision:.2%}" if precision is not None else "-"
        print(
            f"{arm}: answered={row['answered']}/{row['case_count']} "
            f"correct={row['correct']} accuracy={row['accuracy_all_cases']:.2%} "
            f"precision_when_answered={precision_text}"
        )


if __name__ == "__main__":
    main()
