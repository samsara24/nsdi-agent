"""T5 端到端评估：跑通 N1 -> N2 -> N3 -> N4 -> N5a/b/c -> N6，并对比两套路由规则。

这个脚本回答三个问题：

1. 两套路由规则各自把 case 分到哪里，各分支实际判对多少。
2. 置信度标定是否可信：按标定分组算出来的置信度，在留出测试集上兑现了没有。
3. 经过 M9 可靠性门禁与降级之后，剩下那部分的准确率是多少——
   这是「不硬猜」的代价与收益，也是 T7 降级策略要用的曲线。

标定一律只用训练集留一法；留出测试集只用于核对。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.branches import fit_calibration, handle_many  # noqa: E402
from rca_framework.branches.base import majority_label  # noqa: E402
from rca_framework.constraints.library import CONSTRAINT_LIBRARY  # noqa: E402
from rca_framework.data import cases_by_manifest_split, load_cases, load_split_manifest  # noqa: E402
from rca_framework.decision import (  # noqa: E402
    DEFAULT_DECISION_POLICY,
    DecisionPolicy,
    LLMCalibration,
    decide_many,
)
from rca_framework.evidence_graph import (  # noqa: E402
    BOARD_POLICY,
    COVERAGE_POLICY,
    EvidenceGraph,
    match_many,
    routing_summary,
)
from rca_framework.evidence_pack import build_packs, labels_of  # noqa: E402
from rca_framework.feedback import build_case_diagnosis  # noqa: E402
from rca_framework.report import build_report  # noqa: E402
from rca_framework.sop import LEARNED_SOP_VERSION, learn_sop  # noqa: E402
from rca_framework.features.dictionary import dictionary_for  # noqa: E402
from rca_framework.features.extractor import extract_features, fit_feature_model  # noqa: E402
from rca_framework.llm import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    SOP_VERSION,
    prompt_template_hash,
)
from rca_framework.types import ROOT_CAUSES  # noqa: E402


def branch_report(outcomes, decisions, actual: Sequence[str]) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    for outcome, decision, truth in zip(outcomes, decisions, actual):
        row = rows.setdefault(
            decision.branch,
            {"n": 0, "answered": 0, "correct": 0, "abstained": 0, "needs_llm": 0, "needs_human": 0},
        )
        row["n"] += 1
        row["needs_llm"] += int(outcome.needs_llm)
        row["needs_human"] += int(outcome.needs_human)
        if outcome.verdict is None:
            row["abstained"] += 1
        else:
            row["answered"] += 1
            row["correct"] += int(outcome.verdict == truth)
    for row in rows.values():
        row["precision_when_answered"] = (
            round(row["correct"] / row["answered"], 6) if row["answered"] else None
        )
    return rows


def calibration_check(outcomes, actual: Sequence[str]) -> Dict[str, Any]:
    """标定兑现检查：声称的置信度 vs 留出集上的实际正确率。"""
    tally: Dict[str, Dict[str, Any]] = {}
    for outcome, truth in zip(outcomes, actual):
        row = tally.setdefault(
            outcome.calibration_group,
            {"claimed_confidence": outcome.confidence, "claimed_lower_bound": outcome.confidence_lower_bound,
             "answered": 0, "correct": 0},
        )
        if outcome.verdict is not None:
            row["answered"] += 1
            row["correct"] += int(outcome.verdict == truth)
    for row in tally.values():
        row["observed_accuracy"] = (
            round(row["correct"] / row["answered"], 6) if row["answered"] else None
        )
        if row["observed_accuracy"] is not None:
            row["gap"] = round(row["observed_accuracy"] - row["claimed_confidence"], 6)
    return dict(sorted(tally.items()))


def class_metrics(predictions: Sequence[Optional[str]], actual: Sequence[str]) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for label in ROOT_CAUSES:
        tp = sum(prediction == label and truth == label for prediction, truth in zip(predictions, actual))
        fp = sum(prediction == label and truth != label for prediction, truth in zip(predictions, actual))
        fn = sum(prediction != label and truth == label for prediction, truth in zip(predictions, actual))
        support = sum(truth == label for truth in actual)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[label] = {
            "support": support,
            "predicted": tp + fp,
            "true_positive": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return rows


def n5a_report(results, decisions, actual: Sequence[str]) -> Dict[str, Any]:
    selected = [
        (result, truth)
        for result, decision, truth in zip(results, decisions, actual)
        if decision.branch == "N5a"
    ]
    pure = sum(result.is_label_pure for result, _ in selected)
    majority_correct = sum(
        majority_label([item.label for item in result.top_candidates if item.label is not None]) == truth
        for result, truth in selected
    )
    return {
        "case_count": len(selected),
        "pure_signature_cases": pure,
        "mixed_signature_cases": len(selected) - pure,
        "mixed_signature_ratio": round((len(selected) - pure) / len(selected), 6) if selected else 0.0,
        "bucket_majority_correct": majority_correct,
        "bucket_majority_accuracy": round(majority_correct / len(selected), 6) if selected else None,
    }


def match_record(result) -> Dict[str, Any]:
    """逐 case 只保存参与分支判断的最高分候选，避免 top_k=0 时重复整个训练图。"""
    value = result.to_dict()
    value["retrieved_candidate_count"] = len(result.candidates)
    value["candidates"] = [item.to_dict() for item in result.top_candidates]
    return value


def selective_risk_curve(
    outcomes,
    actual: Sequence[str],
    *,
    minimum_support: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        selected = [
            (outcome, truth)
            for outcome, truth in zip(outcomes, actual)
            if outcome.verdict is not None
            and outcome.calibration_support >= minimum_support
            and outcome.confidence_lower_bound >= threshold
        ]
        correct = sum(outcome.verdict == truth for outcome, truth in selected)
        precision = correct / len(selected) if selected else None
        rows.append({
            "minimum_lower_bound": threshold,
            "answered": len(selected),
            "coverage": round(len(selected) / len(actual), 6) if actual else 0.0,
            "correct": correct,
            "precision_at_coverage": round(precision, 6) if precision is not None else None,
            "selective_risk": round(1.0 - precision, 6) if precision is not None else None,
        })
    return rows


def decision_report(final_decisions, actual: Sequence[str]) -> Dict[str, Any]:
    counts = Counter(item.action for item in final_decisions)
    predictions = [item.verdict for item in final_decisions]
    answered = [(prediction, truth) for prediction, truth in zip(predictions, actual) if prediction is not None]
    correct = sum(prediction == truth for prediction, truth in answered)
    total = len(final_decisions)
    return {
        "actions": {action: counts.get(action, 0) for action in ("final", "request_evidence", "human_review")},
        "answered": len(answered),
        "coverage": round(len(answered) / total, 6) if total else 0.0,
        "correct": correct,
        "precision_when_answered": round(correct / len(answered), 6) if answered else None,
        "coverage_accuracy": round(correct / total, 6) if total else 0.0,
        "low_confidence_degradation_rate": round(
            (counts.get("request_evidence", 0) + counts.get("human_review", 0)) / total, 6
        ) if total else 0.0,
        "human_intervention_rate": round(counts.get("human_review", 0) / total, 6) if total else 0.0,
        "class_metrics": class_metrics(predictions, actual),
    }


def run_policy(policy, graph, train_results, train_packs, train_labels,
               test_results, test_packs, test_labels, reasoner=None,
               decision_policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
               calibrate_llm: bool = True,
               train_features=None,
               test_features=None,
               sop_model=None,
               branch_calibration: Optional[Any] = None,
               llm_calibration_override: Optional[LLMCalibration] = None,
               ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    calibration = branch_calibration or fit_calibration(
        train_results, train_packs, train_labels, policy=policy
    )
    llm_calibration = llm_calibration_override
    if reasoner is not None and calibrate_llm and llm_calibration is None:
        calibration_traces: Dict[str, Any] = {}
        calibration_paired = handle_many(
            train_results,
            train_packs,
            calibration,
            policy=policy,
            reasoner=reasoner,
            trace_collector=calibration_traces,
            features=train_features,
            sop_model=sop_model,
        )
        calibration_outcomes = [item[1] for item in calibration_paired]
        llm_calibration = LLMCalibration.fit(
            calibration_outcomes,
            [calibration_traces.get(outcome.case_id) for outcome in calibration_outcomes],
            train_labels,
            source=f"train-loo-llm:{policy.name}",
        )

    traces: Dict[str, Any] = {}
    paired = handle_many(
        test_results,
        test_packs,
        calibration,
        policy=policy,
        reasoner=reasoner,
        llm_calibration=llm_calibration,
        trace_collector=traces,
        features=test_features,
        sop_model=sop_model,
    )
    decisions = [item[0] for item in paired]
    outcomes = [item[1] for item in paired]
    final_decisions = decide_many(outcomes, decision_policy)

    answered = [(o, t) for o, t in zip(outcomes, test_labels) if o.verdict is not None]
    correct = sum(o.verdict == t for o, t in answered)

    report = {
        "policy": policy.to_dict(),
        "calibration": calibration.to_dict(),
        "llm_calibration": llm_calibration.to_dict() if llm_calibration is not None else None,
        "decision_policy": decision_policy.to_dict(),
        "routing": routing_summary(decisions),
        "n5a": n5a_report(test_results, decisions, test_labels),
        "branches": branch_report(outcomes, decisions, test_labels),
        "calibration_check": calibration_check(outcomes, test_labels),
        "answered": len(answered),
        "answer_rate": round(len(answered) / len(outcomes), 6) if outcomes else 0.0,
        "correct": correct,
        "precision_when_answered": round(correct / len(answered), 6) if answered else None,
        "coverage_accuracy": round(correct / len(outcomes), 6) if outcomes else 0.0,
        "raw_class_metrics": class_metrics([outcome.verdict for outcome in outcomes], test_labels),
        "selective_risk_curve": selective_risk_curve(
            outcomes, test_labels, minimum_support=decision_policy.minimum_support
        ),
        "final_decisions": decision_report(final_decisions, test_labels),
    }
    records = []
    for index, (result, routing, outcome, final_decision, truth) in enumerate(
        zip(test_results, decisions, outcomes, final_decisions, test_labels)
    ):
        diagnosis = None
        if test_features is not None:
            diagnosis = build_case_diagnosis(
                test_packs[index],
                test_features[index],
                outcome,
                final_decision,
                sop_version=sop_model.version if sop_model is not None else SOP_VERSION,
                constraint_library_version=CONSTRAINT_LIBRARY.version,
            )
        report_record = build_report(outcome, final_decision, diagnosis=diagnosis).to_dict()
        feature_record = test_features[index].to_dict() if test_features is not None else None
        sop_prediction = (
            sop_model.predict(test_features[index]).to_dict()
            if sop_model is not None and test_features is not None
            else None
        )
        records.append({
            "case_id": outcome.case_id,
            "actual": truth,
            "evidence_pack": test_packs[index].to_dict(),
            "features": feature_record,
            "sop_prediction": sop_prediction,
            "match": match_record(result),
            "routing": routing.to_dict(),
            "branch_outcome": outcome.to_dict(),
            "final_decision": final_decision.to_dict(),
            "diagnosis_graph": diagnosis.to_dict() if diagnosis is not None else None,
            "report": report_record,
            "trace_id": outcome.case_id if outcome.case_id in traces else None,
        })
    return report, records, traces


def show(report: Dict[str, Any]) -> None:
    policy = report["policy"]
    print(f"===== 路由规则：{policy['name']} =====")
    print(f"  {policy['description']}\n")

    routing = report["routing"]
    counts = ", ".join(f"{k}={v}" for k, v in routing["counts"].items())
    print(f"  分流分布   : {counts}（共 {routing['case_count']} 条）")
    print(f"  有补采清单 : {routing['cases_with_missing_evidence']} 条\n")

    print(f"  {'分支':<6} {'n':>4} {'给结论':>6} {'判对':>5} {'给结论时准确率':>14} {'需LLM':>6} {'需人工':>6}")
    for branch in ("N5a", "N5b", "N5c", "N6"):
        row = report["branches"].get(branch)
        if not row:
            continue
        precision = row["precision_when_answered"]
        print(
            f"  {branch:<6} {row['n']:>4} {row['answered']:>6} {row['correct']:>5} "
            f"{(f'{precision:.2%}' if precision is not None else '-'):>14} "
            f"{row['needs_llm']:>6} {row['needs_human']:>6}"
        )
    print()

    print(f"  {'标定分组':<20} {'声称置信度':>10} {'95%下界':>9} {'留出实测':>9} {'差距':>8} {'n':>4}")
    for group, row in report["calibration_check"].items():
        observed = row["observed_accuracy"]
        gap = row.get("gap")
        print(
            f"  {group:<20} {row['claimed_confidence']:>10.2%} {row['claimed_lower_bound']:>9.2%} "
            f"{(f'{observed:.2%}' if observed is not None else '-'):>9} "
            f"{(f'{gap:+.2%}' if gap is not None else '-'):>8} {row['answered']:>4}"
        )
    print()
    print(f"  给出结论    : {report['answered']} 条（{report['answer_rate']:.2%}）")
    print(f"  给结论时准确率: {report['correct']}/{report['answered']} = {report['precision_when_answered']:.2%}")
    print(f"  全量准确率  : {report['coverage_accuracy']:.2%}（弃权计为不对，与 legacy 58/85=68.24% 同口径）")
    final = report["final_decisions"]
    final_precision = final["precision_when_answered"]
    print(
        f"  M9 最终出口 : {final['answered']} 条（覆盖率 {final['coverage']:.2%}，"
        f"给结论时准确率 {(f'{final_precision:.2%}' if final_precision is not None else '-')}）"
    )
    print(
        f"  降级分布    : 补采 {final['actions']['request_evidence']}，"
        f"人工 {final['actions']['human_review']}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/organized_rca_v2_stratified_60_40_seed42"))
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--manifest-split", action="store_true",
                        help="从 data-dir/_metadata/manifest.json 显式读取 train/test split")
    parser.add_argument("--feature-profile", default="v1", choices=("v1", "v2", "all_families"),
                        help="特征字典 profile；l2fixed v2 实验应使用 v2")
    parser.add_argument("--learned-sop", action="store_true",
                        help="在训练集上学习 learned-sop-v1，并接入 N5c dry-run")
    parser.add_argument("--output", type=Path, default=None,
                        help="兼容旧调用：只写聚合 JSON；正式实验请用 --output-dir")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="正式实验目录：写 summary/run_manifest/outcomes/traces 四类产物")
    parser.add_argument("--top-k", type=int, default=0, help="历史候选数；0 表示保留全部候选")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policies", nargs="+", default=(BOARD_POLICY.name, COVERAGE_POLICY.name),
                        choices=(BOARD_POLICY.name, COVERAGE_POLICY.name))
    parser.add_argument("--decision-lower-bound", type=float, default=0.5)
    parser.add_argument("--decision-min-support", type=int, default=10)
    parser.add_argument("--skip-llm-calibration", action="store_true",
                        help="仅用于工程冒烟；正式实验默认在训练集留一法输出上标定 LLM")
    parser.add_argument("--llm-backend", default="none", choices=("none", "vllm"),
                        help="三分支仲裁后端。none 表示不调模型，保留确定性分支输出。")
    parser.add_argument("--model-path", default="", help="--llm-backend vllm 时必填")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="约束校验失败后的最大重写次数，用尽仍不合规则弃权")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="推理型模型的思考段很长，给少了会被截断而解析失败")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--disable-custom-all-reduce", action="store_true",
                        help="无 NVLink 的多卡机器上建议开启，改走标准 NCCL")
    args = parser.parse_args()

    reasoner = None
    if args.llm_backend != "none":
        from rca_framework.llm import ConstrainedReasoner, backend_for

        reasoner = ConstrainedReasoner(
            backend=backend_for(
                args.llm_backend,
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                disable_custom_all_reduce=args.disable_custom_all_reduce,
                seed=args.seed,
            ),
            max_attempts=args.max_attempts,
        )

    split_manifest = None
    if args.manifest_split:
        split_manifest = load_split_manifest(args.data_dir)
        train_cases = cases_by_manifest_split(args.data_dir, "train")
        test_cases = cases_by_manifest_split(args.data_dir, "test")
    else:
        cases = load_cases(args.data_dir)
        train_cases, test_cases = cases[: args.train_size], cases[args.train_size :]
    dictionary = dictionary_for(args.feature_profile)
    thresholds = fit_thresholds(train_cases)
    train_packs = build_packs(train_cases, source_dataset=str(args.data_dir))
    test_packs = build_packs(test_cases, source_dataset=str(args.data_dir))
    model = fit_feature_model(train_packs, dictionary=dictionary)
    train_features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in train_packs]
    test_features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in test_packs]
    sop_model = learn_sop(
        train_features,
        labels_of(train_cases),
        source=f"{args.data_dir.name}:manifest-train" if args.manifest_split else f"{args.data_dir.name}:positional-train",
    ) if args.learned_sop else None
    graph = EvidenceGraph.build(
        train_features, labels_of(train_cases), feature_model=model, dictionary=dictionary, source_dataset=str(args.data_dir)
    )

    train_results = match_many(graph, train_features, top_k=args.top_k, leave_one_out=True)
    test_results = match_many(graph, test_features, top_k=args.top_k)

    print(f"graph_version : {graph.version}")
    print(f"train / test  : {len(train_cases)} / {len(test_cases)}")
    print(f"features      : {dictionary.version} ({dictionary.content_hash()})")
    if sop_model is not None:
        print(f"sop           : {sop_model.version} ({sop_model.content_hash()})")
    print(f"llm backend   : {args.llm_backend}"
          + ("（不执行 LLM 仲裁）" if args.llm_backend == "none" else f"，最多重写 {args.max_attempts} 次") + "\n")

    policy_by_name = {BOARD_POLICY.name: BOARD_POLICY, COVERAGE_POLICY.name: COVERAGE_POLICY}
    decision_policy = DecisionPolicy(
        final_lower_bound=args.decision_lower_bound,
        minimum_support=args.decision_min_support,
    )
    reports: Dict[str, Any] = {}
    all_records: Dict[str, Any] = {}
    all_traces: Dict[str, Any] = {}
    for policy_name in args.policies:
        policy = policy_by_name[policy_name]
        report, records, traces = run_policy(
            policy, graph, train_results, train_packs, labels_of(train_cases),
            test_results, test_packs, labels_of(test_cases), reasoner=reasoner,
            decision_policy=decision_policy,
            calibrate_llm=not args.skip_llm_calibration,
            train_features=train_features,
            test_features=test_features,
            sop_model=sop_model,
        )
        reports[policy.name] = report
        all_records[policy.name] = records
        all_traces[policy.name] = {
            case_id: trace.to_dict() for case_id, trace in sorted(traces.items())
        }
        show(report)

    manifest = {
        "schema_version": "agentic-rca-run-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "data": {
            "data_dir": str(args.data_dir),
            "train_size": len(train_cases),
            "test_size": len(test_cases),
            "train_case_ids": [pack.case_id for pack in train_packs],
            "test_case_ids": [pack.case_id for pack in test_packs],
            "split_seed": args.seed,
            "split_manifest_schema": split_manifest.get("schema_version") if split_manifest else None,
            "split_manifest_hash": split_manifest.get("source_hash") if split_manifest else None,
        },
        "versions": {
            "evidence_graph": graph.version,
            "feature_dictionary": graph.dictionary_version,
            "feature_dictionary_hash": graph.dictionary_hash,
            "constraint_library": CONSTRAINT_LIBRARY.version,
            "constraint_library_hash": CONSTRAINT_LIBRARY.content_hash(),
            "sop": sop_model.version if sop_model is not None else SOP_VERSION,
            "sop_hash": sop_model.content_hash() if sop_model is not None else None,
            "prompt_template": PROMPT_TEMPLATE_VERSION,
            "prompt_template_hash": prompt_template_hash(),
            "decision_policy": decision_policy.version,
        },
        "retrieval": {
            "top_k": args.top_k,
            "routing_policies": [policy_by_name[name].to_dict() for name in args.policies],
        },
        "decision": decision_policy.to_dict(),
        "llm": {
            "backend": args.llm_backend,
            "model_path": args.model_path,
            "max_attempts": args.max_attempts,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "temperature": 0.0,
            "seed": args.seed,
            "calibration": (
                "train-loo confidence-bin reliability"
                if reasoner is not None and not args.skip_llm_calibration
                else "disabled"
            ),
        },
        "training_graph_purity": graph.purity_report(),
        "learned_sop": sop_model.to_dict() if sop_model is not None else None,
        "leakage_policy": (
            "阈值、特征模型、证据图、分支置信度与 LLM 可靠性标定只使用训练集；"
            "留出测试标签仅在全部推理完成后用于指标计算。"
        ),
    }
    payload = {
        "graph_version": graph.version,
        "manifest": manifest,
        "policies": reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Mapping[str, Any] = {
            "summary.json": {"graph_version": graph.version, "policies": reports},
            "run_manifest.json": manifest,
            "outcomes.json": all_records,
            "traces.json": all_traces,
        }
        for filename, value in artifacts.items():
            path = args.output_dir / filename
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path}")


if __name__ == "__main__":
    main()


