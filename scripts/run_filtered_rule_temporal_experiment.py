#!/usr/bin/env python3
"""Run the final filtered-rule experiment on one train split and two test splits.

The training knowledge is fitted once, persisted, reloaded, and then evaluated
independently on ``all_data`` and ``rule1_channel_not_4``.  This entrypoint is
GPU-only and never performs a CPU model dry run.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.constraints.measurement import MEASUREMENT_CONTRACT_LIBRARY  # noqa: E402
from rca_framework.constraints.physics import PHYSICS_LIBRARY  # noqa: E402
from rca_framework.data import (  # noqa: E402
    cases_by_manifest_split,
    load_split_manifest,
    split_manifest_hash,
)
from rca_framework.decision import DecisionPolicy  # noqa: E402
from rca_framework.evidence_graph import (  # noqa: E402
    FILTERED_RULE_THREE_CHANNEL_POLICY,
    MATCH_ALGORITHM_VERSION,
    match_many,
)
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.html_report import render_experiment_html  # noqa: E402
from rca_framework.knowledge import KNOWLEDGE_BUNDLE_SCHEMA, OfflineKnowledgeBundle, fit_offline_knowledge  # noqa: E402
from rca_framework.llm import (  # noqa: E402
    FILTERED_RULE_PROMPT_TEMPLATE_VERSION,
    ConstrainedReasoner,
    backend_for,
    prompt_template_hash,
)
from rca_framework.sop import EXPERT_SOP_VERSION, expert_sop_hash  # noqa: E402
from rca_framework.topology import TOPOLOGY_CONTRACT_VERSION  # noqa: E402
from scripts.evaluate_routing import personal_alignment_gate, run_policy, show  # noqa: E402
from scripts.run_offline_sop_llm_experiment import (  # noqa: E402
    _git_revision,
    _gpu_memory_released,
    _quality_summary,
    _write_json,
    query_gpu_state,
)


TEST_SPLITS = {
    "test_all_data": "test/all_data",
    "test_rule1_channel_not_4": "test/rule1_channel_not_4",
}
EXPECTED_TEST_SIZES = {"test_all_data": 417, "test_rule1_channel_not_4": 67}
FORMAL_MAX_NEW_TOKENS = 16384
FORMAL_MAX_MODEL_LEN = 32768
FORMAL_MAX_ATTEMPTS = 3
FORMAL_GUIDED_JSON = True
DETERMINISTIC_KNOWLEDGE_SUMMARY = Path(
    "artifacts/filtered_rule_deterministic_knowledge_v1/audit_summary.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_visible_gpu_state() -> Dict[str, Any]:
    state = query_gpu_state()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or not state.get("gpus"):
        return state
    allowed = {int(item) for item in visible.split(",") if item.strip().isdigit()}
    state["gpus"] = [row for row in state["gpus"] if row.get("index") in allowed]
    state["available"] = bool(state["gpus"])
    state["cuda_visible_devices"] = visible
    return state


def _validate_dataset(
    train_cases: Sequence[Dict[str, Any]],
    test_sets: Mapping[str, Sequence[Dict[str, Any]]],
    expected_train_size: int,
) -> None:
    if len(train_cases) != expected_train_size:
        raise ValueError(f"expected {expected_train_size} train cases, got {len(train_cases)}")
    train_ids = {str(case.get("case_id", "")) for case in train_cases}
    if len(train_ids) != len(train_cases):
        raise ValueError("duplicate case ids in train split")
    seen_test: set[str] = set()
    for split, cases in test_sets.items():
        expected = EXPECTED_TEST_SIZES[split]
        if len(cases) != expected:
            raise ValueError(f"expected {expected} cases in {split}, got {len(cases)}")
        ids = {str(case.get("case_id", "")) for case in cases}
        if len(ids) != len(cases):
            raise ValueError(f"duplicate case ids in {split}")
        if train_ids & ids:
            raise ValueError(f"train overlaps {split}")
        if seen_test & ids:
            raise ValueError(f"test splits overlap at {split}")
        seen_test.update(ids)
    for case in (*train_cases, *(case for cases in test_sets.values() for case in cases)):
        if case.get("label") not in ("L1", "L2", "fiber"):
            raise ValueError(f"unsupported label in case {case.get('case_id')}")


def _topology_summary(results: Sequence[Any]) -> Dict[str, Any]:
    top_sources = Counter()
    compatible_top = 0
    fallback = 0
    for result in results:
        fallback += int(bool(result.uses_cross_topology_fallback))
        if result.candidates:
            top = result.candidates[0]
            top_sources[top.source_dataset] += 1
            compatible_top += int(bool(top.topology_compatible))
    return {
        "case_count": len(results),
        "cross_topology_fallback_cases": fallback,
        "compatible_top_candidate_cases": compatible_top,
        "top_candidate_source_distribution": dict(sorted(top_sources.items())),
    }


def _assert_retry_contract_traces(
    traces: Mapping[str, Any],
    *,
    expected_case_count: int,
    scope: str,
    max_attempts: int = FORMAL_MAX_ATTEMPTS,
) -> None:
    if len(traces) != expected_case_count:
        raise RuntimeError(
            f"{scope}: expected one trace per case ({expected_case_count}), got {len(traces)}"
        )
    invalid = []
    for case_id, trace in traces.items():
        attempt_count = (
            int(trace.get("attempt_count", 0))
            if isinstance(trace, Mapping)
            else int(getattr(trace, "attempt_count", 0))
        )
        if not 1 <= attempt_count <= max_attempts:
            invalid.append((case_id, attempt_count))
    if invalid:
        raise RuntimeError(
            f"{scope}: retry contract violated (allowed 1..{max_attempts}): {invalid[:10]}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/filtered_rule_temporal_2025_06_09_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-profile", default="filtered_rule_v1", choices=("filtered_rule_v1",))
    parser.add_argument(
        "--policy",
        default=FILTERED_RULE_THREE_CHANNEL_POLICY.name,
        choices=(FILTERED_RULE_THREE_CHANNEL_POLICY.name,),
    )
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-train-size", type=int, default=124)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=FORMAL_MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=FORMAL_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--max-attempts",
        type=int,
        choices=(FORMAL_MAX_ATTEMPTS,),
        default=FORMAL_MAX_ATTEMPTS,
        help="活动正式流程最多三轮；只对解析或物理校验失败的 case 重试",
    )
    parser.add_argument("--decision-lower-bound", type=float, default=0.5)
    parser.add_argument("--decision-min-support", type=int, default=10)
    parser.add_argument("--target-selective-risk", type=float, default=0.15)
    parser.add_argument("--class-conditional-bounds", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    gpu_before = _query_visible_gpu_state()
    if not gpu_before["available"]:
        raise RuntimeError(f"formal experiment requires GPU: {gpu_before['error']}")
    if args.tensor_parallel_size > len(gpu_before["gpus"]):
        raise RuntimeError(
            f"tensor_parallel_size={args.tensor_parallel_size} exceeds visible GPUs={len(gpu_before['gpus'])}"
        )
    lifecycle_path = args.output_dir / "resource_lifecycle.json"
    lifecycle: Dict[str, Any] = {
        "schema_version": "gpu-resource-lifecycle-v1",
        "status": "running",
        "started_at_utc": _utc_now(),
        "gpu_before": gpu_before,
        "backend_close_called": False,
    }
    _write_json(lifecycle_path, lifecycle)

    backend = None
    reasoner = None
    run_manifest: Dict[str, Any] = {}
    manifest_path = args.output_dir / "run_manifest.json"

    try:
        split_manifest = load_split_manifest(args.data_dir)
        manifest_hash = split_manifest_hash(args.data_dir)
        train_cases = cases_by_manifest_split(args.data_dir, "train")
        test_sets = {
            name: cases_by_manifest_split(args.data_dir, manifest_split)
            for name, manifest_split in TEST_SPLITS.items()
        }
        _validate_dataset(train_cases, test_sets, args.expected_train_size)
        source_id = args.data_dir.name
        policy = FILTERED_RULE_THREE_CHANNEL_POLICY
        base_decision_policy = DecisionPolicy(
            final_lower_bound=args.decision_lower_bound,
            minimum_support=args.decision_min_support,
            candidate_order=("branch",),
        )
        bundle, training_artifacts = fit_offline_knowledge(
            train_cases,
            source_dataset=source_id,
            split_manifest_hash=manifest_hash,
            feature_profile=args.feature_profile,
            policies=(policy,),
            # Training is deterministic knowledge deposition. The LLM is only
            # invoked for unseen test/online cases after the bundle is reloaded.
            reasoner=None,
            top_k=args.top_k,
            target_selective_risk=args.target_selective_risk,
            decision_minimum_support=args.decision_min_support,
            decision_candidate_order=("branch",),
            decision_class_conditional=args.class_conditional_bounds,
            build_metadata={
                "knowledge_build_mode": "deterministic-train-only-v1",
                "llm_calls": 0,
                "label_leakage": False,
                "n8_frozen": True,
            },
        )
        if any(training_artifacts.traces.values()):
            raise RuntimeError("formal train build must not produce LLM traces")
        knowledge_dir = args.output_dir / "knowledge"
        bundle_path = bundle.save(knowledge_dir / "knowledge_bundle.json")
        _write_json(knowledge_dir / "training_summary.json", training_artifacts.summary)
        _write_json(knowledge_dir / "training_traces.json", training_artifacts.traces)
        bundle = OfflineKnowledgeBundle.load(bundle_path)
        reference_summary_path = repo / DETERMINISTIC_KNOWLEDGE_SUMMARY
        if reference_summary_path.exists():
            reference = json.loads(reference_summary_path.read_text(encoding="utf-8"))
            observed = {
                "knowledge_bundle_hash": bundle.content_hash(),
                "evidence_graph_version": bundle.graph.version,
                "learned_sop_hash": bundle.sop.content_hash(),
            }
            expected = {key: reference.get(key) for key in observed}
            if observed != expected:
                raise RuntimeError(
                    f"deterministic knowledge does not match committed reference: "
                    f"observed={observed}, expected={expected}"
                )
        decision_policy = bundle.decision_policies.get(policy.name, base_decision_policy)

        # Allocate the model only after deterministic train knowledge has been
        # persisted and reloaded. Training performs zero generation requests.
        backend = backend_for(
            "vllm",
            model_path=str(model_path),
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            enforce_eager=args.enforce_eager,
            guided_json=FORMAL_GUIDED_JSON,
            seed=args.seed,
        )
        reasoner = ConstrainedReasoner(backend=backend, max_attempts=args.max_attempts)

        reports: Dict[str, Any] = {}
        dataset_summaries: Dict[str, Any] = {}
        for split, test_cases in test_sets.items():
            test_packs, test_features = bundle.extract_test_features(test_cases, source_dataset=source_id)
            test_results = match_many(bundle.graph, test_features, top_k=args.top_k)
            report, records, traces = run_policy(
                policy,
                bundle.graph,
                (), (), (),
                test_results,
                test_packs,
                [str(case["label"]) for case in test_cases],
                reasoner=reasoner,
                decision_policy=decision_policy,
                calibrate_llm=False,
                test_features=test_features,
                sop_model=bundle.sop,
                branch_calibration=bundle.branch_calibrations[policy.name],
                llm_calibration_override=bundle.llm_calibrations.get(policy.name),
                expert_calibration=bundle.expert_calibration,
            )
            _assert_retry_contract_traces(
                traces,
                expected_case_count=len(test_cases),
                scope=split,
                max_attempts=args.max_attempts,
            )
            split_dir = args.output_dir / split
            outcomes = {policy.name: records}
            trace_payload = {policy.name: {case_id: trace.to_dict() for case_id, trace in sorted(traces.items())}}
            split_manifest_payload = {
                "split": split,
                "source_dataset": split.removeprefix("test_"),
                "case_count": len(test_cases),
                "topology_retrieval": _topology_summary(test_results),
                "test_case_ids": [pack.case_id for pack in test_packs],
            }
            split_summary = {"schema_version": "filtered-rule-test-summary-v1", "policies": {policy.name: report}}
            _write_json(split_dir / "summary.json", split_summary)
            _write_json(split_dir / "run_manifest.json", split_manifest_payload)
            _write_json(split_dir / "outcomes.json", outcomes)
            _write_json(split_dir / "predictions.json", records)
            _write_json(split_dir / "traces.json", trace_payload)
            bad_cases = [
                record for record in records
                if record.get("final_decision", {}).get("verdict") != record.get("actual")
            ]
            _write_json(split_dir / "bad_cases.json", bad_cases)
            _write_json(split_dir / "label_suspects.json", {
                "schema_version": "label-suspects-v1",
                "cases": [],
                "status": "awaiting post-run human review",
            })
            _write_json(split_dir / "irreducible_cases.json", {
                "schema_version": "irreducible-cases-v1",
                "cases": [],
                "status": "awaiting post-run human review",
            })
            html_manifest = render_experiment_html(
                split_dir / "html",
                summary=split_summary,
                manifest=split_manifest_payload,
                outcomes=outcomes,
                traces=trace_payload,
                training_summary=training_artifacts.summary,
            )
            _write_json(split_dir / "html" / "report_manifest.json", html_manifest)
            reports[split] = report
            dataset_summaries[split] = {
                "quality": _quality_summary(test_cases, test_packs, test_features),
                "topology_retrieval": _topology_summary(test_results),
            }
            show(report)

        run_manifest = {
            "schema_version": "filtered-rule-dual-test-run-manifest-v1",
            "created_at_utc": _utc_now(),
            "python_version": platform.python_version(),
            "git_revision": _git_revision(repo),
            "scope": {
                "self_evolution": False,
                "feedback_update": False,
                "flow": "one temporal train -> persisted knowledge reload -> two independent source tests",
                "training": "deterministic knowledge deposition; zero LLM calls",
                "generation": "test route once -> LLM generation -> failed-case-only retry -> N6 confidence gate",
            },
            "data": {
                "data_dir": str(args.data_dir),
                "manifest_schema": split_manifest.get("schema_version"),
                "manifest_hash": manifest_hash,
                "train_months": split_manifest.get("train_months"),
                "train_size": len(train_cases),
                "test_sizes": {name: len(cases) for name, cases in test_sets.items()},
                "topology_contract": TOPOLOGY_CONTRACT_VERSION,
            },
            "knowledge": {
                "schema_version": KNOWLEDGE_BUNDLE_SCHEMA,
                "path": str(bundle_path),
                "content_hash": bundle.content_hash(),
                "evidence_graph_version": bundle.graph.version,
                "feature_profile": bundle.feature_profile,
                "feature_dictionary_version": bundle.graph.dictionary_version,
                "feature_dictionary_hash": bundle.graph.dictionary_hash,
                "learned_sop_version": bundle.sop.version,
                "learned_sop_hash": bundle.sop.content_hash(),
                "build_mode": "deterministic-train-only-v1",
                "training_llm_calls": 0,
                "training_llm_traces": 0,
            },
            "versions": {
                "topology_contract": TOPOLOGY_CONTRACT_VERSION,
                "physics_library": PHYSICS_LIBRARY.version,
                "physics_library_hash": PHYSICS_LIBRARY.content_hash(),
                "measurement_contract_library": MEASUREMENT_CONTRACT_LIBRARY.version,
                "measurement_contract_library_hash": MEASUREMENT_CONTRACT_LIBRARY.content_hash(),
                "expert_sop": EXPERT_SOP_VERSION,
                "expert_sop_hash": expert_sop_hash(),
                "prompt_template": FILTERED_RULE_PROMPT_TEMPLATE_VERSION,
                "prompt_template_hash": prompt_template_hash("filtered_rule_v1"),
                "decision_policy": decision_policy.version,
            },
            "data_quality": {
                "train": _quality_summary(train_cases, build_packs(train_cases, source_dataset=source_id), bundle.training_features),
                **dataset_summaries,
            },
            "retrieval": {
                "top_k": args.top_k,
                "policy": policy.to_dict(),
                "match_algorithm_version": MATCH_ALGORITHM_VERSION,
                "same_topology_preferred": True,
            },
            "decision": decision_policy.to_dict(),
            "personal_alignment_gate": personal_alignment_gate(decision_policy),
            "llm": {
                "backend": backend.name if backend is not None else "not-initialized",
                "model_path": str(model_path),
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_attempts": args.max_attempts,
                "single_pass": False,
                "retry_failed_cases_only": True,
                "structured_output": "json_schema",
                "guided_json": FORMAL_GUIDED_JSON,
                "seed": args.seed,
            },
            "label_leakage": False,
            "leakage_policy": "test labels are used only after inference for metrics and reports",
            "gpu": {"before": gpu_before, "after": None, "memory_released": None},
        }
        _write_json(args.output_dir / "summary.json", {"schema_version": "filtered-rule-dual-test-summary-v1", "tests": reports})
        _write_json(manifest_path, run_manifest)
        lifecycle["status"] = "completed"
    except Exception as exc:
        lifecycle["status"] = "failed"
        lifecycle["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
                lifecycle["backend_close_called"] = True
        finally:
            reasoner = None
            gc.collect()
            gpu_after = _query_visible_gpu_state()
            released = _gpu_memory_released(gpu_before, gpu_after)
            lifecycle.update({"finished_at_utc": _utc_now(), "gpu_after": gpu_after, "gpu_memory_released": released})
            _write_json(lifecycle_path, lifecycle)
            if run_manifest:
                run_manifest["gpu"].update({"after": gpu_after, "memory_released": released, "backend_close_called": lifecycle["backend_close_called"]})
                _write_json(manifest_path, run_manifest)


if __name__ == "__main__":
    main()
