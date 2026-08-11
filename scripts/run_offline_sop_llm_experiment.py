#!/usr/bin/env python3
"""Build train-only RCA knowledge, evaluate test cases, and render HTML reports.

This is the formal non-evolution experiment entrypoint. It deliberately uses
the manifest split, persists the complete training knowledge bundle, reloads
that immutable bundle, and only then starts test inference.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.constraints.library import CONSTRAINT_LIBRARY  # noqa: E402
from rca_framework.data import (  # noqa: E402
    cases_by_manifest_split,
    load_split_manifest,
)
from rca_framework.decision import (  # noqa: E402
    CANDIDATE_SOURCES,
    FIBER_EVIDENCE_REQUEST,
    DecisionPolicy,
)
from rca_framework.evidence_graph import (  # noqa: E402
    BOARD_POLICY,
    COVERAGE_POLICY,
    match_many,
)
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.html_report import render_experiment_html  # noqa: E402
from rca_framework.knowledge import (  # noqa: E402
    KNOWLEDGE_BUNDLE_SCHEMA,
    OfflineKnowledgeBundle,
    fit_offline_knowledge,
)
from rca_framework.llm import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    ConstrainedReasoner,
    backend_for,
    prompt_template_hash,
)
from rca_framework.types import ROOT_CAUSES  # noqa: E402
from scripts.evaluate_routing import run_policy, show  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def query_gpu_state() -> Dict[str, Any]:
    """Return an auditable GPU snapshot without importing CUDA frameworks."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found", "gpus": []}
    if result.returncode != 0:
        return {
            "available": False,
            "error": result.stderr.strip() or f"nvidia-smi exited {result.returncode}",
            "gpus": [],
        }
    gpus: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 6:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mb": int(parts[3]),
                "memory_used_mb": int(parts[4]),
                "memory_free_mb": int(parts[5]),
            }
        )
    return {
        "available": bool(gpus),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "error": "" if gpus else "nvidia-smi returned no GPU rows",
    }


def _gpu_memory_released(before: Mapping[str, Any], after: Mapping[str, Any]) -> Optional[bool]:
    before_rows = {row["uuid"]: row for row in before.get("gpus", [])}
    after_rows = {row["uuid"]: row for row in after.get("gpus", [])}
    common = sorted(set(before_rows) & set(after_rows))
    if not common:
        return None
    # Allow driver/context bookkeeping to retain at most 512 MiB.
    return all(
        after_rows[uuid]["memory_free_mb"] >= before_rows[uuid]["memory_free_mb"] - 512
        for uuid in common
    )


def _validate_split(
    train_cases: Sequence[Dict[str, Any]],
    test_cases: Sequence[Dict[str, Any]],
    *,
    expected_train_size: int,
    expected_test_size: int,
) -> None:
    if expected_train_size and len(train_cases) != expected_train_size:
        raise ValueError(
            f"train split size mismatch: expected {expected_train_size}, got {len(train_cases)}"
        )
    if expected_test_size and len(test_cases) != expected_test_size:
        raise ValueError(
            f"test split size mismatch: expected {expected_test_size}, got {len(test_cases)}"
        )
    train_ids = [str(case.get("case_id", "")) for case in train_cases]
    test_ids = [str(case.get("case_id", "")) for case in test_cases]
    duplicates = sorted(set(train_ids) & set(test_ids))
    if duplicates:
        raise ValueError(f"manifest train/test case ids overlap: {duplicates[:5]}")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("duplicate case ids within manifest split")
    if any(
        case.get("label") not in ("L1", "L2", "fiber")
        for case in (*train_cases, *test_cases)
    ):
        raise ValueError("manifest split contains missing or unsupported labels")


def _quality_summary(
    cases: Sequence[Dict[str, Any]],
    packs: Sequence[Any],
    features: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    def has_host_snr(case: Mapping[str, Any]) -> bool:
        block = case.get("host_snr")
        if not isinstance(block, Mapping):
            return False
        return any(
            isinstance(side, (Mapping, list, tuple)) and bool(side)
            for side in block.values()
        )

    def width_over_four(case: Mapping[str, Any], side: str) -> bool:
        for metric in ("txpower", "rxpower", "media_snr", "host_snr", "serdes_snr"):
            block = case.get(metric)
            values = block.get(side) if isinstance(block, Mapping) else None
            if isinstance(values, (Mapping, list, tuple)) and len(values) > 4:
                return True
        return False

    return {
        "case_count": len(cases),
        "label_distribution": dict(sorted(Counter(str(case.get("label")) for case in cases).items())),
        "missing_alarm_ip_interface": sum(not case.get("alarm_ip_interface") for case in cases),
        "missing_lane_number": sum(not case.get("Lane number") for case in cases),
        "host_snr_present_cases": sum(has_host_snr(case) for case in cases),
        "l1_metric_width_over_4_cases": sum(width_over_four(case, "L1") for case in cases),
        "l2_metric_width_over_4_cases": sum(width_over_four(case, "L2") for case in cases),
        "telemetry_status": dict(sorted(Counter(pack.telemetry_status for pack in packs).items())),
        "optical_blackout_cases": sum(bool(pack.optical_blackout) for pack in packs),
        "empty_feature_vector_cases": (
            sum(not item.tokens for item in features) if features is not None else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/rca_v2_l2fixed"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-profile", default="v2", choices=("v1", "v2", "all_families"))
    parser.add_argument(
        "--policy",
        default=COVERAGE_POLICY.name,
        choices=(BOARD_POLICY.name, COVERAGE_POLICY.name),
    )
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-train-size", type=int, default=161)
    parser.add_argument("--expected-test-size", type=int, default=107)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--decision-lower-bound", type=float, default=0.5)
    parser.add_argument("--decision-min-support", type=int, default=10)
    parser.add_argument(
        "--target-selective-risk",
        type=float,
        default=None,
        help=(
            "给定时忽略 --decision-lower-bound，改为在训练留一法上反解出"
            "满足该选择性风险的最大覆盖率工作点，并把工作点写进知识包与 manifest"
        ),
    )
    parser.add_argument(
        "--decision-candidate-order",
        nargs="+",
        default=("branch",),
        choices=CANDIDATE_SOURCES,
        help="M9 候选级联；加入 sop 表示分支不达标时允许退到 learned SOP 叶节点先验",
    )
    parser.add_argument(
        "--non-identifiable-labels",
        nargs="*",
        default=(),
        choices=ROOT_CAUSES,
        help="在现有遥测下不可识别的根因（C20）。命中候选转成带定向补采清单的 request_evidence",
    )
    parser.add_argument(
        "--class-conditional-bounds",
        action="store_true",
        help=(
            "在统一门限之上按预测类别逐类校准下界，要求每一类的选择性风险各自达标；"
            "单一门限会因类别先验差异结构性地挡掉少数类"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; use a new run directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpu_before = query_gpu_state()
    if not gpu_before["available"]:
        raise RuntimeError(f"formal SOP+LLM experiment requires GPU: {gpu_before['error']}")
    visible_gpu_count = len(gpu_before["gpus"])
    if args.tensor_parallel_size > visible_gpu_count:
        raise RuntimeError(
            f"tensor_parallel_size={args.tensor_parallel_size} exceeds detected GPUs={visible_gpu_count}"
        )
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"model path does not exist: {args.model_path}")

    lifecycle_path = args.output_dir / "resource_lifecycle.json"
    lifecycle: Dict[str, Any] = {
        "schema_version": "gpu-resource-lifecycle-v1",
        "status": "running",
        "started_at_utc": _utc_now(),
        "gpu_before": gpu_before,
        "backend_close_called": False,
        "gpu_memory_released": None,
    }
    _write_json(lifecycle_path, lifecycle)

    backend = backend_for(
        "vllm",
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        seed=args.seed,
    )
    reasoner = ConstrainedReasoner(backend=backend, max_attempts=args.max_attempts)
    manifest_path = args.output_dir / "run_manifest.json"
    run_manifest: Dict[str, Any] = {}

    try:
        split_manifest = load_split_manifest(args.data_dir)
        train_cases = cases_by_manifest_split(args.data_dir, "train")
        test_cases = cases_by_manifest_split(args.data_dir, "test")
        _validate_split(
            train_cases,
            test_cases,
            expected_train_size=args.expected_train_size,
            expected_test_size=args.expected_test_size,
        )
        policy = {
            BOARD_POLICY.name: BOARD_POLICY,
            COVERAGE_POLICY.name: COVERAGE_POLICY,
        }[args.policy]
        candidate_order = tuple(args.decision_candidate_order)
        non_identifiable = tuple(args.non_identifiable_labels)
        non_identifiable_evidence = {
            label: FIBER_EVIDENCE_REQUEST for label in non_identifiable if label == "fiber"
        }
        decision_policy = DecisionPolicy(
            final_lower_bound=args.decision_lower_bound,
            minimum_support=args.decision_min_support,
            candidate_order=candidate_order,
            non_identifiable_labels=non_identifiable,
            non_identifiable_evidence=non_identifiable_evidence,
        )

        bundle, training_artifacts = fit_offline_knowledge(
            train_cases,
            source_dataset=str(args.data_dir),
            split_manifest_hash=str(split_manifest.get("source_hash", "")),
            feature_profile=args.feature_profile,
            policies=(policy,),
            reasoner=reasoner,
            top_k=args.top_k,
            target_selective_risk=args.target_selective_risk,
            decision_minimum_support=args.decision_min_support,
            decision_candidate_order=candidate_order,
            decision_non_identifiable_labels=non_identifiable,
            decision_non_identifiable_evidence=non_identifiable_evidence,
            decision_class_conditional=args.class_conditional_bounds,
            build_metadata={
                "created_at_utc": _utc_now(),
                "git_revision": _git_revision(repo),
                "seed": args.seed,
                "llm_backend": backend.name,
                "model_path": args.model_path,
                "prompt_template": PROMPT_TEMPLATE_VERSION,
                "prompt_template_hash": prompt_template_hash(),
            },
        )
        knowledge_dir = args.output_dir / "knowledge"
        bundle_path = bundle.save(knowledge_dir / "knowledge_bundle.json")
        _write_json(knowledge_dir / "training_summary.json", training_artifacts.summary)
        _write_json(knowledge_dir / "training_traces.json", training_artifacts.traces)

        # Enforce the train/test boundary: evaluation uses a fresh object loaded
        # from the persisted bundle and never fits on test data.
        bundle = OfflineKnowledgeBundle.load(bundle_path)
        # 门禁工作点也必须来自持久化的知识包，不能用进程内的临时对象，
        # 否则「阈值只在训练集上定出来」这条边界就没有被实际验证过。
        if policy.name in bundle.decision_policies:
            decision_policy = bundle.decision_policies[policy.name]
            print(f"decision policy from bundle: {decision_policy.fitted_on}")
        test_packs, test_features = bundle.extract_test_features(
            test_cases,
            source_dataset=str(args.data_dir),
        )
        test_results = match_many(bundle.graph, test_features, top_k=args.top_k)
        report, records, traces = run_policy(
            policy,
            bundle.graph,
            (),
            (),
            (),
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
        )
        reports = {policy.name: report}
        outcomes = {policy.name: records}
        trace_payload = {
            policy.name: {
                case_id: trace.to_dict() for case_id, trace in sorted(traces.items())
            }
        }

        run_manifest = {
            "schema_version": "offline-sop-llm-run-manifest-v1",
            "created_at_utc": _utc_now(),
            "python_version": platform.python_version(),
            "git_revision": _git_revision(repo),
            "scope": {
                "self_evolution": False,
                "feedback_update": False,
                "flow": "manifest train knowledge build -> persisted bundle reload -> manifest test inference",
            },
            "data": {
                "data_dir": str(args.data_dir),
                "train_size": len(train_cases),
                "test_size": len(test_cases),
                "split_manifest_schema": split_manifest.get("schema_version"),
                "split_manifest_hash": split_manifest.get("source_hash"),
                "train_case_ids": list(bundle.train_case_ids),
                "test_case_ids": [pack.case_id for pack in test_packs],
            },
            "knowledge": {
                "schema_version": KNOWLEDGE_BUNDLE_SCHEMA,
                "path": str(bundle_path),
                "content_hash": bundle.content_hash(),
                "historical_vector_count": len(bundle.training_features),
                "evidence_graph_version": bundle.graph.version,
                "evidence_graph_diagnosis_count": len(bundle.graph.case_diagnoses),
                "feature_profile": bundle.feature_profile,
                "feature_dictionary_version": bundle.graph.dictionary_version,
                "feature_dictionary_hash": bundle.graph.dictionary_hash,
                "learned_sop_version": bundle.sop.version,
                "learned_sop_hash": bundle.sop.content_hash(),
                "constraint_library_version": CONSTRAINT_LIBRARY.version,
                "constraint_library_hash": CONSTRAINT_LIBRARY.content_hash(),
            },
            "versions": {
                "evidence_graph": bundle.graph.version,
                "feature_dictionary": bundle.graph.dictionary_version,
                "feature_dictionary_hash": bundle.graph.dictionary_hash,
                "constraint_library": CONSTRAINT_LIBRARY.version,
                "constraint_library_hash": CONSTRAINT_LIBRARY.content_hash(),
                "sop": bundle.sop.version,
                "sop_hash": bundle.sop.content_hash(),
                "prompt_template": PROMPT_TEMPLATE_VERSION,
                "prompt_template_hash": prompt_template_hash(),
                "decision_policy": decision_policy.version,
            },
            "data_quality": {
                "manifest_all_cases": split_manifest.get("quality_summary", {}),
                "train": _quality_summary(
                    train_cases,
                    build_packs(train_cases, source_dataset=str(args.data_dir)),
                    bundle.training_features,
                ),
                "test": _quality_summary(test_cases, test_packs, test_features),
            },
            "retrieval": {"top_k": args.top_k, "policy": policy.to_dict()},
            "decision": decision_policy.to_dict(),
            "llm": {
                "backend": backend.name,
                "model_path": args.model_path,
                "max_attempts": args.max_attempts,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "disable_custom_all_reduce": args.disable_custom_all_reduce,
                "seed": args.seed,
                "prompt_template": PROMPT_TEMPLATE_VERSION,
                "prompt_template_hash": prompt_template_hash(),
            },
            "leakage_policy": (
                "训练标签仅用于训练阈值、特征模型、历史证据图、SOP 与 train-LOO 标定；"
                "测试标签不进入证据包、特征抽取、检索、路由、SOP、LLM prompt 或 M9 决策，"
                "只在推理完成后计算指标与报告对错。"
            ),
            "gpu": {"before": gpu_before, "after": None, "memory_released": None},
        }
        summary = {
            "schema_version": "offline-sop-llm-summary-v1",
            "knowledge": training_artifacts.summary,
            "policies": reports,
        }
        _write_json(args.output_dir / "summary.json", summary)
        _write_json(manifest_path, run_manifest)
        _write_json(args.output_dir / "outcomes.json", outcomes)
        _write_json(args.output_dir / "traces.json", trace_payload)
        html_manifest = render_experiment_html(
            args.output_dir / "html",
            summary=summary,
            manifest=run_manifest,
            outcomes=outcomes,
            traces=trace_payload,
            training_summary=training_artifacts.summary,
        )
        _write_json(args.output_dir / "html" / "report_manifest.json", html_manifest)
        show(report)
    except Exception as exc:
        lifecycle["status"] = "failed"
        lifecycle["error"] = f"{type(exc).__name__}: {exc}"
        raise
    else:
        lifecycle["status"] = "completed"
    finally:
        try:
            backend.close()
            lifecycle["backend_close_called"] = True
        finally:
            reasoner = None
            gc.collect()
            gpu_after = query_gpu_state()
            released = _gpu_memory_released(gpu_before, gpu_after)
            lifecycle["finished_at_utc"] = _utc_now()
            lifecycle["gpu_after"] = gpu_after
            lifecycle["gpu_memory_released"] = released
            _write_json(lifecycle_path, lifecycle)
            if run_manifest:
                run_manifest["gpu"]["after"] = gpu_after
                run_manifest["gpu"]["memory_released"] = released
                run_manifest["gpu"]["backend_close_called"] = lifecycle["backend_close_called"]
                _write_json(manifest_path, run_manifest)


if __name__ == "__main__":
    main()
