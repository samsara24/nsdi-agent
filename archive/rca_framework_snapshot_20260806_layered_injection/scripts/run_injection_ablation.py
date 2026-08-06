#!/usr/bin/env python
"""Ablate KG injection and LLM scoring on one shared model load.

The four arms isolate the two changes made on 2026-08-05:

- ``full`` + ``legacy``   reproduces the pre-change behaviour.
- ``full`` + ``llm_only`` removes only the KG score blended back into the LLM route.
- ``layered`` + ``legacy`` withholds only the KG score that does not apply to the case.
- ``layered`` + ``llm_only`` is the proposed configuration.

``RCAPipeline`` caches reasoners by model-loading settings only, so all four
arms share a single vLLM instance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from rca_framework.data import load_cases
from rca_framework.pipeline import PipelineConfig, RCAPipeline

ARMS = (
    ("full", "legacy"),
    ("full", "llm_only"),
    ("layered", "legacy"),
    ("layered", "llm_only"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/organized_rca_v2_stratified_60_40_seed42")
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=("none", "vllm", "transformers"), default="vllm")
    parser.add_argument("--model-path", default="/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--disable-guided-json", action="store_true")
    parser.add_argument("--insufficient-confidence-scale", type=float, default=1.0)
    return parser.parse_args()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {output}")

    cases = load_cases(Path(args.data_dir))
    if not 0 < args.train_size < len(cases):
        raise ValueError(f"train-size must be between 1 and {len(cases) - 1}")
    train, test = cases[: args.train_size], cases[args.train_size :]
    pipeline = RCAPipeline(PipelineConfig()).fit(train)
    pipeline.save(output / "model")

    runtime = {
        "llm_backend": args.backend,
        "model_path": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "guided_json": not args.disable_guided_json,
        "insufficient_confidence_scale": args.insufficient_confidence_scale,
    }

    comparison: List[Dict[str, Any]] = []
    for injection_mode, score_mode in ARMS:
        arm = f"{injection_mode}__{score_mode}"
        print(f"[ablation] arm {arm}", flush=True)
        started = time.time()
        evaluation = pipeline.evaluate(test, injection_mode=injection_mode, score_mode=score_mode, **runtime)
        elapsed = round(time.time() - started, 2)
        summary = evaluation["summary"]
        summary["wall_clock_seconds"] = elapsed
        dump_json(output / arm / "evaluation_summary.json", summary)
        dump_json(output / arm / "predictions.json", evaluation["predictions"])
        comparison.append({
            "arm": arm,
            "injection_mode": injection_mode,
            "score_mode": score_mode,
            "correct": summary["correct"],
            "case_count": summary["case_count"],
            "accuracy": summary["accuracy"],
            "recall": summary["recall"],
            "confusion_matrix": summary["confusion_matrix"],
            "valid_llm_outputs": summary["valid_llm_outputs"],
            "decision_status": summary["decision_status"],
            "kg_coverage_regime": summary["kg_coverage_regime"],
            "evidence_sufficiency": summary["evidence_sufficiency"],
            "wall_clock_seconds": elapsed,
        })
        print(f"[ablation] {arm}: {summary['correct']}/{summary['case_count']} "
              f"acc={summary['accuracy']:.4f} in {elapsed}s", flush=True)

    dump_json(output / "run_manifest.json", {
        "data_dir": str(Path(args.data_dir)),
        "train_size": args.train_size,
        "test_size": len(test),
        "train_case_ids": [case["case_id"] for case in train],
        "test_case_ids": [case["case_id"] for case in test],
        "llm_runtime": runtime,
        "arms": [f"{a}__{b}" for a, b in ARMS],
        "shared_model_load": "all arms reuse one reasoner instance keyed by model-loading settings",
        "leakage_policy": "All fitted artifacts use the training prefix only; test labels are read after inference solely for metrics.",
    })
    dump_json(output / "ablation_comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
