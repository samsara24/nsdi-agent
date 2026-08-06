from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict

from .data import load_cases, prepare_dataset
from .pipeline import PipelineConfig, RCAPipeline
from .runtime import RuntimeConfig


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def runtime_from_args(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        llm_backend=args.backend,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        guided_json=not args.disable_guided_json,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )


def prepare_command(args: argparse.Namespace) -> Dict[str, Any]:
    secret = os.environ.get("RCA_ANONYMIZATION_SECRET") or secrets.token_urlsafe(32)
    report = prepare_dataset(Path(args.input_dir), Path(args.output_dir), secret, Path(args.archive_manifest))
    return {
        "output_dir": args.output_dir,
        "source_file_count": report["source_file_count"],
        "output_file_count": report["output_file_count"],
        "skipped_file_count": report["skipped_file_count"],
        "label_distribution": report["label_distribution"],
        "input_speed_patterns": report["input_speed_patterns"],
        "residual_sensitive_patterns": report["residual_sensitive_patterns"],
        "manifest": str(Path(args.output_dir) / "_metadata" / "manifest.json"),
    }


def train_command(args: argparse.Namespace) -> Dict[str, Any]:
    cases = load_cases(Path(args.data_dir))
    if args.train_size <= 0 or args.train_size >= len(cases):
        raise ValueError(f"train-size must be between 1 and {len(cases) - 1}")
    config = PipelineConfig(
        min_edge_count=args.min_edge_count,
        min_rule_count=args.min_rule_count,
        max_rules_per_class=args.max_rules_per_class,
    )
    pipeline = RCAPipeline(config).fit(cases[:args.train_size])
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {output}")
    model_dir = output / "model"
    pipeline.save(model_dir)
    evaluation = pipeline.evaluate(cases[args.train_size:], runtime=runtime_from_args(args))
    dump_json(output / "evaluation_summary.json", evaluation["summary"])
    dump_json(output / "predictions.json", evaluation["predictions"])
    run_manifest = {
        "data_dir": str(Path(args.data_dir)),
        "train_size": args.train_size,
        "test_size": len(cases) - args.train_size,
        "train_case_ids": [case["case_id"] for case in cases[:args.train_size]],
        "test_case_ids": [case["case_id"] for case in cases[args.train_size:]],
        "backend": args.backend,
        "llm_runtime": {
            "model_path": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
            "guided_json": not args.disable_guided_json,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
        },
        "leakage_policy": "All fitted artifacts use the first train-size cases only; test labels are read after inference solely for metrics.",
    }
    dump_json(output / "run_manifest.json", run_manifest)
    return {
        "output_dir": str(output),
        "model_dir": str(model_dir),
        "summary": evaluation["summary"],
        "graph": {"nodes": len(pipeline.graph.nodes), "edges": len(pipeline.graph.edges)},
        "rules": {label: len(items) for label, items in pipeline.rules.rule_sets.items()},
        "rule_overlap": pipeline.rules.overlap_audit()["total_overlap_count"],
    }


def infer_command(args: argparse.Namespace) -> Dict[str, Any]:
    pipeline = RCAPipeline.load(Path(args.model))
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    result = pipeline.infer(case, runtime=runtime_from_args(args))
    if args.output:
        dump_json(Path(args.output), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RCA v2: anonymization, anomaly KG-RAG-LLM, exclusive rules and conflict fusion")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="preserve source manifest and create a new anonymized L1/L2 dataset")
    prepare.add_argument("--input-dir", default="data")
    prepare.add_argument("--output-dir", default="datasets/rca_v2")
    prepare.add_argument("--archive-manifest", default="archive/legacy_exploration/source_data_manifest.json")
    prepare.set_defaults(handler=prepare_command)

    train = sub.add_parser("train-evaluate", help="fit on the training prefix and evaluate without test-label leakage")
    train.add_argument("--data-dir", default="datasets/rca_v2")
    train.add_argument("--output-dir", default="artifacts/rca_v2_baseline")
    train.add_argument("--train-size", type=int, default=200)
    train.add_argument("--min-edge-count", type=int, default=1)
    train.add_argument("--min-rule-count", type=int, default=2)
    train.add_argument("--max-rules-per-class", type=int, default=40)
    train.add_argument("--backend", choices=("none", "vllm", "transformers"), default="none")
    train.add_argument("--model-path", default="")
    train.add_argument("--max-new-tokens", type=int, default=512)
    train.add_argument("--tensor-parallel-size", type=int, default=1)
    train.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    train.add_argument("--max-model-len", type=int, default=8192)
    train.add_argument("--dtype", default="auto")
    train.add_argument("--enforce-eager", action="store_true")
    train.add_argument("--disable-guided-json", action="store_true")
    train.add_argument("--disable-custom-all-reduce", action="store_true")
    train.set_defaults(handler=train_command)

    infer = sub.add_parser("infer", help="infer one already-anonymized schema-v2 case")
    infer.add_argument("--model", required=True)
    infer.add_argument("--case", required=True)
    infer.add_argument("--output")
    infer.add_argument("--backend", choices=("none", "vllm", "transformers"), default="none")
    infer.add_argument("--model-path", default="")
    infer.add_argument("--max-new-tokens", type=int, default=512)
    infer.add_argument("--tensor-parallel-size", type=int, default=1)
    infer.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    infer.add_argument("--max-model-len", type=int, default=8192)
    infer.add_argument("--dtype", default="auto")
    infer.add_argument("--enforce-eager", action="store_true")
    infer.add_argument("--disable-guided-json", action="store_true")
    infer.add_argument("--disable-custom-all-reduce", action="store_true")
    infer.set_defaults(handler=infer_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
