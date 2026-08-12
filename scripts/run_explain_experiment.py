"""迭代 5：LLM 作为解释器，评测机器可判的可核对性。

结论由专家规则给出，模型只负责讲清楚。因此这里不评准确率——
准确率衡量的是规则，不是解释。评的是四项可核对性
（token 存在、token 相关、方向一致、结论一致），全部无需人工打分，
也无需标签，因此同一套指标可以直接搬到生产上做持续监控。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.expert import diagnose_many
from rca_framework.features import dictionary_for, extract_features, fit_feature_model
from rca_framework.llm.backend import Backend, NoneBackend, VLLMBackend
from rca_framework.llm.explain import (
    EXPLAIN_PROMPT_VERSION,
    build_explain_prompt,
    check_explanation,
    parse_explanation,
    summarize_checks,
)


def _rule_evidence(diagnosis) -> List[Tuple[str, str]]:
    """规则实际依据的 (侧, 指标)。兜底类规则没有依据，返回空。"""
    pairs: List[Tuple[str, str]] = []
    for side in diagnosis.sides:
        for metric, _kind in side.anomalies:
            pairs.append((side.side, metric))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-profile", default="v3")
    parser.add_argument("--disable-custom-all-reduce", action="store_true", default=True)
    parser.add_argument(
        "--enable-custom-all-reduce",
        dest="disable_custom_all_reduce",
        action="store_false",
    )
    args = parser.parse_args()

    train_cases = cases_by_manifest_split(args.data_dir, "train")
    test_cases = cases_by_manifest_split(args.data_dir, "test")
    train_packs, test_packs = build_packs(train_cases), build_packs(test_cases)
    test_labels = labels_of(test_cases)
    diagnoses = diagnose_many(test_packs)

    thresholds = fit_thresholds(train_cases)
    dictionary = dictionary_for(args.feature_profile)
    model = fit_feature_model(train_packs, dictionary=dictionary)
    features = [
        extract_features(pack, thresholds, model, dictionary=dictionary)
        for pack in test_packs
    ]

    prompts = [
        build_explain_prompt(
            case_id=pack.case_id,
            expert_verdict=diagnosis.verdict or "abstain",
            expert_group=diagnosis.group,
            expert_reason=diagnosis.reason,
            rule_evidence=_rule_evidence(diagnosis),
            evidence_tokens=feature.tokens,
            missing_fields=sorted(getattr(pack, "missing_fields", ()) or ()),
        )
        for pack, diagnosis, feature in zip(test_packs, diagnoses, features)
    ]

    backend: Backend = (
        VLLMBackend(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_tokens,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            seed=args.seed,
        )
        if args.model_path
        else NoneBackend()
    )
    outputs = backend.generate(prompts)
    backend.close()

    records: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    for pack, diagnosis, feature, label, prompt, raw in zip(
        test_packs, diagnoses, features, test_labels, prompts, outputs
    ):
        explanation = parse_explanation(raw)
        report = check_explanation(
            explanation,
            available_tokens=feature.tokens,
            rule_evidence=_rule_evidence(diagnosis),
            rule_verdict=diagnosis.verdict,
        )
        reports.append(report)
        records.append({
            "case_id": pack.case_id,
            "gold": label,
            "expert_verdict": diagnosis.verdict,
            "expert_group": diagnosis.group,
            "rule_correct": diagnosis.verdict == label,
            "explanation": explanation.to_dict() if explanation else None,
            "checks": report,
            "raw_output": raw,
            "prompt": prompt,
        })

    summary = {
        "schema_version": "explain-experiment-v1",
        "prompt_version": EXPLAIN_PROMPT_VERSION,
        "backend": backend.name,
        "feature_profile": args.feature_profile,
        "seed": args.seed,
        "checkability": summarize_checks(reports),
        # 解释合格率在「规则判对」与「规则判错」两组上分开报：
        # 若两组差异很大，说明解释质量与结论正确性纠缠，
        # 那么这个指标就不能脱离标签独立使用。
        "by_rule_correctness": {
            key: summarize_checks([
                report
                for report, record in zip(reports, records)
                if record["rule_correct"] is value
            ])
            for key, value in (("rule_correct", True), ("rule_wrong", False))
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
