"""迭代 4 主实验：LLM 作为质疑器，而不是定界器。

专家规则对每条 case 出 verdict，LLM 只回答「这条规则的前提是否有一条不成立」。
考核指标是质疑命中率——被质疑的 case 里规则确实判错的比例——
而不是准确率，因为这个岗位不要求 LLM 比规则更准。

同时评测两个必须被超过的基线：

1. **随机质疑**：命中率等于规则错误率本身。
2. **按 train 组可靠性查表**：在训练集上算每个专家规则组的准确率，
   把低于阈值的组在测试集上全部标为可疑。这是「不用 LLM 也能做的质疑」，
   如果 LLM 赢不过它，这个岗位就不成立。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.expert import diagnose_many
from rca_framework.features import dictionary_for, extract_features, fit_feature_model
from rca_framework.llm.backend import Backend, NoneBackend, VLLMBackend
from rca_framework.llm.challenge import (
    CHALLENGE_PROMPT_VERSION,
    build_challenge_prompt,
    challenge_metrics,
    parse_challenge,
)


def _missing_fields(pack) -> List[str]:
    return sorted(pack.missing_fields) if getattr(pack, "missing_fields", None) else []


def group_reliability_baseline(
    train_rows: Sequence[Tuple[str, bool]],
    test_groups: Sequence[str],
    *,
    threshold: float,
    minimum_support: int = 3,
) -> List[bool]:
    """train 上准确率低于 threshold 的组，在 test 上一律标为可疑。"""
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for group, correct in train_rows:
        stats[group][1] += 1
        stats[group][0] += int(correct)
    suspect = {
        group
        for group, (correct, total) in stats.items()
        if total >= minimum_support and correct / total < threshold
    }
    return [group in suspect for group in test_groups]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-profile", default="v3")
    parser.add_argument("--baseline-threshold", type=float, default=0.60)
    args = parser.parse_args()

    train_cases = cases_by_manifest_split(args.data_dir, "train")
    test_cases = cases_by_manifest_split(args.data_dir, "test")
    train_packs, test_packs = build_packs(train_cases), build_packs(test_cases)
    train_labels, test_labels = labels_of(train_cases), labels_of(test_cases)
    train_diag, test_diag = diagnose_many(train_packs), diagnose_many(test_packs)

    thresholds = fit_thresholds(train_cases)
    dictionary = dictionary_for(args.feature_profile)
    model = fit_feature_model(train_packs, dictionary=dictionary)
    features = [
        extract_features(pack, thresholds, model, dictionary=dictionary)
        for pack in test_packs
    ]

    prompts = [
        build_challenge_prompt(
            case_id=pack.case_id,
            expert_verdict=diagnosis.verdict or "abstain",
            expert_group=diagnosis.group,
            expert_reason=diagnosis.reason,
            evidence_tokens=feature.tokens,
            missing_fields=_missing_fields(pack),
        )
        for pack, diagnosis, feature in zip(test_packs, test_diag, features)
    ]

    backend: Backend = (
        VLLMBackend(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_tokens,
            seed=args.seed,
        )
        if args.model_path
        else NoneBackend()
    )
    outputs = backend.generate(prompts)
    backend.close()

    records: List[Dict[str, Any]] = []
    rows: List[Tuple[bool, bool]] = []
    parse_failures = 0
    for pack, diagnosis, feature, label, prompt, raw in zip(
        test_packs, test_diag, features, test_labels, prompts, outputs
    ):
        parsed = parse_challenge(raw)
        if parsed is None:
            parse_failures += 1
        # 解析失败按 agree 处理：无法解析不构成质疑理由，
        # 算成质疑会直接污染命中率。
        challenged = bool(parsed and parsed.challenges)
        rule_wrong = diagnosis.verdict != label
        rows.append((challenged, rule_wrong))
        records.append({
            "case_id": pack.case_id,
            "gold": label,
            "expert_verdict": diagnosis.verdict,
            "expert_group": diagnosis.group,
            "rule_wrong": rule_wrong,
            "challenged": challenged,
            "response": parsed.to_dict() if parsed else None,
            "raw_output": raw,
            "prompt": prompt,
            "evidence_tokens": list(feature.tokens),
        })

    llm_metrics = challenge_metrics(rows)
    baseline_flags = group_reliability_baseline(
        [(d.group, d.verdict == l) for d, l in zip(train_diag, train_labels)],
        [d.group for d in test_diag],
        threshold=args.baseline_threshold,
    )
    baseline_metrics = challenge_metrics(
        [(flag, wrong) for flag, (_, wrong) in zip(baseline_flags, rows)]
    )

    summary = {
        "schema_version": "challenge-experiment-v1",
        "prompt_version": CHALLENGE_PROMPT_VERSION,
        "backend": backend.name,
        "feature_profile": args.feature_profile,
        "seed": args.seed,
        "parse_failures": parse_failures,
        "llm_challenger": llm_metrics,
        "group_reliability_baseline": {
            "threshold": args.baseline_threshold,
            **baseline_metrics,
        },
        "premise_distribution": {
            premise: sum(
                1
                for record in records
                if record["response"] and record["response"]["premise_at_risk"] == premise
            )
            for premise in sorted(
                {
                    record["response"]["premise_at_risk"]
                    for record in records
                    if record["response"] and record["response"]["premise_at_risk"]
                }
            )
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
