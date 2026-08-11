#!/usr/bin/env python3
"""审计候选规则：它是新证据，还是已有 token 的代理？

挖掘脚本（`mine_knowledge_candidates.py`）只回答「这条规则在训练集上准不准」。
一条规则可以又准又毫无价值：如果它总是和某个已有 token 同时出现，
那它没有引入任何新信息，写进约束库只是把同一件事说两遍，
并且会让 LLM 误以为自己找到了两条独立证据。

本脚本对每个候选做三项审计，全部只在 train split 上算标签：

1. **共线性**：找出与候选重叠最大的已有字典 token（Jaccard 与条件命中率）。
2. **分层剩余增益**：在最强共线者**不命中**的子集里，候选是否还能预测同一标签、
   且 Wilson 下界仍超过该标签的先验。过不了这一项的候选是冗余的。
3. **形态偏差检验**：候选在「遥测无任何断 lane」的 case 上的命中率。
   一个真正的故障信号不应该在没有断 lane 的 case 上普遍命中；
   若普遍命中，它更可能是两端端口形态（400G vs 200G）的固定差异，
   也就是 C3 里已经记录过的那类系统性偏移，属于过拟合陷阱。

测试集只用于报告候选的触发率，不参与任何标签统计。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mine_knowledge_candidates import ROOT_CAUSES, probe_tokens, wilson_bounds  # noqa: E402
from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.features import dictionary_for, extract_features, fit_feature_model  # noqa: E402

#: 默认审计对象：挖掘结果里 Wilson 下界最高、且尚未进入特征字典的 probe。
DEFAULT_CANDIDATES = (
    "probe:txpower_side_gap:L1_worse",
    "probe:media_snr_spread:L1:wide",
    "probe:media_snr_side_gap:L1_worse",
    "probe:rxpower_spread:L1:wide",
    "probe:rxpower_side_gap:L1_worse",
    "probe:media_snr_spread:L2:tight",
    "probe:alarm_side:L2_ENDPOINT",
    "probe:lane_width:equal",
)


def summarise(labels: Sequence[str], prior: Dict[str, float]) -> Dict[str, Any]:
    if not labels:
        return {"support": 0}
    counts = Counter(labels)
    best = max(ROOT_CAUSES, key=lambda label: (counts.get(label, 0), -ROOT_CAUSES.index(label)))
    correct = counts.get(best, 0)
    lower, _ = wilson_bounds(correct, len(labels))
    base = prior.get(best, 0.0)
    return {
        "support": len(labels),
        "predicts": best,
        "precision": round(correct / len(labels), 4),
        "wilson_lower_bound": lower,
        "prior_for_predicted_label": round(base, 4),
        "beats_prior_at_95": lower > base,
        "label_counts": {label: counts.get(label, 0) for label in ROOT_CAUSES},
    }


def audit(
    candidate: str,
    train_tokens: Sequence[Set[str]],
    train_labels: Sequence[str],
    quiet_flags: Sequence[bool],
    test_tokens: Sequence[Set[str]],
    prior: Dict[str, float],
    *,
    min_support: int,
    top_confounders: int,
) -> Dict[str, Any]:
    hit = [index for index, tokens in enumerate(train_tokens) if candidate in tokens]
    overall = summarise([train_labels[index] for index in hit], prior)

    dictionary_tokens: Counter = Counter()
    for tokens in train_tokens:
        for token in tokens:
            if not token.startswith("probe:"):
                dictionary_tokens[token] += 1

    hit_set = set(hit)
    overlaps: List[Dict[str, Any]] = []
    for token, count in dictionary_tokens.items():
        if count < min_support:
            continue
        other = {index for index, tokens in enumerate(train_tokens) if token in tokens}
        if not other:
            continue
        intersection = len(hit_set & other)
        union = len(hit_set | other)
        overlaps.append(
            {
                "token": token,
                "jaccard": round(intersection / union, 4) if union else 0.0,
                "candidate_given_token": round(intersection / len(other), 4),
                "token_given_candidate": round(intersection / len(hit_set), 4) if hit_set else 0.0,
                "support": count,
            }
        )
    overlaps.sort(key=lambda item: -item["jaccard"])

    strata: List[Dict[str, Any]] = []
    for item in overlaps[:top_confounders]:
        token = item["token"]
        outside = [
            index
            for index in range(len(train_tokens))
            if token not in train_tokens[index]
        ]
        with_candidate = [index for index in outside if candidate in train_tokens[index]]
        without_candidate = [index for index in outside if candidate not in train_tokens[index]]
        strata.append(
            {
                "controlled_for": token,
                "jaccard": item["jaccard"],
                "candidate_within_token_negative_stratum": summarise(
                    [train_labels[index] for index in with_candidate], prior
                ),
                "baseline_within_token_negative_stratum": summarise(
                    [train_labels[index] for index in without_candidate], prior
                ),
            }
        )

    quiet_indices = [index for index, flag in enumerate(quiet_flags) if flag]
    quiet_hits = [index for index in quiet_indices if candidate in train_tokens[index]]
    return {
        "candidate": candidate,
        "train": overall,
        "test_hits": sum(1 for tokens in test_tokens if candidate in tokens),
        "test_total": len(test_tokens),
        "top_collinear_dictionary_tokens": overlaps[:top_confounders],
        "stratified_residual_gain": strata,
        "morphology_check": {
            "quiet_cases": len(quiet_indices),
            "candidate_fires_on_quiet_cases": len(quiet_hits),
            "fire_rate_on_quiet_cases": (
                round(len(quiet_hits) / len(quiet_indices), 4) if quiet_indices else None
            ),
            "fire_rate_overall": round(len(hit) / len(train_tokens), 4) if train_tokens else 0.0,
        },
    }


def verdict(entry: Dict[str, Any]) -> str:
    """把三项审计压成一个可执行结论。"""
    train = entry["train"]
    if not train.get("beats_prior_at_95"):
        return "reject:训练集上就没有超过先验"
    morphology = entry["morphology_check"]
    quiet_rate = morphology.get("fire_rate_on_quiet_cases")
    if quiet_rate is not None and quiet_rate >= 0.5:
        return "reject:在无断 lane 的 case 上也过半命中，更像端口形态偏差"
    strata = entry["stratified_residual_gain"]
    if not strata:
        return "promote:没有可比的共线 token，属于独立信号"
    survived = [
        item
        for item in strata
        if item["candidate_within_token_negative_stratum"].get("support", 0) >= 8
        and item["candidate_within_token_negative_stratum"].get("beats_prior_at_95")
    ]
    if survived:
        return f"promote:在 {len(survived)}/{len(strata)} 个控制分层里仍有剩余增益"
    testable = [
        item
        for item in strata
        if item["candidate_within_token_negative_stratum"].get("support", 0) >= 8
    ]
    if not testable:
        return "hold:去掉共线 token 后样本不足，无法判定"
    return "reject:控制共线 token 后不再有增益，属于已有 token 的代理"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--feature-profile", default="v2")
    parser.add_argument("--candidates", nargs="*", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--top-confounders", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/i1_candidate_audit.json"))
    args = parser.parse_args()

    train_cases = cases_by_manifest_split(args.data_dir, "train")
    test_cases = cases_by_manifest_split(args.data_dir, "test")
    train_labels = [str(case["label"]) for case in train_cases]

    dictionary = dictionary_for(args.feature_profile)
    train_packs = build_packs(train_cases, source_dataset=str(args.data_dir))
    test_packs = build_packs(test_cases, source_dataset=str(args.data_dir))
    thresholds = fit_thresholds(train_cases)
    model = fit_feature_model(train_packs, dictionary=dictionary)

    def tokens_for(packs, cases) -> List[Set[str]]:
        out: List[Set[str]] = []
        for pack, case in zip(packs, cases):
            out.append(
                set(extract_features(pack, thresholds, model, dictionary=dictionary).tokens)
                | probe_tokens(case)
            )
        return out

    train_tokens = tokens_for(train_packs, train_cases)
    test_tokens = tokens_for(test_packs, test_cases)

    # 「安静」case：没有任何 drop token，即遥测里看不到断 lane。
    quiet_flags = [
        not any(token.startswith("drop:") for token in tokens) for tokens in train_tokens
    ]

    counts = Counter(train_labels)
    prior = {label: counts.get(label, 0) / len(train_labels) for label in ROOT_CAUSES}

    entries = [
        audit(
            candidate,
            train_tokens,
            train_labels,
            quiet_flags,
            test_tokens,
            prior,
            min_support=args.min_support,
            top_confounders=args.top_confounders,
        )
        for candidate in args.candidates
    ]
    for entry in entries:
        entry["verdict"] = verdict(entry)

    report = {
        "schema_version": "candidate-audit-v1",
        "data_dir": str(args.data_dir),
        "feature_dictionary_version": dictionary.version,
        "train_size": len(train_cases),
        "test_size": len(test_cases),
        "train_prior": {label: round(value, 4) for label, value in prior.items()},
        "quiet_case_definition": "训练集中不含任何 drop: 前缀 token 的 case",
        "label_policy": "标签统计只在 train split；测试集只报触发次数。",
        "audits": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train={len(train_cases)} test={len(test_cases)} prior={report['train_prior']}")
    print(f"安静 case（无 drop token）：{sum(quiet_flags)}/{len(train_labels)}\n")
    for entry in entries:
        train = entry["train"]
        morphology = entry["morphology_check"]
        print(f"── {entry['candidate']}")
        print(
            f"   train: n={train.get('support')} predicts={train.get('predicts')}"
            f" prec={train.get('precision')} lb={train.get('wilson_lower_bound')}"
            f" prior={train.get('prior_for_predicted_label')}"
            f" | test 触发 {entry['test_hits']}/{entry['test_total']}"
        )
        print(
            f"   形态检验: 安静 case 命中率 {morphology['fire_rate_on_quiet_cases']}"
            f" vs 全体 {morphology['fire_rate_overall']}"
        )
        for item in entry["stratified_residual_gain"]:
            inner = item["candidate_within_token_negative_stratum"]
            base = item["baseline_within_token_negative_stratum"]
            print(
                f"   控制 {item['controlled_for']} (J={item['jaccard']}):"
                f" 候选 n={inner.get('support')} prec={inner.get('precision')}"
                f" lb={inner.get('wilson_lower_bound')} 超先验={inner.get('beats_prior_at_95')}"
                f" | 该层基线 n={base.get('support')} prec={base.get('precision')}"
            )
        print(f"   => {entry['verdict']}\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
