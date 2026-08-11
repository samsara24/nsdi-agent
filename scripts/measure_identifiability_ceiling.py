"""迭代 2 离线分析：当前证据空间的可辨识上限。

前两轮的结论都是负面的（SOP 打不过多数类、门限只能在覆盖与精度间搬运、
L1 候选纯度不超过先验），但一直没回答最基本的那个问题：
**在当前特征字典能表达的证据里，L1 与 L2 到底可不可分？**

方法是不依赖任何模型的：把每条 case 归到它的证据签名（激活 token 的集合），
签名相同的 case 在这个特征空间里**不可区分**——任何分类器（决策树、
最近邻、LLM、人）看到的输入都一样，只能给它们同一个答案。
于是「每个签名取其多数类」就是这个特征空间上任何算法的准确率上界：

    ceiling = Σ_签名 max_类 计数(签名, 类) / N

这个上界是乐观的（它按每个签名的实际多数类取值，等于让模型看过标签），
所以真实算法只会更差。如果上界本身就贴着多数类先验，
那么「提升准确率」这件事在当前遥测下就是不可能的，
需要改的是采集什么，而不是怎么算。

只用训练集标签。测试集只用来统计「签名落点」，不读标签。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.features.dictionary import dictionary_for
from rca_framework.features.extractor import extract_features, fit_feature_model


def signature(feature) -> Tuple[str, ...]:
    return tuple(sorted(feature.tokens))


def ceiling(signatures: Sequence[Tuple[str, ...]], labels: Sequence[str]) -> Dict[str, object]:
    groups: Dict[Tuple[str, ...], Counter] = {}
    for sig, label in zip(signatures, labels):
        groups.setdefault(sig, Counter())[label] += 1
    total = len(labels)
    best = sum(counts.most_common(1)[0][1] for counts in groups.values())
    prior = Counter(labels)
    majority_label, majority_count = prior.most_common(1)[0]
    ambiguous = {sig: counts for sig, counts in groups.items() if len(counts) > 1}
    ambiguous_cases = sum(sum(counts.values()) for counts in ambiguous.values())
    # 冲突对：签名相同、标签不同的 case 对数占所有 case 对的比例，
    # 用来看不可分性是集中在少数大签名上还是散布在全体。
    collisions = 0
    for counts in ambiguous.values():
        items = list(counts.items())
        for index, (_, count_a) in enumerate(items):
            for _, count_b in items[index + 1 :]:
                collisions += count_a * count_b
    return {
        "cases": total,
        "distinct_signatures": len(groups),
        "singleton_signatures": sum(1 for counts in groups.values() if sum(counts.values()) == 1),
        "ceiling_accuracy": round(best / total, 6) if total else None,
        "majority_label": majority_label,
        "majority_accuracy": round(majority_count / total, 6) if total else None,
        "headroom_over_majority": round((best - majority_count) / total, 6) if total else None,
        "ambiguous_signatures": len(ambiguous),
        "cases_in_ambiguous_signatures": ambiguous_cases,
        "share_in_ambiguous_signatures": round(ambiguous_cases / total, 6) if total else None,
        "conflicting_pairs": collisions,
        "largest_ambiguous": [
            {
                "size": sum(counts.values()),
                "labels": dict(sorted(counts.items())),
                "tokens": list(sig),
            }
            for sig, counts in sorted(
                ambiguous.items(), key=lambda item: -sum(item[1].values())
            )[:5]
        ],
    }


def neighbourhood_consistency(
    signatures: Sequence[Tuple[str, ...]],
    labels: Sequence[str],
    thresholds: Sequence[float],
) -> List[Dict[str, object]]:
    """证据越像，根因越可能相同吗？

    精确签名的上界会被稀疏性架空：147 个签名里 137 个只出现一次，
    「每个签名取多数类」等于逐条背答案，上界自然接近 100%，
    却完全无法推广到新 case（实测只有 14% 的测试签名在训练集出现过）。

    所以要在**能推广的粒度**上量：对每个 Jaccard 相似度门限，
    统计相似度达标的 case 对中同根因的比例，并与随机两条同根因的概率
    （Σ p_c²）比较。如果高相似度下同根因率并不显著高于随机，
    那就说明「证据相似」不蕴含「根因相同」——
    这不是算法不够好，是当前遥测不足以确定根因。
    """
    sets = [set(sig) for sig in signatures]
    prior = Counter(labels)
    total = len(labels)
    chance = sum((count / total) ** 2 for count in prior.values()) if total else 0.0
    pairs: List[Tuple[float, int]] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            similarity = len(sets[i] & sets[j]) / len(union) if union else 1.0
            pairs.append((similarity, int(labels[i] == labels[j])))
    out = []
    for threshold in thresholds:
        selected = [hit for similarity, hit in pairs if similarity >= threshold]
        cases_with_neighbour = sum(
            1
            for i in range(len(sets))
            if any(
                i != j
                and (
                    len(sets[i] & sets[j]) / len(sets[i] | sets[j]) if sets[i] | sets[j] else 1.0
                )
                >= threshold
                for j in range(len(sets))
            )
        )
        out.append(
            {
                "min_similarity": threshold,
                "pairs": len(selected),
                "same_root_cause_rate": round(sum(selected) / len(selected), 6)
                if selected
                else None,
                "chance_rate": round(chance, 6),
                "lift_over_chance": round(sum(selected) / len(selected) - chance, 6)
                if selected
                else None,
                "cases_with_at_least_one_neighbour": cases_with_neighbour,
            }
        )
    return out


def similarity_vote_curve(
    signatures: Sequence[Tuple[str, ...]],
    labels: Sequence[str],
    thresholds: Sequence[float],
) -> List[Dict[str, object]]:
    """邻域一致性的可操作版本：留一法下按相似邻居投票。

    上一个函数量的是「证据相似是否蕴含根因相同」，这个函数把它变成预测器，
    好处是它的置信度是**邻居的一致度**（别人的标签），
    而不是「去掉自己重拟合」得到的叶纯度，因此不存在迭代 2 修掉的那个反序问题。

    没有达到相似度门限的邻居就弃答——这是一条诚实的覆盖率-精度曲线，
    与 M9 的门限曲线可直接比较：关键看它在被保留的同一批 case 上
    有没有打赢「一律报该子集的多数类」。
    """
    sets = [set(sig) for sig in signatures]
    out = []
    for threshold in thresholds:
        predictions: List[Tuple[int, str]] = []
        for i in range(len(sets)):
            votes: Counter = Counter()
            for j in range(len(sets)):
                if i == j:
                    continue
                union = sets[i] | sets[j]
                similarity = len(sets[i] & sets[j]) / len(union) if union else 1.0
                if similarity >= threshold:
                    votes[labels[j]] += similarity
            if votes:
                predictions.append((i, votes.most_common(1)[0][0]))
        answered = len(predictions)
        correct = sum(1 for index, verdict in predictions if verdict == labels[index])
        kept_truths = [labels[index] for index, _ in predictions]
        kept_prior = Counter(kept_truths)
        majority_on_kept = (
            kept_prior.most_common(1)[0][1] / answered if answered else None
        )
        recalls = []
        for label, count in Counter(kept_truths).items():
            hits = sum(
                1
                for index, verdict in predictions
                if labels[index] == label and verdict == label
            )
            recalls.append(hits / count)
        out.append(
            {
                "min_similarity": threshold,
                "answered": answered,
                "coverage": round(answered / len(sets), 6) if sets else None,
                "precision_when_answered": round(correct / answered, 6) if answered else None,
                "majority_on_kept": round(majority_on_kept, 6)
                if majority_on_kept is not None
                else None,
                "lift_over_majority_on_kept": round(correct / answered - majority_on_kept, 6)
                if answered
                else None,
                "balanced_recall_on_kept": round(sum(recalls) / len(recalls), 6)
                if recalls
                else None,
            }
        )
    return out


def per_label_recall_ceiling(
    signatures: Sequence[Tuple[str, ...]], labels: Sequence[str]
) -> Dict[str, Dict[str, object]]:
    """每个类别在上界方案下最多能被召回多少。

    某个类别在所有签名里都不是多数类时，它的召回上界就是 0——
    这比总体上界更能说明「这个类别有没有立足之地」。
    """
    groups: Dict[Tuple[str, ...], Counter] = {}
    for sig, label in zip(signatures, labels):
        groups.setdefault(sig, Counter())[label] += 1
    winners = {sig: counts.most_common(1)[0][0] for sig, counts in groups.items()}
    out: Dict[str, Dict[str, object]] = {}
    for label in sorted(set(labels)):
        total = sum(1 for item in labels if item == label)
        recalled = sum(
            counts[label] for sig, counts in groups.items() if winners[sig] == label
        )
        owned = sum(1 for sig in groups if winners[sig] == label)
        out[label] = {
            "cases": total,
            "recall_ceiling": round(recalled / total, 6) if total else None,
            "signatures_won": owned,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--feature-profile", default="v2")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    train_cases = cases_by_manifest_split(data_dir, "train")
    test_cases = cases_by_manifest_split(data_dir, "test")
    train_labels = labels_of(train_cases)
    dictionary = dictionary_for(args.feature_profile)
    thresholds = fit_thresholds(train_cases)
    train_packs = build_packs(train_cases, source_dataset=str(data_dir))
    test_packs = build_packs(test_cases, source_dataset=str(data_dir))
    model = fit_feature_model(train_packs, dictionary=dictionary)
    train_features = [
        extract_features(pack, thresholds, model, dictionary=dictionary) for pack in train_packs
    ]
    test_features = [
        extract_features(pack, thresholds, model, dictionary=dictionary) for pack in test_packs
    ]
    train_signatures = [signature(feature) for feature in train_features]
    test_signatures = [signature(feature) for feature in test_features]

    report: Dict[str, object] = {
        "feature_profile": args.feature_profile,
        "dictionary_version": dictionary.version,
        "dictionary_hash": dictionary.content_hash(),
        "train": ceiling(train_signatures, train_labels),
        "train_per_label": per_label_recall_ceiling(train_signatures, train_labels),
        "neighbourhood_consistency": neighbourhood_consistency(
            train_signatures, train_labels, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        ),
        "similarity_vote_loo": similarity_vote_curve(
            train_signatures, train_labels, (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
    }

    train_lookup: Dict[Tuple[str, ...], Counter] = {}
    for sig, label in zip(train_signatures, train_labels):
        train_lookup.setdefault(sig, Counter())[label] += 1
    # 测试集只看签名落点，不读标签：这是部署时就能算的量。
    seen = sum(1 for sig in test_signatures if sig in train_lookup)
    ambiguous_hit = sum(
        1 for sig in test_signatures if len(train_lookup.get(sig, ())) > 1
    )
    report["test_signature_placement"] = {
        "cases": len(test_signatures),
        "signature_seen_in_train": seen,
        "share_seen": round(seen / len(test_signatures), 6) if test_signatures else None,
        "landing_on_ambiguous_train_signature": ambiguous_hit,
        "share_ambiguous": round(ambiguous_hit / len(test_signatures), 6)
        if test_signatures
        else None,
    }

    train = report["train"]
    print(f"特征空间      : {dictionary.version} ({dictionary.content_hash()})")
    print(f"训练 case     : {train['cases']}，不同证据签名 {train['distinct_signatures']} 个"
          f"（其中只出现一次的 {train['singleton_signatures']} 个）")
    print(f"可辨识上界    : {train['ceiling_accuracy']:.2%}"
          f"（一律报 {train['majority_label']} 为 {train['majority_accuracy']:.2%}，"
          f"上限余量 {train['headroom_over_majority']:+.2%}）")
    print(f"落在混标签签名: {train['cases_in_ambiguous_signatures']} 条"
          f"（{train['share_in_ambiguous_signatures']:.2%}），冲突对 {train['conflicting_pairs']} 对")
    print("\n各类召回上界（上界方案下该类最多能被认出多少）：")
    for label, item in report["train_per_label"].items():
        print(f"  {label:6s} n={item['cases']:3d}  召回上界 {item['recall_ceiling']:.2%}"
              f"  占据签名 {item['signatures_won']} 个")
    print("\n证据邻域一致性（相似度达标的 case 对中同根因的比例）：")
    print("  相似度>=   case 对    同根因      随机       lift   至少有一个邻居的 case")
    for item in report["neighbourhood_consistency"]:
        rate = item["same_root_cause_rate"]
        if rate is None:
            print(f"  {item['min_similarity']:.2f}      {item['pairs']:6d}        —")
            continue
        print(
            f"  {item['min_similarity']:.2f}      {item['pairs']:6d}   {rate:7.2%}  "
            f"{item['chance_rate']:7.2%}  {item['lift_over_chance']:+7.2%}   "
            f"{item['cases_with_at_least_one_neighbour']:4d}"
        )

    print("\n留一法相似度投票（与 M9 门限曲线同口径）：")
    print("  相似度>=   覆盖率     精度   同子集多数类      lift   平衡召回")
    for item in report["similarity_vote_loo"]:
        if not item["answered"]:
            print(f"  {item['min_similarity']:.2f}      0.00%        —")
            continue
        print(
            f"  {item['min_similarity']:.2f}     {item['coverage']:6.2%}  {item['precision_when_answered']:6.2%}  "
            f"{item['majority_on_kept']:11.2%}  {item['lift_over_majority_on_kept']:+7.2%}  "
            f"{item['balanced_recall_on_kept']:8.4f}"
        )

    print("\n最大的混标签签名：")
    for item in train["largest_ambiguous"]:
        print(f"  size={item['size']:3d} {item['labels']}")
        print(f"    tokens: {', '.join(item['tokens']) or '（无激活 token）'}")
    placement = report["test_signature_placement"]
    print(f"\n测试集签名落点（不读标签）：{placement['signature_seen_in_train']}/"
          f"{placement['cases']} 条的签名在训练集出现过（{placement['share_seen']:.2%}），"
          f"其中落在混标签签名上的 {placement['landing_on_ambiguous_train_signature']} 条"
          f"（{placement['share_ambiguous']:.2%}）")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
