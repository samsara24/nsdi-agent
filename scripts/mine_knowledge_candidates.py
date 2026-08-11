#!/usr/bin/env python3
"""从训练集统计出候选物理约束与 SOP 规则，并用测试集的**无标签**观测检验覆盖率。

方法边界（这是本脚本存在的理由，不是免责声明）：

- 只有训练集提供标签。所有 support / precision / Wilson 下界都在 train split 上算。
- 测试集只提供「这条规则在测试集上触发了几次」。触发次数不依赖标签，
  因此它是 transductive 的覆盖率检查，不是标签泄漏；它回答的是
  「我们沉淀的知识在未见 case 上是否还会说话」，而不是「它在未见 case 上对不对」。
- 因此本脚本永远不输出测试集准确率。测试集准确率只能由正式实验脚本给出。

候选分三层：

1. `token`：单个特征 token 与标签的关联。
2. `pair`：两个 token 的合取，用于捕捉「两端对比」这类单 token 表达不了的物理关系。
3. `probe`：直接从遥测算出的、当前特征字典里还没有的派生量。
   probe 的作用是回答「值不值得新增一个特征家族」，命中后要先写进字典再进推理链路。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.features import dictionary_for, extract_features, fit_feature_model  # noqa: E402

ROOT_CAUSES = ("L1", "L2", "fiber")
DOWN_SENTINEL = -39.0


def wilson_bounds(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * total)) / total)
    return (
        round(max(0.0, (centre - margin) / denominator), 6),
        round(min(1.0, (centre + margin) / denominator), 6),
    )


# --------------------------------------------------------------------------------------
# probe：字典里还没有的派生量
# --------------------------------------------------------------------------------------


def _lane_values(case: Mapping[str, Any], metric: str, side: str) -> List[float]:
    block = case.get(metric)
    if not isinstance(block, Mapping):
        return []
    values = block.get(side)
    if isinstance(values, Mapping):
        items = list(values.values())
    elif isinstance(values, (list, tuple)):
        items = list(values)
    else:
        return []
    out: List[float] = []
    for item in items:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _healthy(values: Sequence[float]) -> List[float]:
    return [value for value in values if value > DOWN_SENTINEL]


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def probe_tokens(case: Mapping[str, Any]) -> Set[str]:
    """派生探针。命名一律带 `probe:` 前缀，避免与正式特征 token 混淆。"""
    tokens: Set[str] = set()

    alarm_interface = str(case.get("alarm_ip_interface") or "")
    tokens.add(f"probe:alarm_side:{alarm_interface.split('--')[0] or 'missing'}")

    alarm_name = str(case.get("alarm_name") or "")
    for keyword, tag in (
        ("未恢复", "unrecovered"),
        ("频繁", "flapping"),
        ("error", "error_down"),
        ("CRC", "crc"),
    ):
        if keyword.lower() in alarm_name.lower():
            tokens.add(f"probe:alarm_text:{tag}")

    # 两端对比：同一物理量在 L1 与 L2 上谁更差。这是三分类里唯一能把
    # 「设备侧」定位到具体一端的方向性信息，单侧 token 表达不了。
    for metric, worse_is_low in (("rxpower", True), ("media_snr", True), ("txpower", True)):
        l1 = _healthy(_lane_values(case, metric, "L1"))
        l2 = _healthy(_lane_values(case, metric, "L2"))
        if not l1 or not l2:
            continue
        m1 = sum(l1) / len(l1)
        m2 = sum(l2) / len(l2)
        delta = m1 - m2
        threshold = 0.5 if metric != "media_snr" else 1.0
        if abs(delta) < threshold:
            tokens.add(f"probe:{metric}_side_gap:balanced")
        elif (delta < 0) == worse_is_low:
            tokens.add(f"probe:{metric}_side_gap:L1_worse")
        else:
            tokens.add(f"probe:{metric}_side_gap:L2_worse")

    # 光学正常但电口失效：光侧读数完整且在正常带，serdes 却触底。
    # 这种组合在物理上只能落在模块与主机芯片之间，不可能是光纤。
    for side in ("L1", "L2"):
        serdes = _lane_values(case, "serdes_snr", side)
        media = _healthy(_lane_values(case, "media_snr", side))
        rx = _healthy(_lane_values(case, "rxpower", side))
        if not serdes or not media or not rx:
            continue
        serdes_dead = all(value <= 1.0 for value in serdes)
        optics_ok = min(media) >= 20.0 and min(rx) > -12.5
        if serdes_dead and optics_ok:
            tokens.add(f"probe:electrical_only_fault:{side}")

    # 单 lane 离群 vs 全 lane 一致：区分通道级与端口级。
    for metric in ("rxpower", "media_snr"):
        for side in ("L1", "L2"):
            values = _healthy(_lane_values(case, metric, side))
            if len(values) < 3:
                continue
            spread = max(values) - min(values)
            tokens.add(
                f"probe:{metric}_spread:{side}:"
                + ("wide" if spread > (1.5 if metric == "rxpower" else 2.0) else "tight")
            )

    # 状态位组合：RxLOS 在哪一端，是判断「谁收不到光」的硬证据。
    for key in ("RxLOS", "TxLOS", "RxLOL", "TxLOL"):
        block = case.get(key)
        if not isinstance(block, Mapping):
            continue
        hit = tuple(
            side for side in ("L1", "L2") if str(block.get(side, "")).strip().lower() not in ("normal", "", "none")
        )
        if hit:
            tokens.add(f"probe:{key}:{'+'.join(hit)}")

    # 哪一端的 lane 更不齐。极差不受标定偏移影响（偏移是共模的），
    # 所以两端极差的相对大小仍是合法的跨端量，是 C22 加上对侧对照后的版本：
    # 只有一端的 lane 不齐，才能说明不齐来自对端发送阵列而不是共同的链路条件。
    for metric, floor in (("rxpower", 0.4), ("media_snr", 0.8)):
        spreads: Dict[str, float] = {}
        for side in ("L1", "L2"):
            values = _healthy(_lane_values(case, metric, side))
            if len(values) >= 3:
                spreads[side] = max(values) - min(values)
        if len(spreads) == 2:
            delta = spreads["L1"] - spreads["L2"]
            if abs(delta) < floor:
                tag = "similar"
            else:
                tag = "L1_more_dispersed" if delta > 0 else "L2_more_dispersed"
            tokens.add(f"probe:{metric}_spread_asymmetry:{tag}")

    # 最差 lane 的编号是否两端一致。这是 C9（双向对称指向共享部分）在通道粒度上的细化：
    # 若两端最差的是同一号 lane，问题落在该 lane 共享的部分（纤芯 / 芯位）；
    # 若各自最差的 lane 不同，更像两端各自的通道器件。它只用序信息，
    # 因此绕开了 C12 禁止的绝对电平相减。注意 C12 也提示两端 lane 编号可能并不对应，
    # 所以这个探针同时是对「编号是否真的对应」的一次检验。
    for metric in ("rxpower", "media_snr"):
        worst: Dict[str, Any] = {}
        for side in ("L1", "L2"):
            block = case.get(metric)
            values = block.get(side) if isinstance(block, Mapping) else None
            if not isinstance(values, Mapping):
                continue
            healthy = {
                key: float(value)
                for key, value in values.items()
                if _is_number(value) and float(value) > DOWN_SENTINEL
            }
            if len(healthy) >= 2:
                worst[side] = min(healthy, key=lambda key: healthy[key])
        if len(worst) == 2:
            tokens.add(
                f"probe:{metric}_worst_lane:"
                + ("aligned" if worst["L1"] == worst["L2"] else "misaligned")
            )

    # 同侧 rxpower 与 media_snr 的最差 lane 是否同一条。收光最弱的 lane 同时信噪比最差，
    # 说明该通道的劣化来自功率不足；两者指向不同 lane 则更像与功率无关的质量损伤。
    for side in ("L1", "L2"):
        worst_by_metric: Dict[str, Any] = {}
        for metric in ("rxpower", "media_snr"):
            block = case.get(metric)
            values = block.get(side) if isinstance(block, Mapping) else None
            if not isinstance(values, Mapping):
                continue
            healthy = {
                key: float(value)
                for key, value in values.items()
                if _is_number(value) and float(value) > DOWN_SENTINEL
            }
            if len(healthy) >= 2:
                worst_by_metric[metric] = min(healthy, key=lambda key: healthy[key])
        if len(worst_by_metric) == 2:
            tokens.add(
                f"probe:{side}:rx_snr_worst_lane:"
                + (
                    "same"
                    if worst_by_metric["rxpower"] == worst_by_metric["media_snr"]
                    else "different"
                )
            )

    lanes = case.get("Lane number")
    if isinstance(lanes, Mapping):
        l1 = lanes.get("L1")
        l2 = lanes.get("L2")
        if l1 and l2:
            tokens.add(f"probe:lane_width:{'equal' if l1 == l2 else 'unequal'}")

    return tokens


# --------------------------------------------------------------------------------------
# 关联统计
# --------------------------------------------------------------------------------------


def evaluate_rule(
    hit_labels: Sequence[str],
    prior: Mapping[str, float],
    *,
    test_hits: int,
    test_total: int,
    train_total: int,
) -> Optional[Dict[str, Any]]:
    if not hit_labels:
        return None
    counts = Counter(hit_labels)
    total = len(hit_labels)
    best = max(ROOT_CAUSES, key=lambda label: (counts.get(label, 0), -ROOT_CAUSES.index(label)))
    correct = counts.get(best, 0)
    lower, upper = wilson_bounds(correct, total)
    base = prior.get(best, 0.0)
    return {
        "predicts": best,
        "train_support": total,
        "train_correct": correct,
        "train_precision": round(correct / total, 6),
        "wilson_lower_bound": lower,
        "wilson_upper_bound": upper,
        "prior_for_predicted_label": round(base, 6),
        "lift": round((correct / total) / base, 4) if base else None,
        "beats_prior_at_95": lower > base,
        "label_counts": {label: counts.get(label, 0) for label in ROOT_CAUSES},
        "train_fire_rate": round(total / train_total, 6) if train_total else 0.0,
        "test_hits": test_hits,
        "test_fire_rate": round(test_hits / test_total, 6) if test_total else 0.0,
    }


def mine(
    train_tokens: Sequence[Set[str]],
    train_labels: Sequence[str],
    test_tokens: Sequence[Set[str]],
    *,
    min_support: int,
    max_pairs: int,
) -> Dict[str, Any]:
    prior = {
        label: count / len(train_labels)
        for label, count in Counter(train_labels).items()
    }
    prior = {label: prior.get(label, 0.0) for label in ROOT_CAUSES}

    train_index: Dict[str, List[int]] = defaultdict(list)
    for position, tokens in enumerate(train_tokens):
        for token in tokens:
            train_index[token].append(position)
    test_index: Dict[str, int] = Counter()
    for tokens in test_tokens:
        for token in tokens:
            test_index[token] += 1

    singles: List[Dict[str, Any]] = []
    for token, positions in train_index.items():
        if len(positions) < min_support:
            continue
        stats = evaluate_rule(
            [train_labels[position] for position in positions],
            prior,
            test_hits=test_index.get(token, 0),
            test_total=len(test_tokens),
            train_total=len(train_tokens),
        )
        if stats:
            singles.append({"rule": token, "kind": "token", **stats})

    frequent = sorted(
        (token for token, positions in train_index.items() if len(positions) >= min_support),
        key=lambda token: -len(train_index[token]),
    )[:max_pairs]
    pairs: List[Dict[str, Any]] = []
    train_sets = [set(item) for item in train_tokens]
    test_sets = [set(item) for item in test_tokens]
    for left, right in combinations(frequent, 2):
        positions = [
            position
            for position in train_index[left]
            if right in train_sets[position]
        ]
        if len(positions) < min_support:
            continue
        test_hits = sum(1 for tokens in test_sets if left in tokens and right in tokens)
        stats = evaluate_rule(
            [train_labels[position] for position in positions],
            prior,
            test_hits=test_hits,
            test_total=len(test_sets),
            train_total=len(train_sets),
        )
        if stats:
            pairs.append({"rule": f"{left} AND {right}", "kind": "pair", **stats})

    def rank(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(items, key=lambda item: (-item["wilson_lower_bound"], -item["train_support"]))

    return {
        "train_prior": {label: round(value, 6) for label, value in prior.items()},
        "singles": rank(singles),
        "pairs": rank(pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--feature-profile", default="v2")
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("artifacts/i1_knowledge_candidates.json"))
    args = parser.parse_args()

    train_cases = cases_by_manifest_split(args.data_dir, "train")
    test_cases = cases_by_manifest_split(args.data_dir, "test")
    train_labels = [str(case["label"]) for case in train_cases]

    dictionary = dictionary_for(args.feature_profile)
    train_packs = build_packs(train_cases, source_dataset=str(args.data_dir))
    test_packs = build_packs(test_cases, source_dataset=str(args.data_dir))
    thresholds = fit_thresholds(train_cases)
    model = fit_feature_model(train_packs, dictionary=dictionary)

    def tokens_for(packs, cases, with_probe: bool) -> List[Set[str]]:
        out: List[Set[str]] = []
        for pack, case in zip(packs, cases):
            base = set(extract_features(pack, thresholds, model, dictionary=dictionary).tokens)
            if with_probe:
                base |= probe_tokens(case)
            out.append(base)
        return out

    report: Dict[str, Any] = {
        "schema_version": "knowledge-candidates-v1",
        "data_dir": str(args.data_dir),
        "feature_profile": args.feature_profile,
        "feature_dictionary_version": dictionary.version,
        "train_size": len(train_cases),
        "test_size": len(test_cases),
        "min_support": args.min_support,
        "label_policy": (
            "训练集提供标签；测试集只提供触发次数，不参与任何 precision 计算。"
        ),
    }

    for scope, with_probe in (("dictionary_only", False), ("with_probe", True)):
        mined = mine(
            tokens_for(train_packs, train_cases, with_probe),
            train_labels,
            tokens_for(test_packs, test_cases, with_probe),
            min_support=args.min_support,
            max_pairs=args.max_pairs,
        )
        report[scope] = mined

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    prior = report["with_probe"]["train_prior"]
    print(f"train={len(train_cases)} test={len(test_cases)} prior={prior}")
    for scope in ("dictionary_only", "with_probe"):
        for kind in ("singles", "pairs"):
            rows = report[scope][kind]
            beating = [row for row in rows if row["beats_prior_at_95"]]
            print()
            print(
                f"== {scope} / {kind}: {len(rows)} 条达到最小支持数，"
                f"{len(beating)} 条 95% 下界超过其预测类别的先验 =="
            )
            print(
                f"{'predict':>7} {'n':>4} {'prec':>6} {'lb':>6} {'lift':>6} {'test_hit':>8}  rule"
            )
            for row in rows[: args.top]:
                print(
                    f"{row['predicts']:>7} {row['train_support']:>4}"
                    f" {row['train_precision']:>6.3f} {row['wilson_lower_bound']:>6.3f}"
                    f" {(row['lift'] or 0):>6.2f} {row['test_hits']:>8}  {row['rule']}"
                )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
