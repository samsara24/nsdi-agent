"""逐规则核对专家决策树的方向映射与 fiber 兜底是否成立。

用法：
  python scripts/analyze_expert_rule_defects.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from validate_expert_rules import (
    DIRECTION_VARIANTS,
    LABELS,
    OPPOSITE,
    diagnose_side,
    load_cases,
    port_status,
)


def main() -> None:
    cases = load_cases(Path("organized_data"))
    direction = DIRECTION_VARIANTS["code"]

    # 1. 逐规则方向核对：命中规则的端 + 真值标签分布
    bucket: dict[str, Counter] = defaultdict(Counter)
    for case in cases:
        if (port_status(case, "local"), port_status(case, "remote")) != (1, 1):
            continue
        for side in ("local", "remote"):
            res = diagnose_side(case, side, direction)
            if res is None:
                continue
            bucket[f"{res['rule']}@{side}"][case["label"]] += 1

    print("=" * 96)
    print("单端规则命中桶的真值分布（该端命中此规则时，真实根因是谁）")
    print("=" * 96)
    print(f"{'规则@命中端':<34}{'n':>5}{'local':>8}{'remote':>8}{'fiber':>7}"
          f"{'文档方向':>10}{'该方向命中率':>13}{'反向命中率':>12}")
    for key in sorted(bucket, key=lambda k: -sum(bucket[k].values())):
        dist = bucket[key]
        rule, side = key.split("@")
        n = sum(dist.values())
        rule_key = rule.replace("single:", "")
        doc_loc = side if direction[rule_key] == "same" else OPPOSITE[side]
        rev_loc = OPPOSITE[doc_loc]
        print(f"{key:<34}{n:>5}{dist['local']:>8}{dist['remote']:>8}{dist['fiber']:>7}"
              f"{doc_loc:>10}{dist[doc_loc] / n:>12.1%}{dist[rev_loc] / n:>12.1%}")

    # 2. fiber 兜底条件为何不触发
    print("\n" + "=" * 96)
    print("双端结果分布（fiber 需要：两端都有结果 + priority 相同 + 定界不同）")
    print("=" * 96)
    stats = Counter()
    equal_prio_same_loc: Counter = Counter()
    for case in cases:
        if (port_status(case, "local"), port_status(case, "remote")) != (1, 1):
            stats["被端口状态门拦截"] += 1
            continue
        rl = diagnose_side(case, "local", direction)
        rr = diagnose_side(case, "remote", direction)
        got = [r for r in (rl, rr) if r is not None]
        if len(got) == 0:
            stats["两端均无异常（兜底 local）"] += 1
        elif len(got) == 1:
            stats["仅单端有结果"] += 1
        else:
            same_prio = rl["priority"] == rr["priority"]
            same_loc = rl["location"] == rr["location"]
            if same_prio and not same_loc:
                stats["两端同优先级且定界不同 -> fiber"] += 1
            elif same_prio and same_loc:
                stats["两端同优先级但定界相同 -> 不判 fiber"] += 1
                equal_prio_same_loc[case["label"]] += 1
            else:
                stats["两端优先级不同 -> 取高优先级"] += 1
    for k, v in stats.most_common():
        print(f"  {k:<44}{v:>5}")
    if equal_prio_same_loc:
        print(f"\n  「同优先级但定界相同」这批的真值分布：{dict(equal_prio_same_loc)}")

    # 3. 18 例 fiber 真值分别被判成什么、由哪条规则决定
    print("\n" + "=" * 96)
    print("18 例 fiber 真值的实际裁决来源")
    print("=" * 96)
    preds = json.loads(Path("artifacts/expert_rule_validation/predictions_code.json").read_text())
    fiber_rows = [r for r in preds if r["label"] == "fiber"]
    src = Counter(r["source"] for r in fiber_rows)
    for s, n in src.most_common():
        print(f"  {s:<34}{n:>4}")

    # 4. 兜底 local 的 48 例：其真值分布 == 先验偏置检验
    fallback = [r for r in preds if r["source"] == "no_anomaly_fallback"]
    print(f"\n无异常兜底 {len(fallback)} 例真值分布：{dict(Counter(r['label'] for r in fallback))}")


if __name__ == "__main__":
    main()
