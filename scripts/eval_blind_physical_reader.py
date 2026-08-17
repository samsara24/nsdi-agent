"""把「不看标签、只按物理读原始遥测」的推导过程写成规则，并在全库上打分。

写这个脚本的目的不是造一个更好的定界器，而是**防止自欺**。逐 case 分析里很容易
写出一段看起来很有说服力的物理推理，然后发现它恰好和标签一致——但那可能只是
后见之明。只有把同一套推理无差别地施加到全部 268 条上，才能知道它是不是真规律。

规则集刻意只用「同一时刻能从原始 lane 读数里看出来的东西」，不用任何标签统计量，
也不用专家阈值表，因为要检验的正是「一个更强的推理者能否从这份遥测里做得更好」。

判据的物理依据：
  P1 本端收不到光 + 对端同 lane 也不发光 -> 光根本没出发 -> 对端发端故障。
  P2 收到的光「变少且同步变脏」（功率与 SNR 一起掉）-> 上游衰减 -> 对端或链路。
  P3 本端收不到光 + 对端在正常发光 -> 光出发了却没到 -> fiber / 本端收端 / 对端
     光口，三者在这份遥测里无法区分，只能请求补采（这是 P3 存在的全部意义）。
  P4 只有 serdes_snr 异常而光层全正常 -> 故障在电通道，不在光链路两端，
     当前标签体系（L1/L2/fiber）里没有对应类别 -> 不判。
"""

from __future__ import annotations

import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.expert import diagnose_many
from rca_framework.types import wilson_lower_bound

DATASET = ROOT / "datasets/rca_v2_l2fixed"
PEER = {"L1": "L2", "L2": "L1"}
DARK = -39.0          # 断光哨兵：数据里写 -40，留 1 dB 余量
SNR_DEAD = 1.0        # media_snr 归零同样是哨兵，不是真实测量
SNR_DROP = 2.0        # 相对同侧中位数的 SNR 下降门限（dB）
RX_DROP = 1.0         # 相对同侧中位数的功率下降门限（dB）


def lanes(case: Dict[str, Any], metric: str, side: str) -> List[Optional[float]]:
    block = (case.get(metric) or {}).get(side)
    if isinstance(block, dict):
        return [block[k] for k in sorted(block, key=lambda x: int(x) if str(x).isdigit() else 0)]
    return []


def dark_lanes(case, side) -> List[int]:
    return [i for i, v in enumerate(lanes(case, "rxpower", side)) if v is not None and v <= DARK]


def median_of(values: Sequence[Optional[float]], floor: float) -> Optional[float]:
    live = [v for v in values if v is not None and v > floor]
    return st.median(live) if len(live) >= 3 else None


def read_case(case: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
    """返回 (判定, 依据编号, 人话解释)。判定为 None 表示这份遥测判不了。"""
    for side in ("L1", "L2"):
        peer = PEER[side]
        rx = lanes(case, "rxpower", side)
        peer_tx = lanes(case, "txpower", peer)
        dark = dark_lanes(case, side)
        if not dark:
            continue
        # 只有两端 lane 数一致时，lane 索引才能跨端对齐
        aligned = len(rx) == len(peer_tx) and len(peer_tx) > 0
        if aligned:
            peer_dark = [i for i in dark if peer_tx[i] is not None and peer_tx[i] <= DARK]
            if peer_dark:
                return peer, "P1", (
                    f"{side} 侧 lane {peer_dark} 收不到光，而对端 {peer} 在同一条 lane 上也没有发光输出，"
                    f"光根本没有离开 {peer} 的发端，故障在 {peer}"
                )
            peer_live = [i for i in dark if peer_tx[i] is not None and peer_tx[i] > DARK]
            if peer_live:
                return None, "P3", (
                    f"{side} 侧 lane {peer_live} 收不到光，但对端 {peer} 在这条 lane 上发光功率正常"
                    f"（{peer_tx[peer_live[0]]:.2f} dBm）。光出发了却没到达，可能是光缆、"
                    f"{side} 的收端、或 {peer} 的光口——这三者在当前遥测里无法区分，需要 OTDR 或双向同步快照"
                )
        return None, "P3", (
            f"{side} 侧 lane {dark} 收不到光，但两端 lane 数不一致（{len(rx)} vs {len(peer_tx)}），"
            f"无法把断光 lane 对齐到对端发端，判不了"
        )

    # 没有断光 lane，再看「光变少」
    for side in ("L1", "L2"):
        snr = lanes(case, "media_snr", side)
        rx = lanes(case, "rxpower", side)
        n = min(len(snr), len(rx))
        msnr, mrx = median_of(snr[:n], SNR_DEAD), median_of(rx[:n], DARK)
        if msnr is None or mrx is None:
            continue
        for i in range(n):
            s, r = snr[i], rx[i]
            if s is None or r is None or s <= SNR_DEAD or r <= DARK:
                continue
            if msnr - s >= SNR_DROP and mrx - r >= RX_DROP:
                return PEER[side], "P2", (
                    f"{side} 侧 lane {i} 的光功率比同侧中位数低 {mrx-r:.2f} dB，信噪比同步低 {msnr-s:.2f} dB，"
                    f"功率与质量一起下降说明是上游衰减而非本端解调问题，指向对端 {PEER[side]}"
                )

    serdes_bad = any(
        v is not None and v <= SNR_DEAD
        for side in ("L1", "L2")
        for v in lanes(case, "serdes_snr", side)
    )
    if serdes_bad:
        return None, "P4", (
            "光层（功率、media_snr）两端均正常，只有 serdes 电通道某条 lane 归零。"
            "故障在电域（主机接口 / gearbox），不属于 L1/L2/fiber 三分类的任何一类，不判"
        )
    return None, "P5", "两端光层与电层都没有可判读的异常，这份快照不含定位信息"


def score(name: str, preds: Sequence[Optional[str]], golds: Sequence[str]) -> Dict[str, Any]:
    answered = [(p, g) for p, g in zip(preds, golds) if p is not None]
    ok = sum(1 for p, g in answered if p == g)
    return {
        "name": name,
        "coverage": len(answered) / len(golds),
        "answered": len(answered),
        "precision": ok / len(answered) if answered else 0.0,
        "ok": ok,
        "overall": ok / len(golds),
    }


def main() -> int:
    train = cases_by_manifest_split(DATASET, "train")
    test = cases_by_manifest_split(DATASET, "test")

    for split_name, cases in (("test(107)", test), ("全库(268)", train + test)):
        golds = labels_of(cases)
        reads = [read_case(c) for c in cases]
        preds = [r[0] for r in reads]
        expert = [d.verdict for d in diagnose_many(build_packs(cases))]

        print(f"===== {split_name} =====")
        blind = score("物理盲读", preds, golds)
        exp_all = score("专家规则(全答)", expert, golds)
        # 在盲读肯回答的子集上比，才是同口径对比
        mask = [p is not None for p in preds]
        exp_sub = score(
            "专家规则(同覆盖子集)",
            [e if m else None for e, m in zip(expert, mask)],
            golds,
        )
        for s in (blind, exp_sub, exp_all):
            print(f"  {s['name']:20s} 覆盖={s['coverage']:6.1%} ({s['answered']:3d}) "
                  f"给结论精度={s['precision']:6.2%} 全集正确率={s['overall']:6.2%}")

        print("  各判据的触发量与精度：")
        by_rule: Dict[str, Dict[str, Any]] = {}
        for (pred, rule, _), gold in zip(reads, golds):
            slot = by_rule.setdefault(rule, {"n": 0, "ok": 0, "answered": 0, "dist": Counter()})
            slot["n"] += 1
            slot["dist"][gold] += 1
            if pred is not None:
                slot["answered"] += 1
                slot["ok"] += int(pred == gold)
        for rule in sorted(by_rule):
            s = by_rule[rule]
            acc = s["ok"] / s["answered"] if s["answered"] else 0.0
            lb = wilson_lower_bound(s["ok"], s["answered"]) if s["answered"] else 0.0
            tag = "不判" if not s["answered"] else f"精度={acc:6.1%} 下界={lb:5.1%}"
            print(f"    {rule} 触发={s['n']:3d} {tag:28s} 真值分布={dict(s['dist'])}")
        print()

    print("P3（判不了，需要补采）子集里三类的构成，决定了补采能拿回多少：")
    golds = labels_of(train + test)
    reads = [read_case(c) for c in train + test]
    p3 = [g for (p, r, _), g in zip(reads, golds) if r == "P3"]
    print(f"  n={len(p3)} 分布={dict(Counter(p3))} fiber 占比={Counter(p3)['fiber']/len(p3):.1%}"
          f"（全库 fiber 先验 7.5%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
