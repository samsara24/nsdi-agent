"""把缺陷分析里所有结论性数字算出来存成一份 JSON，供报告直接引用。

报告里每个数字都必须能追到一段可重跑的计算，否则过几周就没人能确认它是不是还成立。
这个脚本汇总的是「为什么现在错」这一类问题的量化答案：规则分格可靠性、物理盲读的
覆盖上限、fiber 策略修法的实测收益、元数据是否泄漏、以及数据管道本身的卫生问题。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_graph import EvidenceGraph, match_many
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.expert import diagnose_many
from rca_framework.features import dictionary_for, extract_features, fit_feature_model
from rca_framework.types import wilson_lower_bound
from scripts.eval_blind_physical_reader import read_case

DATASET = ROOT / "datasets/rca_v2_l2fixed"
CASE_BUNDLE = ROOT / "artifacts/report_bundle.json"
OUT = ROOT / "artifacts/defect_bundle.json"
LABELS = ("L1", "L2", "fiber")
METRICS = ("rxpower", "txpower", "media_snr", "host_snr", "serdes_snr", "bias")


def lanes(case, metric, side):
    block = (case.get(metric) or {}).get(side)
    if isinstance(block, dict):
        return [block[k] for k in sorted(block, key=lambda x: int(x) if str(x).isdigit() else 0)]
    return []


def blind_reader_stats(cases, golds) -> Dict[str, Any]:
    reads = [read_case(c) for c in cases]
    preds = [r[0] for r in reads]
    expert = [d.verdict for d in diagnose_many(build_packs(cases))]

    by_rule: Dict[str, Dict[str, Any]] = {}
    for (pred, rule, _), gold in zip(reads, golds):
        slot = by_rule.setdefault(rule, {"n": 0, "ok": 0, "answered": 0, "dist": Counter()})
        slot["n"] += 1
        slot["dist"][gold] += 1
        if pred is not None:
            slot["answered"] += 1
            slot["ok"] += int(pred == gold)
    for rule, slot in by_rule.items():
        slot["dist"] = dict(slot["dist"])
        slot["acc"] = slot["ok"] / slot["answered"] if slot["answered"] else None
        slot["wilson_lb"] = wilson_lower_bound(slot["ok"], slot["answered"]) if slot["answered"] else None
        slot["fiber_rate"] = slot["dist"].get("fiber", 0) / slot["n"]

    answered_idx = [i for i, p in enumerate(preds) if p is not None]
    blind_ok = sum(1 for i in answered_idx if preds[i] == golds[i])
    expert_on_same = sum(1 for i in answered_idx if expert[i] == golds[i])
    return {
        "n": len(cases),
        "answered": len(answered_idx),
        "coverage": len(answered_idx) / len(cases),
        "blind_ok": blind_ok,
        "blind_precision": blind_ok / len(answered_idx) if answered_idx else 0.0,
        "expert_ok_on_same_subset": expert_on_same,
        "expert_precision_on_same_subset": expert_on_same / len(answered_idx) if answered_idx else 0.0,
        "expert_full_coverage_acc": sum(1 for e, g in zip(expert, golds) if e == g) / len(golds),
        "by_rule": by_rule,
    }


def rule_cells(cases, golds) -> Dict[str, Any]:
    prior = {k: v / len(golds) for k, v in Counter(golds).items()}
    diags = diagnose_many(build_packs(cases))
    cells: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "ok": 0, "dist": Counter()})
    for gold, diag in zip(golds, diags):
        if diag.sides:
            best = diag.sides[0]
            key = f"{best.rule}|{best.side}->{best.location}"
        else:
            key = f"{diag.group}|-"
        cell = cells[key]
        cell["n"] += 1
        cell["ok"] += int(diag.verdict == gold)
        cell["dist"][gold] += 1
    out = {}
    for key, cell in cells.items():
        side = key.split("->")[-1]
        p = prior.get(side, 0.0)
        lb = wilson_lower_bound(cell["ok"], cell["n"])
        out[key] = {
            "rule": key.split("|")[0],
            "direction": key.split("|")[1],
            "n": cell["n"],
            "ok": cell["ok"],
            "acc": cell["ok"] / cell["n"],
            "wilson_lb": lb,
            "verdict_prior": p,
            "beats_prior": bool(p and lb > p),
            "dist": dict(cell["dist"]),
            "fiber_rate": cell["dist"]["fiber"] / cell["n"],
        }
    return {"prior": prior, "cells": out}


def fiber_policy(train, test) -> Dict[str, Any]:
    """检索通道能找到 fiber，专家规则找不到。量化两种利用方式的实测收益。"""
    train_labels, test_labels = labels_of(train), labels_of(test)
    dic = dictionary_for("v3")
    th = fit_thresholds(train)
    train_packs = build_packs(train, source_dataset=str(DATASET))
    test_packs = build_packs(test, source_dataset=str(DATASET))
    fm = fit_feature_model(train_packs, dictionary=dic)
    train_feats = [extract_features(p, th, fm, dictionary=dic) for p in train_packs]
    test_feats = [extract_features(p, th, fm, dictionary=dic) for p in test_packs]
    graph = EvidenceGraph.build(train_feats, train_labels, feature_model=fm, dictionary=dic,
                                source_dataset=str(DATASET))
    matches = match_many(graph, test_feats, top_k=5)
    top1 = [(m.candidates[0].label, m.candidates[0].similarity) if m.candidates else (None, 0.0)
            for m in matches]
    expert = [d.verdict for d in diagnose_many(test_packs)]
    base_ok = sum(1 for e, g in zip(expert, test_labels) if e == g)

    override, abstain = [], []
    for t in (0.6, 0.7, 0.8, 0.85, 0.9, 1.0):
        pred = ["fiber" if (c == "fiber" and s >= t) else e for e, (c, s) in zip(expert, top1)]
        ok = sum(1 for p, g in zip(pred, test_labels) if p == g)
        fib_hit = sum(1 for p, g in zip(pred, test_labels) if p == "fiber" and g == "fiber")
        fib_n = sum(1 for p in pred if p == "fiber")
        override.append({"t": t, "acc": ok / len(test_labels), "delta": ok - base_ok,
                         "fiber_recall": fib_hit / 8, "fiber_precision": fib_hit / fib_n if fib_n else 0.0,
                         "triggered": sum(1 for e, p in zip(expert, pred) if e != p)})
        keep = [(e, g) for e, g, (c, s) in zip(expert, test_labels, top1)
                if not (c == "fiber" and s >= t)]
        k_ok = sum(1 for e, g in keep if e == g)
        abstain.append({"t": t, "coverage": len(keep) / len(test_labels),
                        "precision": k_ok / len(keep) if keep else 0.0, "answered": len(keep)})

    fiber_detail = [
        {"id": c["case_id"], "expert": e, "top1": t[0], "sim": round(t[1], 3)}
        for c, g, e, t in zip(test, test_labels, expert, top1) if g == "fiber"
    ]
    return {"baseline_acc": base_ok / len(test_labels), "override": override,
            "abstain": abstain, "fiber_cases": fiber_detail}


def selective_policy(train, test) -> Dict[str, Any]:
    """按规则格子的可靠性做选择性预测，扫出覆盖-精度曲线。

    这里要小心区分两个不同的判据，早期版本把它们混在一起会得出反向的结论：

      informative  格子的 Wilson 下界是否高于**它所判类别的先验**。这衡量的是
                   「这条规则有没有提供信息」。但它对判少数类的格子门槛很低
                   （判 L1 只需超过 30.2%），所以不能拿来做拒答策略——
                   照它拒答会把 74% 的 L2 格子丢掉、留下 51% 的 L1 格子。
      reliable     格子准确率的 Wilson 下界是否高于给定目标精度。这才是
                   「该不该给结论」的判据，与所判类别无关。

    训练集用来估每个格子的可靠性，test 上只做查表，不看 test 标签定门槛。
    """
    train_labels = labels_of(train)
    train_cells = rule_cells(train, train_labels)["cells"]
    test_labels = labels_of(test)
    diags = diagnose_many(build_packs(test))

    def key_of(diag):
        if diag.sides:
            b = diag.sides[0]
            return f"{b.rule}|{b.side}->{b.location}"
        return f"{diag.group}|-"

    base = sum(1 for d, g in zip(diags, test_labels) if d.verdict == g)
    curve = []
    for target in (0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        keep = []
        for gold, diag in zip(test_labels, diags):
            cell = train_cells.get(key_of(diag))
            # 训练集里没见过这个格子 -> 无可靠性估计 -> 拒答
            if cell is None or cell["wilson_lb"] < target:
                continue
            keep.append((diag.verdict, gold))
        ok = sum(1 for p, g in keep if p == g)
        curve.append({
            "target_wilson_lb": target,
            "answered": len(keep),
            "coverage": len(keep) / len(test_labels),
            "precision": ok / len(keep) if keep else 0.0,
            "correct": ok,
            "recovered_share": ok / len(test_labels),
        })

    informative = {k: v["beats_prior"] for k, v in train_cells.items()}
    return {
        "baseline_full_coverage": base / len(test_labels),
        "curve": curve,
        "train_cell_informative": informative,
        "note": "门槛在 train 上估、test 上查表；informative 与 reliable 是两个不同判据",
    }


def data_hygiene(cases) -> Dict[str, Any]:
    unparsed: Dict[str, List[str]] = defaultdict(list)
    missing = {}
    for metric in METRICS:
        for side in ("L1", "L2"):
            n_missing = 0
            for c in cases:
                block = (c.get(metric) or {}).get(side)
                if isinstance(block, str):
                    unparsed[metric].append(c["case_id"])
                if not lanes(c, metric, side):
                    n_missing += 1
            missing[f"{metric}.{side}"] = {"missing": n_missing, "rate": n_missing / len(cases)}

    lane_mismatch = 0
    serdes_vs_optical: Counter = Counter()
    for c in cases:
        if len(lanes(c, "rxpower", "L1")) != len(lanes(c, "txpower", "L2")):
            lane_mismatch += 1
        for side in ("L1", "L2"):
            n_opt, n_ser = len(lanes(c, "media_snr", side)), len(lanes(c, "serdes_snr", side))
            if n_opt and n_ser:
                serdes_vs_optical[f"{n_opt}光lane/{n_ser}serdes"] += 1

    flat_tx = sum(
        1 for c in cases for side in ("L1", "L2")
        if len({v for v in lanes(c, "txpower", side) if v is not None}) == 1
        and len([v for v in lanes(c, "txpower", side) if v is not None]) >= 2
    )
    return {
        "unparsed_blocks": {k: sorted(set(v)) for k, v in unparsed.items()},
        "missing": missing,
        "cross_side_lane_mismatch": {"n": lane_mismatch, "rate": lane_mismatch / len(cases)},
        "serdes_vs_optical_lanes": dict(serdes_vs_optical),
        "flat_txpower_sides": flat_tx,
        "total": len(cases),
    }


def metadata_leakage(cases, golds) -> Dict[str, Any]:
    out = {}
    for field in ("alarm_name", "alarm_ip_interface", "Lane number", "region"):
        groups: Dict[str, Counter] = defaultdict(Counter)
        for c, g in zip(cases, golds):
            v = c.get(field)
            if field == "alarm_ip_interface":
                key = "存在" if v else "缺失"
            else:
                key = str(v)
            groups[key][g] += 1
        rows = []
        for key, cnt in groups.items():
            n = sum(cnt.values())
            if n < 5:
                continue
            rows.append({"value": key, "n": n,
                         **{f"{lab}_rate": cnt[lab] / n for lab in LABELS},
                         "dist": dict(cnt)})
        out[field] = sorted(rows, key=lambda r: -r["n"])
    return out


def llm_alignment(bundle) -> Dict[str, Any]:
    """LLM 定界结论与最终结论的一致性。

    比「LLM 准确率」更能说明问题的是这个：在系统<b>判对</b>的 case 上，
    LLM 有多少次给出了不同的答案。如果它在正确答案旁边频繁走偏，
    那么把它接进主干只会拉低成绩，无论它的整体准确率看起来是多少。
    """
    cases = bundle["cases"]
    groups = {
        "correct": [c for c in cases if c["ok"]],
        "wrong": [c for c in cases if c["pred"] is not None and not c["ok"]],
        "abstain": [c for c in cases if c["pred"] is None],
    }
    out = {}
    for name, rows in groups.items():
        with_trace = [c for c in rows if c["llm"]]
        parsed = [c for c in with_trace if c["llm"]["verdict"]]
        agree = [c for c in parsed if c["llm"]["verdict"] == c["pred"]]
        gold_hit = [c for c in parsed if c["llm"]["verdict"] == c["gold"]]
        out[name] = {
            "n": len(rows),
            "with_trace": len(with_trace),
            "parsed": len(parsed),
            "unparsed_or_abstain": len(with_trace) - len(parsed),
            "agree_with_final": len(agree),
            "agree_rate": len(agree) / len(parsed) if parsed else 0.0,
            "matches_gold": len(gold_hit),
            "gold_rate": len(gold_hit) / len(parsed) if parsed else 0.0,
        }
    violations: Counter = Counter()
    fatal_cases = 0
    for c in cases:
        if not c["llm"]:
            continue
        has_fatal = False
        for a in c["llm"]["attempts"]:
            for v in a["violations"]:
                violations[v["kind"]] += 1
                if v["severity"] == "fatal":
                    has_fatal = True
        fatal_cases += int(has_fatal)
    out["violation_kinds"] = dict(violations.most_common())
    out["cases_with_fatal"] = fatal_cases
    return out


def final_system(bundle) -> Dict[str, Any]:
    cases = bundle["cases"]
    conf: Dict[str, Counter] = defaultdict(Counter)
    for c in cases:
        conf[c["gold"]][c["pred"] if c["pred"] else "abstain"] += 1
    per = {}
    for lab in LABELS:
        support = sum(conf[lab].values())
        hit = conf[lab][lab]
        pred_n = sum(conf[g][lab] for g in LABELS)
        per[lab] = {"support": support, "hit": hit, "pred_n": pred_n,
                    "recall": hit / support if support else 0.0,
                    "precision": hit / pred_n if pred_n else 0.0,
                    "row": dict(conf[lab])}
    answered = sum(1 for c in cases if c["pred"] is not None)
    ok = sum(1 for c in cases if c["ok"])
    return {"n": len(cases), "answered": answered, "coverage": answered / len(cases),
            "correct": ok, "precision": ok / answered if answered else 0.0,
            "overall": ok / len(cases), "per_class": per,
            "confusion": {g: dict(conf[g]) for g in LABELS}}


def main() -> int:
    train = cases_by_manifest_split(DATASET, "train")
    test = cases_by_manifest_split(DATASET, "test")
    every = train + test
    golds_all, golds_test = labels_of(every), labels_of(test)
    bundle = json.loads(CASE_BUNDLE.read_text(encoding="utf-8"))

    payload = {
        "blind_reader": {
            "test": blind_reader_stats(test, golds_test),
            "all": blind_reader_stats(every, golds_all),
        },
        "rule_cells_all": rule_cells(every, golds_all),
        "fiber_policy": fiber_policy(train, test),
        "selective_policy": selective_policy(train, test),
        "data_hygiene": data_hygiene(every),
        "metadata": metadata_leakage(every, golds_all),
        "final_system": final_system(bundle),
        "llm_alignment": llm_alignment(bundle),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} KB)")

    b = payload["blind_reader"]["all"]
    print(f"物理盲读(全库) 覆盖={b['coverage']:.1%} 精度={b['blind_precision']:.1%} "
          f"同子集专家={b['expert_precision_on_same_subset']:.1%} 专家全答={b['expert_full_coverage_acc']:.1%}")
    fs = payload["final_system"]
    print(f"当前系统 覆盖={fs['coverage']:.2%} 给结论精度={fs['precision']:.2%} 全集={fs['overall']:.2%}")
    for lab in LABELS:
        p = fs["per_class"][lab]
        print(f"  {lab:5s} 召回={p['recall']:6.2%} ({p['hit']}/{p['support']}) 精度={p['precision']:6.2%}")
    sp = payload["selective_policy"]
    print(f"选择性预测曲线（基线全答 {sp['baseline_full_coverage']:.2%}）:")
    for row in sp["curve"]:
        print(f"  门槛下界>={row['target_wilson_lb']:.2f} 覆盖={row['coverage']:6.1%} "
              f"({row['answered']:3d}) 精度={row['precision']:6.2%}")
    la = payload["llm_alignment"]
    print("LLM 定界与最终结论的一致性：")
    for k in ("correct", "wrong", "abstain"):
        g = la[k]
        print(f"  {k:8s} n={g['n']:3d} 有trace={g['with_trace']:3d} 可解析={g['parsed']:3d} "
              f"未解析/弃答={g['unparsed_or_abstain']:3d} 与最终一致={g['agree_with_final']:3d}"
              f"({g['agree_rate']:.1%}) 命中真值={g['matches_gold']:3d}({g['gold_rate']:.1%})")
    print(f"  出现 fatal 违规的 case 数={la['cases_with_fatal']}  违规类型={la['violation_kinds']}")
    dh = payload["data_hygiene"]
    print(f"数据卫生 未解析块={ {k: len(v) for k, v in dh['unparsed_blocks'].items()} } "
          f"跨端lane数不一致={dh['cross_side_lane_mismatch']['n']}/268 "
          f"tx四条完全相同的侧={dh['flat_txpower_sides']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
