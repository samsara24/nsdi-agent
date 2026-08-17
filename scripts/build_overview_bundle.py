"""汇总总览报告需要的全部框架事实与对照实验数字。

总览页要回答两件事：这套系统由什么构成，它到底做到了什么水平。前者从代码里把约束库、
SOP 树、专家阈值表这些**当前实际生效的**定义抽出来（不照抄文档，文档会滞后）；后者把
所有对照口径放在同一张表上，否则 76.6% 这个数字没有意义——多数类基线已经有 62.3%。

装配顺序与 `scripts/evaluate_routing.py` 完全一致（同一份 manifest 划分、同一个特征
profile、阈值与 IDF 只在 train 上拟合），保证这里报出的基线和主实验可比。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rca_framework.anomaly import fit_thresholds
from rca_framework.constraints.library import CONSTRAINT_LIBRARY
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_graph import EvidenceGraph, match_many
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.expert import (
    ANOMALY_LEVEL,
    ANOMALY_ORDER,
    EXPERT_METRICS,
    EXPERT_THRESHOLDS,
    MULTI_METRIC_PRIORITY,
    MULTI_METRIC_REQUIRES,
    SINGLE_METRIC_BASE,
    SINGLE_METRIC_DIRECTION,
    diagnose_many,
)
from rca_framework.features import dictionary_for, extract_features, fit_feature_model
from rca_framework.sop import learn_sop
from rca_framework.types import wilson_lower_bound

DATASET = ROOT / "datasets/rca_v2_l2fixed"
OUT = ROOT / "artifacts/overview_bundle.json"
LABELS = ("L1", "L2", "fiber")
PRIOR = {"L2": 0.6231, "L1": 0.3022, "fiber": 0.0746}


def constraint_rows() -> List[Dict[str, Any]]:
    rows = []
    for c in CONSTRAINT_LIBRARY.constraints:
        rows.append(
            {
                "id": c.constraint_id,
                "category": c.category,
                "kind": c.kind,
                "title": c.title,
                "statement": c.physical_statement,
                "formal": c.formal_expression,
                "params": [list(p) for p in c.parameters],
                "provenance": c.provenance,
                "evidence": c.measured_evidence,
                "use": c.diagnostic_use,
                "prefixes": list(c.applies_to_token_prefixes),
                "effects": list(c.allowed_effects),
                "targets": list(c.allowed_targets),
            }
        )
    return rows


def per_class(preds: Sequence[Any], golds: Sequence[str]) -> Dict[str, Any]:
    conf: Dict[str, Counter] = defaultdict(Counter)
    for p, g in zip(preds, golds):
        conf[g][p if p is not None else "abstain"] += 1
    out = {}
    for label in LABELS:
        support = sum(conf[label].values())
        hit = conf[label][label]
        pred_n = sum(conf[g][label] for g in LABELS)
        out[label] = {
            "support": support,
            "hit": hit,
            "pred_n": pred_n,
            "recall": hit / support if support else 0.0,
            "precision": hit / pred_n if pred_n else 0.0,
            "row": dict(conf[label]),
        }
    return out


def cell_of(diag: Any) -> str:
    if not diag.sides:
        return f"{diag.group}|-"
    best = diag.sides[0]
    return f"{best.rule}|{best.side}->{best.location}"


def reliability(train_cases, test_cases) -> Dict[str, Any]:
    """规则可靠性只能在全库上统计：test 只有 107 条，单格样本量不足以给出下界。"""
    every = list(train_cases) + list(test_cases)
    labels = labels_of(every)
    diags = diagnose_many(build_packs(every))
    cells: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "ok": 0, "dist": Counter()})
    for label, diag in zip(labels, diags):
        cell = cells[cell_of(diag)]
        cell["n"] += 1
        cell["ok"] += int(diag.verdict == label)
        cell["dist"][label] += 1
    table = {}
    for key, cell in cells.items():
        side = key.split("->")[-1]
        prior = PRIOR.get(side, 0.0)
        lower = wilson_lower_bound(cell["ok"], cell["n"])
        table[key] = {
            "n": cell["n"],
            "ok": cell["ok"],
            "acc": cell["ok"] / cell["n"],
            "wilson_lb": lower,
            "verdict_prior": prior,
            "beats_prior": bool(prior and lower > prior),
            "dist": dict(cell["dist"]),
            "fiber_rate": cell["dist"]["fiber"] / cell["n"],
        }
    return table


def main() -> int:
    train_cases = cases_by_manifest_split(DATASET, "train")
    test_cases = cases_by_manifest_split(DATASET, "test")
    train_labels, test_labels = labels_of(train_cases), labels_of(test_cases)

    dictionary = dictionary_for("v3")
    thresholds = fit_thresholds(train_cases)
    train_packs = build_packs(train_cases, source_dataset=str(DATASET))
    test_packs = build_packs(test_cases, source_dataset=str(DATASET))
    model = fit_feature_model(train_packs, dictionary=dictionary)
    train_features = [extract_features(p, thresholds, model, dictionary=dictionary) for p in train_packs]
    test_features = [extract_features(p, thresholds, model, dictionary=dictionary) for p in test_packs]

    sop = learn_sop(train_features, train_labels, source=f"{DATASET.name}:manifest-train")
    graph = EvidenceGraph.build(
        train_features, train_labels, feature_model=model, dictionary=dictionary,
        source_dataset=str(DATASET),
    )
    test_match = match_many(graph, test_features, top_k=5)

    top1 = [m.candidates[0].label if m.candidates else None for m in test_match]
    sop_preds = [sop.predict(f).verdict for f in test_features]
    expert_preds = [d.verdict for d in diagnose_many(test_packs)]
    majority = Counter(train_labels).most_common(1)[0][0]

    sim_hist: Counter = Counter()
    for m in test_match:
        top = m.candidates[0].similarity if m.candidates else 0.0
        sim_hist[f"{int(top*10)/10:.1f}"] += 1

    bundle = {
        "dataset": {
            "name": DATASET.name,
            "train": len(train_cases),
            "test": len(test_cases),
            "train_dist": dict(Counter(train_labels)),
            "test_dist": dict(Counter(test_labels)),
        },
        "constraints": constraint_rows(),
        "constraint_version": CONSTRAINT_LIBRARY.version,
        "constraint_measured_on": CONSTRAINT_LIBRARY.measured_on,
        "expert_rules": {
            "metrics": list(EXPERT_METRICS),
            "thresholds": {k: dict(v) for k, v in EXPERT_THRESHOLDS.items()},
            "single_base": dict(SINGLE_METRIC_BASE),
            "single_direction": dict(SINGLE_METRIC_DIRECTION),
            "anomaly_level": dict(ANOMALY_LEVEL),
            "anomaly_order": list(ANOMALY_ORDER),
            "multi_metric_priority": MULTI_METRIC_PRIORITY,
            "multi_metric_requires": list(MULTI_METRIC_REQUIRES),
        },
        "sop": {
            "version": sop.version,
            "max_depth": sop.max_depth,
            "min_leaf_size": sop.min_leaf_size,
            "hash": sop.content_hash(),
            "root": sop.root.to_dict(),
        },
        "graph": {
            "version": graph.version,
            "nodes": len(train_features),
            "dictionary": dictionary.version,
            "dictionary_hash": dictionary.content_hash(),
            "label_dist": dict(Counter(train_labels)),
            "test_sim_hist": dict(sorted(sim_hist.items())),
        },
        "token_families": dict(
            Counter(t.split(":")[0] for f in train_features for t in f.tokens)
        ),
        "token_vocab": len({t for f in train_features for t in f.tokens}),
        "baselines": {
            name: {
                "acc": sum(1 for p, g in zip(preds, test_labels) if p == g) / len(test_labels),
                "answered": sum(1 for p in preds if p is not None),
                "per_class": per_class(preds, test_labels),
            }
            for name, preds in (
                ("majority", [majority] * len(test_labels)),
                ("graph_top1", top1),
                ("sop", sop_preds),
                ("expert", expert_preds),
            )
        },
        "reliability": reliability(train_cases, test_cases),
    }
    OUT.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} KB)")
    for name, b in bundle["baselines"].items():
        pc = b["per_class"]
        print(f"  {name:11s} acc={b['acc']:6.2%} answered={b['answered']:3d} "
              f"L1召回={pc['L1']['recall']:6.1%} L2召回={pc['L2']['recall']:6.1%} fiber召回={pc['fiber']['recall']:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
