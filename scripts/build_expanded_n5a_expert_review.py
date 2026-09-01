#!/usr/bin/env python3
"""Build the expert-review HTML for weighted N5a label conflicts.

This reproduces the retrospective routing snapshot requested for review:
``S_weighted = 0.8 * S_feature + 0.2 * S_graph`` and N5a when the
weighted score is at least 0.80.  Labels are inspected only after matching.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.expanded_dual import DualMatcher, EvidenceView, build_views

import scripts.analyze_expanded_rca_patterns as analysis


FEATURE_WEIGHT = 0.8
GRAPH_WEIGHT = 0.2
N5A_THRESHOLD = 0.8
REPORT_SCHEMA = "expanded-weighted-n5a-expert-review-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def path_signature(path: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(path[key]) for key in ("side", "measurement", "predicate", "symptom", "layer"))


def graph_nodes(edges: Iterable[str]) -> set[str]:
    nodes: set[str] = set()
    for edge in edges:
        source, _relation, target = edge.split("|", 2)
        nodes.update((source, target))
    return nodes


def display_graph_match(query: EvidenceView, history: EvidenceView, similarity: float) -> dict[str, Any]:
    query_edges, history_edges = set(query.graph_edges), set(history.graph_edges)
    query_nodes, history_nodes = graph_nodes(query_edges), graph_nodes(history_edges)
    query_paths = {path_signature(path): path for path in query.paths}
    history_paths = {path_signature(path): path for path in history.paths}
    shared_paths = [query_paths[key] for key in sorted(set(query_paths) & set(history_paths))]
    node_union = query_nodes | history_nodes
    return {
        "similarity": similarity,
        "node_similarity": len(query_nodes & history_nodes) / len(node_union) if node_union else 1.0,
        "shared_nodes": sorted(query_nodes & history_nodes),
        "query_only_nodes": sorted(query_nodes - history_nodes),
        "train_only_nodes": sorted(history_nodes - query_nodes),
        "shared_edges": sorted(query_edges & history_edges),
        "query_only_edges": sorted(query_edges - history_edges),
        "train_only_edges": sorted(history_edges - query_edges),
        "shared_predicate_paths": shared_paths,
    }


def label_audit_map(data_contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in data_contract.get("cases", [])}


def build_report(input_dir: Path, output_html: Path, output_cases: Path) -> dict[str, Any]:
    train = load_jsonl(input_dir / "clean_train.jsonl")
    test = load_jsonl(input_dir / "expanded_test.jsonl")
    if (len(train), len(test)) != (122, 341):
        raise ValueError(f"expected expanded expert-clean 122/341, got {len(train)}/{len(test)}")

    train_views, thresholds, feature_model, _ = build_views(train)
    test_views, _, _, _ = build_views(test, thresholds=thresholds, feature_model=feature_model)
    matcher = DualMatcher(train, train_views)
    train_by_id = {str(case["case_id"]): case for case in train}
    train_view_by_id = {view.case_id: view for view in train_views}
    comparison_model = analysis.fit_comparison_model(train)

    data_contract = json.loads((input_dir / "data_contract.json").read_text(encoding="utf-8"))
    audit_by_id = label_audit_map(data_contract)
    existing_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    learned_model = json.loads((input_dir / "learned_predicate_model.json").read_text(encoding="utf-8"))
    added_manifest = json.loads((input_dir / "added_cases_manifest.json").read_text(encoding="utf-8"))

    selected: list[tuple[dict[str, Any], EvidenceView, Any, float]] = []
    n5a_count = n5a_correct = 0
    for case, view in zip(test, test_views):
        result = matcher.match(case, view, feature_threshold=0.0, graph_threshold=0.0)
        # Preserve the exact tie-break used for the published 113/147 retrospective
        # snapshot: score, feature score, graph score, then lexicographically larger ID.
        candidate = max(
            result.candidates,
            key=lambda row: (
                FEATURE_WEIGHT * row.feature_similarity + GRAPH_WEIGHT * row.graph_similarity,
                row.feature_similarity,
                row.graph_similarity,
                row.case_id,
            ),
        )
        weighted = FEATURE_WEIGHT * candidate.feature_similarity + GRAPH_WEIGHT * candidate.graph_similarity
        if weighted < N5A_THRESHOLD:
            continue
        n5a_count += 1
        n5a_correct += int(candidate.label == str(case["label"]))
        if candidate.label != str(case["label"]):
            selected.append((case, view, candidate, weighted))

    if (n5a_count, n5a_correct, len(selected)) != (147, 113, 34):
        raise AssertionError(
            f"weighted N5a regression changed: expected 147/113/34, "
            f"got {n5a_count}/{n5a_correct}/{len(selected)}"
        )

    patterns: list[dict[str, Any]] = []
    raw_cases: dict[str, dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []
    for index, (case, query_view, candidate, weighted) in enumerate(selected, start=1):
        history = train_by_id[candidate.case_id]
        history_view = train_view_by_id[candidate.case_id]
        query_tokens, history_tokens = set(query_view.feature_tokens), set(history_view.feature_tokens)
        shared_tokens = sorted(query_tokens & history_tokens)
        shared_weight = sum(matcher.feature_idf.get(token, 1.0) for token in shared_tokens)
        union_weight = sum(matcher.feature_idf.get(token, 1.0) for token in query_tokens | history_tokens)
        graph_detail = display_graph_match(query_view, history_view, candidate.graph_similarity)
        shared_criteria = [
            {
                "path": path_signature(path),
                "source_tokens": (str(path["token"]),),
                "criteria": (str(path.get("criterion", "固定证据谓词")),),
                "predicate_type": str(path.get("predicate_type", "feature_projection")),
                "provenance": str(path.get("provenance", "expanded-dual-match-v1")),
                "quantifier": str(path.get("quantifier", "token-defined")),
            }
            for path in graph_detail["shared_predicate_paths"]
        ]
        differences = analysis.compare_case_features(history, case, comparison_model)
        quality_ok = candidate.quality_compatible
        critical_conflicts = [list(item) for item in candidate.critical_conflicts]
        exact = (
            candidate.feature_similarity == 1.0
            and candidate.graph_similarity == 1.0
            and quality_ok
            and not critical_conflicts
        )
        train_audit = audit_by_id.get(candidate.case_id, {})
        test_audit = audit_by_id.get(str(case["case_id"]), {})
        pattern_id = f"N5A-{index:03d}"
        routing_note = (
            f"S_weighted = 0.8×{candidate.feature_similarity:.3f} + "
            f"0.2×{candidate.graph_similarity:.3f} = {weighted:.3f} ≥ 0.800；"
            f"因此该 test case 在本次回溯估算中进入 N5a，并复用历史标签 {candidate.label}。"
        )
        review_reason = (
            f"加权相似度 {weighted:.3f} 已达 N5a 阈值，但历史标签 "
            f"{candidate.label} 与当前测试标签 {case['label']} 不同；需要专家确认哪一侧标签应修改。"
        )
        query_learned = sum(bool(path.get("learned")) for path in query_view.paths)
        history_learned = sum(bool(path.get("learned")) for path in history_view.paths)
        pattern = {
            "pattern_id": pattern_id,
            "summary": f"{case['case_id']} ↔ {candidate.case_id}：N5a 复用标签冲突",
            "feature_similarity": candidate.feature_similarity,
            "graph_similarity": candidate.graph_similarity,
            "weighted_similarity": weighted,
            "routing_note": routing_note,
            "quadrant": "N5a_weighted",
            "shared_evidence": shared_tokens,
            "query_only_evidence": sorted(query_tokens - history_tokens),
            "train_only_evidence": sorted(history_tokens - query_tokens),
            "shared_weight": shared_weight,
            "union_weight": union_weight,
            "weighted_shared_terms": [
                {"token": token, "idf": matcher.feature_idf.get(token, 1.0), "meaning": analysis.token_logic(token)}
                for token in shared_tokens
            ],
            "physical_logic": [{"token": token, "logic": analysis.token_logic(token)} for token in shared_tokens],
            "physical_summary": "；".join(analysis.token_logic(token) for token in shared_tokens[:2]),
            "query_graph": {"learned_range_path_count": query_learned},
            "train_graph": {"learned_range_path_count": history_learned},
            "graph_match": graph_detail,
            "shared_relation_criteria": shared_criteria,
            "feature_differences": differences,
            "exact_two_dimensional_match": exact,
            "quality_compatible": quality_ok,
            "critical_evidence_conflicts": critical_conflicts,
            "review_priority": "high",
            "review_reason": review_reason,
            "confirmed_path_status": "unavailable",
            "identifiability_boundary": (
                "当前图是可观测证据图，不包含人工确认的排他性根因路径；"
                "N5a 高相似不能单独证明历史标签正确。"
            ),
            "train_labels": [candidate.label],
            "test_label": str(case["label"]),
            "why_same_pattern": (
                f"特征 IDF-Jaccard={candidate.feature_similarity:.3f}，五层 typed-edge "
                f"IDF-Jaccard={candidate.graph_similarity:.3f}，特征主导加权分={weighted:.3f}。"
            ),
            "label_conflict_analysis": (
                f"历史 case 当前标签={candidate.label}，测试 case 当前标签={case['label']}。"
                "两个标签只在相似度计算完成后用于识别冲突。"
            ),
            "impact": "若直接走 N5a，系统会输出历史标签；专家复核结果将用于下一版历史桶纯度校准。",
            "cases": [
                {
                    "case_id": candidate.case_id,
                    "split": "train",
                    "label": candidate.label,
                    "original_label": train_audit.get("original_label", history["label"]),
                    "label_status": train_audit.get("label_status", "unreviewed"),
                    "prediction": None,
                    "feature_similarity": candidate.feature_similarity,
                    "graph_similarity": candidate.graph_similarity,
                    "quadrant": "N5a_weighted",
                    "tokens": list(history_view.feature_tokens),
                },
                {
                    "case_id": str(case["case_id"]),
                    "split": "test",
                    "label": str(case["label"]),
                    "original_label": test_audit.get("original_label", case["label"]),
                    "label_status": test_audit.get("label_status", "unreviewed"),
                    "prediction": candidate.label,
                    "feature_similarity": candidate.feature_similarity,
                    "graph_similarity": candidate.graph_similarity,
                    "quadrant": "N5a_weighted",
                    "tokens": list(query_view.feature_tokens),
                },
            ],
        }
        patterns.append(pattern)
        for raw_case, split, audit in ((history, "train", train_audit), (case, "test", test_audit)):
            case_id = str(raw_case["case_id"])
            raw_cases[case_id] = {
                "label": str(raw_case["label"]),
                "split": split,
                "label_audit": audit,
                "raw": raw_case,
            }
        case_rows.append(
            {
                "pattern_id": pattern_id,
                "train_case_id": candidate.case_id,
                "train_label": candidate.label,
                "test_case_id": str(case["case_id"]),
                "test_label": str(case["label"]),
                "test_label_status": test_audit.get("label_status", "unreviewed"),
                "feature_similarity": candidate.feature_similarity,
                "graph_similarity": candidate.graph_similarity,
                "weighted_similarity": weighted,
            }
        )

    patterns.sort(key=lambda row: (-row["weighted_similarity"], row["cases"][1]["case_id"]))
    for index, pattern in enumerate(patterns, start=1):
        pattern["pattern_id"] = f"N5A-{index:03d}"
    case_rows.sort(key=lambda row: (-row["weighted_similarity"], row["test_case_id"]))
    for index, row in enumerate(case_rows, start=1):
        row["pattern_id"] = f"N5A-{index:03d}"

    summary = copy.deepcopy(existing_summary)
    summary.update(
        {
            "schema_version": REPORT_SCHEMA,
            "pattern_count": len(patterns),
            "review_priority_counts": {"high": len(patterns)},
            "weighted_feature_weight": FEATURE_WEIGHT,
            "weighted_graph_weight": GRAPH_WEIGHT,
            "weighted_n5a_threshold": N5A_THRESHOLD,
            "weighted_n5a_count": n5a_count,
            "weighted_n5a_correct": n5a_correct,
            "weighted_n5a_conflict_count": len(patterns),
        }
    )
    review_banner = (
        f"<b>审核目标：</b>回溯路由使用 <code>S_weighted = 0.8×S_feature + 0.2×S_graph</code>，"
        f"<code>S_weighted ≥ 0.80</code> 的 N5a 共 <b>{n5a_count}</b> 条；历史标签与当前测试标签一致 "
        f"<b>{n5a_correct}</b> 条，本页仅展示待专家复核的 <b>{len(patterns)}</b> 条不一致 case。"
        "该阈值是同一测试集上的回溯工作点，不是已冻结的生产 N4 阈值。"
    )
    report_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "learned_predicates": learned_model,
        "adjudication": {"pairs": []},
        "patterns": patterns,
        "raw_cases": raw_cases,
        "added_cases": added_manifest.get("cases", []),
        "missing_old_cases": [],
        "focus_pairs": [],
        "report_title": "Expanded RCA N5a 异标签专家复核",
        "page_heading": "RCA N5a 相似 Case 标签审核",
        "candidate_heading": "4. N5a 历史复用异标签候选",
        "review_list_heading": "待审核：N5a 历史标签与当前测试标签不同",
        "review_banner": review_banner,
        "storage_key": "rca-expert-annotations:20260816-expanded-n5a-weighted:v1",
        "export_filename": "expert_label_annotations_n5a_weighted.json",
        "export_experiment_id": "20260816_expanded-n5a-weighted-review",
    }
    html = analysis.render_html(report_data)
    html = html.replace(
        "报告用于展示的临时阈值是 S_feature≥",
        "本页候选使用的回溯路由是 S_weighted=0.8×S_feature+0.2×S_graph≥0.80。原报告用于展示的二维分区阈值是 S_feature≥",
        1,
    )
    output_html.write_text(html, encoding="utf-8")
    output_cases.write_text(
        json.dumps(
            {
                "schema_version": REPORT_SCHEMA,
                "feature_weight": FEATURE_WEIGHT,
                "graph_weight": GRAPH_WEIGHT,
                "n5a_threshold": N5A_THRESHOLD,
                "n5a_count": n5a_count,
                "n5a_correct": n5a_correct,
                "conflict_count": len(case_rows),
                "cases": case_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/20260816_expanded-pattern-conflict"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("experiments/20260816_expanded-pattern-conflict/expanded_n5a_label_conflict_review.html"),
    )
    parser.add_argument(
        "--output-cases",
        type=Path,
        default=Path("experiments/20260816_expanded-pattern-conflict/n5a_label_conflict_cases.json"),
    )
    args = parser.parse_args()
    summary = build_report(args.input_dir, args.output_html, args.output_cases)
    print(
        json.dumps(
            {
                "output_html": str(args.output_html),
                "output_cases": str(args.output_cases),
                "n5a": summary["weighted_n5a_count"],
                "correct": summary["weighted_n5a_correct"],
                "conflicts": summary["weighted_n5a_conflict_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
