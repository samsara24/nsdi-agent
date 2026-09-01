"""Deterministic causal gates shared by decision-graph build and evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


RECEIVE_SYMPTOM_GROUPS = frozenset({
    "expert:single:rxpower",
    "expert:single:media_snr",
})

# Selected exclusively from leave-one-out evaluation on the 124 training cases.
# Each path is topology-, expert-rule-, and direction-specific.  The thresholds
# do not use test labels and do not change physical constraints.
LEARNED_PATH_CONFIGS: Mapping[str, Mapping[str, float | int]] = {
    "expert:multi_metric": {"min_similarity": 0.30, "min_neighbors": 1, "min_purity": 0.70, "top_k": 3},
    "expert:single:serdes_snr": {"min_similarity": 0.30, "min_neighbors": 3, "min_purity": 0.70, "top_k": 3, "terminal": 0},
    "expert:single:media_snr": {"min_similarity": 0.50, "min_neighbors": 1, "min_purity": 0.70, "top_k": 3},
    "expert:single:rxpower": {"min_similarity": 0.60, "min_neighbors": 2, "min_purity": 0.70, "top_k": 3, "terminal": 1},
}


def token_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def learned_path_match(
    *,
    tokens: Sequence[str],
    topology_id: str,
    group: str,
    verdict: str | None,
    training_rows: Sequence[Mapping[str, Any]],
    exclude_case_id: str | None = None,
    configs: Mapping[str, Mapping[str, float | int]] = LEARNED_PATH_CONFIGS,
) -> Mapping[str, Any] | None:
    """Return a calibrated positive path using only same-domain history."""

    config = configs.get(group)
    if config is None or verdict not in {"L1", "L2", "fiber"}:
        return None
    neighbors = []
    for row in training_rows:
        if row.get("case_id") == exclude_case_id:
            continue
        if row.get("topology_id") != topology_id or row.get("group") != group or row.get("verdict") != verdict:
            continue
        similarity = token_jaccard(tokens, row.get("tokens", ()))
        if similarity >= float(config["min_similarity"]):
            neighbors.append((similarity, str(row["case_id"]), row))
    neighbors.sort(key=lambda item: (-item[0], item[1]))
    neighbors = neighbors[: int(config["top_k"])]
    if len(neighbors) < int(config["min_neighbors"]):
        return None
    supporting = sum(row.get("label") == verdict for _, _, row in neighbors)
    purity = supporting / len(neighbors)
    if purity < float(config["min_purity"]):
        return None
    return {
        "verdict": verdict,
        "group": group,
        "topology_id": topology_id,
        "support": len(neighbors),
        "supporting": supporting,
        "purity": purity,
        "min_similarity": min(score for score, _, _ in neighbors),
        "neighbors": [
            {"case_id": case_id, "similarity": score, "label": row.get("label")}
            for score, case_id, row in neighbors
        ],
        "config": dict(config),
    }


def receive_symptom_context(
    group: str,
    verdict: str | None,
    symptom_side: str | None,
    tokens: Sequence[str],
    side_anomalies: Mapping[str, Mapping[str, Any]],
) -> str:
    """Classify whether a receive-side symptom has directional causal support."""

    if group not in RECEIVE_SYMPTOM_GROUPS:
        return "not_applicable"
    if verdict not in {"L1", "L2"} or symptom_side not in {"L1", "L2"}:
        return "unresolved_direction"

    token_set = set(tokens)
    direction = f"{verdict}_to_{symptom_side}"
    tx_fault = (
        f"lane:{direction}:tx_down" in token_set
        or any(f"status:{verdict}:{status}" in token_set for status in ("TxLOS", "TxLOL"))
        or side_anomalies.get(verdict, {}).get("txpower") == "lane_down"
    )
    if tx_fault:
        return "opposite_tx_fault"
    if f"lane:{direction}:tx_ok_rx_down" in token_set:
        return "sender_tx_ok_receive_down"
    return "uncorroborated_receive_symptom"
