"""Causal refinement of legacy expert directions for the active filtered-rule data.

The legacy expert tree maps receive-side multi-metric symptoms to the opposite
transmitter.  That is a useful candidate direction, but it is not a sufficient
causal rule: the same observations can be produced by the receiving module or
the medium.  This module keeps the legacy diagnosis for audit and adds the
lane-aligned checks needed by the active dataset contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


VERSION = "filtered-rule-expert-causal-v2"
SIDES = ("L1", "L2")
RX_DOWN = -40.0
TX_DOWN = -40.0
MEDIA_DOWN = 0.0
SERDES_LOW = 458750.0


def opposite(side: str) -> str:
    return "L2" if side == "L1" else "L1"


def _side_block(telemetry: Mapping[str, Any], metric: str, side: str) -> Mapping[str, Any]:
    block = telemetry.get(metric, {})
    if not isinstance(block, Mapping):
        return {}
    values = block.get(side, {})
    return values if isinstance(values, Mapping) else {}


def _status_abnormal(telemetry: Mapping[str, Any], status: str, side: str) -> bool:
    block = telemetry.get(status, {})
    value = block.get(side) if isinstance(block, Mapping) else None
    return isinstance(value, str) and value.strip().lower() not in {"", "normal", "none", "missing"}


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _aligned_receive_lanes(telemetry: Mapping[str, Any], side: str) -> tuple[str, ...]:
    """Return lanes where far Tx is present and local Rx/media/SerDes co-fail."""
    sender = opposite(side)
    tx = _side_block(telemetry, "txpower", sender)
    rx = _side_block(telemetry, "rxpower", side)
    media = _side_block(telemetry, "media_snr", side)
    serdes = _side_block(telemetry, "serdes_snr", side)
    aligned = []
    for lane in sorted(set(tx) & set(rx) & set(media) & set(serdes), key=str):
        tx_value, rx_value = _numeric(tx.get(lane)), _numeric(rx.get(lane))
        media_value, serdes_value = _numeric(media.get(lane)), _numeric(serdes.get(lane))
        if (
            tx_value is not None and tx_value > TX_DOWN
            and rx_value is not None and rx_value <= RX_DOWN
            and media_value is not None and media_value <= MEDIA_DOWN
            and serdes_value is not None and serdes_value < SERDES_LOW
        ):
            aligned.append(str(lane))
    return tuple(aligned)


def _host_fault(telemetry: Mapping[str, Any], side: str) -> bool:
    values = [_numeric(value) for value in _side_block(telemetry, "host_snr", side).values()]
    observed = [value for value in values if value is not None]
    # All-null/all-zero blocks are the known missing sentinel, not a local fault.
    if not any(value > 0 for value in observed):
        return False
    return any(value <= 0 or value < 22.8 for value in observed)


def _opposite_tx_fault(telemetry: Mapping[str, Any], receive_side: str) -> bool:
    sender = opposite(receive_side)
    if _status_abnormal(telemetry, "TxLOS", sender) or _status_abnormal(telemetry, "TxLOL", sender):
        return True
    values = [_numeric(value) for value in _side_block(telemetry, "txpower", sender).values()]
    return any(value is not None and value <= TX_DOWN for value in values)


@dataclass(frozen=True)
class FilteredRuleExpertAssessment:
    verdict: str | None
    strength: str
    rule: str
    reason: str
    evidence: tuple[str, ...]
    candidates: tuple[str, ...] = ()
    original_verdict: str | None = None
    version: str = VERSION

    @property
    def terminal(self) -> bool:
        return self.strength == "strong" and self.verdict is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "verdict": self.verdict,
            "strength": self.strength,
            "terminal": self.terminal,
            "rule": self.rule,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "candidates": list(self.candidates),
            "original_verdict": self.original_verdict,
        }


def assess_filtered_rule_expert(
    *,
    expert_group: str,
    expert_verdict: str | None,
    symptom_side: str | None,
    tokens: Sequence[str],
    telemetry: Mapping[str, Any],
) -> FilteredRuleExpertAssessment:
    """Refine a legacy diagnosis without consulting a case label."""
    token_set = set(tokens)

    # A confirmed local transmitter failure remains the strongest endpoint rule.
    for side in SIDES:
        if any(token.startswith(f"lane:{side}_to_") and ":tx_down" in token for token in token_set):
            status = _status_abnormal(telemetry, "TxLOS", side) or _status_abnormal(telemetry, "TxLOL", side)
            return FilteredRuleExpertAssessment(
                verdict=side,
                strength="strong" if status else "moderate",
                rule="confirmed_tx_fault",
                reason=f"{side}发送lane触底" + ("且TxLOS/TxLOL异常" if status else "，缺少发送状态复核"),
                evidence=tuple(sorted(token for token in token_set if token.startswith(f"lane:{side}_to_") and ":tx_down" in token)),
                candidates=(side,),
                original_verdict=expert_verdict,
            )

    directions = {
        token.split(":")[1]
        for token in token_set
        if token.startswith("lane:") and ":tx_ok_rx_down" in token
    }
    if {"L1_to_L2", "L2_to_L1"} <= directions:
        return FilteredRuleExpertAssessment(
            verdict="fiber", strength="moderate", rule="bidirectional_receive_loss",
            reason="两端均已发光且双向同编号接收异常，形成介质候选；仍需现场证据终裁",
            evidence=tuple(sorted(token for token in token_set if ":tx_ok_rx_down" in token)),
            candidates=("fiber", "L1", "L2"),
            original_verdict=expert_verdict,
        )

    if symptom_side in SIDES and expert_group in {
        "expert:multi_metric", "expert:single:rxpower", "expert:single:media_snr"
    }:
        sender = opposite(symptom_side)
        if _opposite_tx_fault(telemetry, symptom_side):
            return FilteredRuleExpertAssessment(
                verdict=sender, strength="strong", rule="receive_with_far_tx_fault",
                reason=f"{symptom_side}接收异常且{sender}发送链存在独立故障证据",
                evidence=(f"{sender}.TxLOS/TxLOL/txpower",), original_verdict=expert_verdict,
                candidates=(sender,),
            )

        aligned = _aligned_receive_lanes(telemetry, symptom_side)
        rx_status = _status_abnormal(telemetry, "RxLOS", symptom_side) or _status_abnormal(telemetry, "RxLOL", symptom_side)
        host_fault = _host_fault(telemetry, symptom_side)
        if aligned and host_fault:
            return FilteredRuleExpertAssessment(
                verdict=symptom_side, strength="moderate", rule="local_receive_chain_with_host_fault",
                reason=f"{sender}正常发光，{symptom_side}同lane光学/SerDes异常且可选host侧证据同向，增强本端接收链候选",
                evidence=tuple(f"lane:{lane}" for lane in aligned) + (f"host_snr:{symptom_side}",),
                candidates=(symptom_side, "fiber"),
                original_verdict=expert_verdict,
            )
        if aligned and rx_status:
            return FilteredRuleExpertAssessment(
                verdict=None, strength="none", rule="aligned_receive_chain_ambiguous",
                reason=f"{sender}正常发光，{symptom_side}同lane Rx/media/SerDes 与RxLOS/RxLOL共同异常；训练校准证明本端、介质与对端标签混合，取消旧对端终裁",
                evidence=tuple(f"lane:{lane}" for lane in aligned) + (f"RxLOS/RxLOL:{symptom_side}",),
                candidates=(symptom_side, "fiber", sender),
                original_verdict=expert_verdict,
            )
        if f"{sender}_to_{symptom_side}" in directions:
            return FilteredRuleExpertAssessment(
                verdict=None,
                strength="none",
                rule="tx_ok_rx_down_unresolved",
                reason=f"{sender}已发光而{symptom_side}接收异常；接收模块、介质和实际出光质量仍不可区分",
                evidence=(f"lane:{sender}_to_{symptom_side}:tx_ok_rx_down",),
                candidates=(symptom_side, "fiber", sender),
                original_verdict=expert_verdict,
            )
        return FilteredRuleExpertAssessment(
            verdict=None, strength="none", rule="receive_symptom_without_causal_support",
            reason="接收侧异常没有对端发送故障或本端同lane接收链证据，旧规则的对端终裁被取消",
            evidence=(), original_verdict=expert_verdict,
            candidates=(symptom_side, "fiber", sender),
        )

    if expert_group == "expert:single:serdes_snr" and symptom_side in SIDES:
        return FilteredRuleExpertAssessment(
            verdict=symptom_side, strength="advisory", rule="local_serdes_advisory",
            reason="SerDes异常保留本端候选，但缺少host或状态复核时不终裁",
            evidence=(f"serdes_snr:{symptom_side}",), original_verdict=expert_verdict,
            candidates=(symptom_side,),
        )

    return FilteredRuleExpertAssessment(
        verdict=expert_verdict, strength="advisory" if expert_verdict else "none",
        rule="legacy_advisory", reason="旧专家结论仅作为候选，等待独立物理或训练校准证据",
        evidence=(), original_verdict=expert_verdict,
        candidates=(expert_verdict,) if expert_verdict else (),
    )
