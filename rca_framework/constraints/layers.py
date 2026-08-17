"""Machine-readable split of the former v6 constraint library.

The old library mixed physical facts, measurement contracts, and train-set
decision statistics.  This registry keeps the migration auditable: every old
constraint id must appear here exactly once, even when its content is split
across layers.
"""

from __future__ import annotations

from typing import Dict, Tuple


LAYER_PHYSICS = "physics"
LAYER_MEASUREMENT = "measurement_contract"
LAYER_DECISION_TREE = "decision_tree"

OLD_CONSTRAINT_IDS: Tuple[str, ...] = (
    "C1_bias_zero_means_laser_off",
    "C2_bias_healthy_band",
    "C3_temperature_operating_range",
    "C4_voltage_nominal_band",
    "C5_tx_power_range",
    "C6_tx_down_excludes_medium",
    "C7_rx_power_range",
    "C8_tx_ok_rx_down_indicates_medium",
    "C9_bidirectional_symmetry",
    "C10_all_lanes_vs_single_lane",
    "C11_media_snr_floor",
    "C12_no_absolute_link_loss",
    "C13_serdes_snr_unit_unknown",
    "C14_host_snr_mostly_missing",
    "C15_blackout_sentinel_is_not_laser_off",
    "C16_receive_symptom_constrains_far_transmit_chain",
    "C17_l2_side_receive_symptom_is_not_discriminative",
    "C18_single_lane_scope_does_not_exclude_fiber",
    "C19_population_prior_is_not_case_evidence",
    "C20_fiber_not_identifiable_from_current_telemetry",
    "C21_healthy_band_tx_level_is_not_attribution_evidence",
    "C22_receive_lane_imbalance_indicates_far_transmit_array",
    "C23_expert_receive_anomaly_on_l1_supports_l2",
    "C24_expert_receive_anomaly_on_l2_supports_l1",
    "C25_expert_local_chain_anomaly_on_l1_supports_l1",
    "C26_expert_local_chain_anomaly_on_l2_is_not_discriminative",
)

CONSTRAINT_LAYER_MAP: Dict[str, Tuple[str, ...]] = {
    "C1_bias_zero_means_laser_off": (LAYER_PHYSICS,),
    "C2_bias_healthy_band": (LAYER_DECISION_TREE,),
    "C3_temperature_operating_range": (LAYER_PHYSICS,),
    "C4_voltage_nominal_band": (LAYER_PHYSICS,),
    "C5_tx_power_range": (LAYER_PHYSICS, LAYER_DECISION_TREE),
    "C6_tx_down_excludes_medium": (LAYER_PHYSICS,),
    "C7_rx_power_range": (LAYER_PHYSICS, LAYER_DECISION_TREE),
    "C8_tx_ok_rx_down_indicates_medium": (LAYER_PHYSICS,),
    "C9_bidirectional_symmetry": (LAYER_PHYSICS,),
    "C10_all_lanes_vs_single_lane": (LAYER_PHYSICS,),
    "C11_media_snr_floor": (LAYER_DECISION_TREE,),
    "C12_no_absolute_link_loss": (LAYER_MEASUREMENT,),
    "C13_serdes_snr_unit_unknown": (LAYER_MEASUREMENT,),
    "C14_host_snr_mostly_missing": (LAYER_MEASUREMENT,),
    "C15_blackout_sentinel_is_not_laser_off": (LAYER_MEASUREMENT,),
    "C16_receive_symptom_constrains_far_transmit_chain": (LAYER_PHYSICS, LAYER_DECISION_TREE),
    "C17_l2_side_receive_symptom_is_not_discriminative": (LAYER_DECISION_TREE,),
    "C18_single_lane_scope_does_not_exclude_fiber": (LAYER_PHYSICS,),
    "C19_population_prior_is_not_case_evidence": (LAYER_MEASUREMENT,),
    "C20_fiber_not_identifiable_from_current_telemetry": (LAYER_MEASUREMENT,),
    "C21_healthy_band_tx_level_is_not_attribution_evidence": (
        LAYER_MEASUREMENT,
        LAYER_DECISION_TREE,
    ),
    "C22_receive_lane_imbalance_indicates_far_transmit_array": (
        LAYER_PHYSICS,
        LAYER_DECISION_TREE,
    ),
    "C23_expert_receive_anomaly_on_l1_supports_l2": (
        LAYER_PHYSICS,
        LAYER_DECISION_TREE,
    ),
    "C24_expert_receive_anomaly_on_l2_supports_l1": (
        LAYER_PHYSICS,
        LAYER_DECISION_TREE,
    ),
    "C25_expert_local_chain_anomaly_on_l1_supports_l1": (
        LAYER_PHYSICS,
        LAYER_DECISION_TREE,
    ),
    "C26_expert_local_chain_anomaly_on_l2_is_not_discriminative": (LAYER_DECISION_TREE,),
}


def validate_layer_map() -> None:
    missing = sorted(set(OLD_CONSTRAINT_IDS) - set(CONSTRAINT_LAYER_MAP))
    extra = sorted(set(CONSTRAINT_LAYER_MAP) - set(OLD_CONSTRAINT_IDS))
    if missing or extra:
        raise ValueError(f"constraint layer map mismatch: missing={missing}, extra={extra}")
    valid = {LAYER_PHYSICS, LAYER_MEASUREMENT, LAYER_DECISION_TREE}
    invalid = {
        layer
        for layers in CONSTRAINT_LAYER_MAP.values()
        for layer in layers
        if layer not in valid
    }
    if invalid:
        raise ValueError(f"unknown constraint layers: {sorted(invalid)}")


validate_layer_map()
