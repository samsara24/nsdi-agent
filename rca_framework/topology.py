"""Active filtered-rule dataset topology and logical-lane contracts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


TOPOLOGY_CONTRACT_VERSION = "filtered-rule-topology-v1"

SOURCE_TOPOLOGIES: Dict[str, Dict[str, Any]] = {
    "all_data": {
        "topology_id": "400g-200g-logical4",
        "endpoint_speeds": {"L1": "400G", "L2": "200G"},
        "optical_lane_count": {"L1": 4, "L2": 4},
        "serdes_lane_count": {"L1": 4, "L2": 4},
        "same_index_optical_pairing": True,
        "absolute_link_loss_allowed": False,
    },
    "rule1_channel_not_4": {
        "topology_id": "400g-400g-logical8",
        "endpoint_speeds": {"L1": "400G", "L2": "400G"},
        "optical_lane_count": {"L1": 8, "L2": 8},
        "serdes_lane_count": {"L1": 4, "L2": 4},
        "same_index_optical_pairing": True,
        "absolute_link_loss_allowed": False,
    },
}


def source_dataset_of(case: Mapping[str, Any], fallback: str = "") -> str:
    contract = case.get("_dataset_contract")
    if isinstance(contract, Mapping) and contract.get("source_dataset"):
        return str(contract["source_dataset"])
    return fallback


def topology_contract_for_source(source_dataset: str) -> Dict[str, Any]:
    return dict(SOURCE_TOPOLOGIES.get(source_dataset, {}))


def topology_id_of(case: Mapping[str, Any], fallback_source: str = "") -> str:
    contract = case.get("_dataset_contract")
    if isinstance(contract, Mapping) and contract.get("topology_id"):
        return str(contract["topology_id"])
    source = source_dataset_of(case, fallback_source)
    return str(SOURCE_TOPOLOGIES.get(source, {}).get("topology_id", source))


def lane_widths_of(case: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for metric in ("bias", "txpower", "rxpower", "media_snr", "host_snr", "serdes_snr"):
        block = case.get(metric)
        if not isinstance(block, Mapping):
            continue
        widths: Dict[str, int] = {}
        for side in ("L1", "L2"):
            values = block.get(side)
            if isinstance(values, (Mapping, list, tuple)):
                widths[side] = len(values)
        if widths:
            result[metric] = widths
    return result


def lane_profile_of(case: Mapping[str, Any], fallback_source: str = "") -> str:
    source = source_dataset_of(case, fallback_source)
    topology = topology_id_of(case, source)
    widths = lane_widths_of(case)
    optical = widths.get("txpower") or widths.get("rxpower") or {}
    serdes = widths.get("serdes_snr") or {}
    return (
        f"{topology}:optical-{optical.get('L1', 0)}x{optical.get('L2', 0)}:"
        f"serdes-{serdes.get('L1', 0)}x{serdes.get('L2', 0)}"
    )


def topology_compatible(left: str, right: str) -> bool:
    return not left or not right or left == right


def shared_lane_keys(
    case: Mapping[str, Any], metric: str, left: str = "L1", right: str = "L2"
) -> Tuple[str, ...]:
    block = case.get(metric)
    if not isinstance(block, Mapping):
        return ()
    left_values, right_values = block.get(left), block.get(right)
    if not isinstance(left_values, Mapping) or not isinstance(right_values, Mapping):
        return ()
    return tuple(sorted(set(map(str, left_values)) & set(map(str, right_values))))
