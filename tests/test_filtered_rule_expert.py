from rca_framework.filtered_rule_expert import assess_filtered_rule_expert


def telemetry(*, host_fault=False):
    host = {"0": 25.0, "1": 0.0, "2": 25.0, "3": 25.0} if host_fault else {str(i): None for i in range(4)}
    return {
        "txpower": {
            "L1": {"0": 0.8, "1": 0.8, "2": 0.8, "3": 0.8},
            "L2": {"0": 1.1, "1": 1.2, "2": 1.3, "3": 1.4},
        },
        "rxpower": {"L1": {str(i): 0.5 for i in range(4)}, "L2": {"0": 1.5, "1": 1.4, "2": -40, "3": 1.2}},
        "media_snr": {"L1": {str(i): 25.0 for i in range(4)}, "L2": {"0": 25.0, "1": 25.0, "2": 0, "3": 25.0}},
        "serdes_snr": {"L1": {str(i): 700000 for i in range(4)}, "L2": {"0": 700000, "1": 700000, "2": 116885, "3": 700000}},
        "host_snr": {"L1": {str(i): None for i in range(4)}, "L2": host},
        "RxLOS": {"L1": "Normal", "L2": "Abnormal"},
        "RxLOL": {"L1": "Normal", "L2": "Abnormal"},
        "TxLOS": {"L1": "Normal", "L2": "Normal"},
        "TxLOL": {"L1": "Normal", "L2": "Normal"},
    }


def test_multi_metric_no_longer_blindly_flips_aligned_receive_failure_to_sender():
    assessment = assess_filtered_rule_expert(
        expert_group="expert:multi_metric",
        expert_verdict="L1",
        symptom_side="L2",
        tokens=("lane:L1_to_L2:tx_ok_rx_down",),
        telemetry=telemetry(),
    )
    assert assessment.verdict is None
    assert assessment.strength == "none"
    assert assessment.terminal is False
    assert assessment.rule == "aligned_receive_chain_ambiguous"
    assert assessment.candidates == ("L2", "fiber", "L1")


def test_host_corroboration_makes_aligned_local_receive_chain_strong():
    assessment = assess_filtered_rule_expert(
        expert_group="expert:multi_metric",
        expert_verdict="L1",
        symptom_side="L2",
        tokens=("lane:L1_to_L2:tx_ok_rx_down",),
        telemetry=telemetry(host_fault=True),
    )
    assert assessment.verdict == "L2"
    assert assessment.strength == "moderate"
    assert assessment.terminal is False


def test_far_tx_fault_retains_opposite_direction():
    values = telemetry()
    values["TxLOS"]["L1"] = "Abnormal"
    assessment = assess_filtered_rule_expert(
        expert_group="expert:multi_metric",
        expert_verdict="L1",
        symptom_side="L2",
        tokens=(),
        telemetry=values,
    )
    assert assessment.verdict == "L1"
    assert assessment.strength == "strong"
