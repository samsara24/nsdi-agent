from __future__ import annotations

from rca_framework.decision_tree import NumericFeatureRow, fit_numeric_decision_tree


def row(case_id: str, value: float, *, tx_down: float = 0.0) -> NumericFeatureRow:
    return NumericFeatureRow(
        case_id=case_id,
        values={
            "L1.rxpower.min": value,
            "L1.txpower.down_count": tx_down,
        },
    )


def test_numeric_tree_uses_readable_interval_paths():
    rows = [row(f"l2-{index}", -4.0) for index in range(8)]
    rows += [row(f"l1-{index}", 0.0) for index in range(8)]
    labels = ["L2"] * 8 + ["L1"] * 8
    tree = fit_numeric_decision_tree(rows, labels, max_depth=1, min_leaf_size=2)

    prediction = tree.predict(row("query", -5.0))

    assert prediction.verdict == "L2"
    assert prediction.path
    assert "<=" in prediction.path[0] or ">" in prediction.path[0]
    assert prediction.support >= 2


def test_physical_pruning_blocks_fiber_leaf_after_tx_down():
    rows = [row(f"fiber-{index}", 0.0, tx_down=1.0) for index in range(8)]
    rows += [row(f"l2-{index}", 0.0, tx_down=0.0) for index in range(8)]
    labels = ["fiber"] * 8 + ["L2"] * 8
    tree = fit_numeric_decision_tree(rows, labels, max_depth=1, min_leaf_size=2)

    prediction = tree.predict(row("query", 0.0, tx_down=1.0))

    assert prediction.verdict is None
    assert "P5_tx_down_excludes_medium" in prediction.reason
