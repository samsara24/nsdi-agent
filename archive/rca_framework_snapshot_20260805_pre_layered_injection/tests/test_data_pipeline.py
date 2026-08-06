import json
from copy import deepcopy
from pathlib import Path

from rca_framework.data import prepare_dataset


def test_prepare_swaps_entire_200g_400g_case_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "prepared"
    source.mkdir()
    case = {
        "alarm_time": "2025-01-01 00:00:00",
        "alarm_ip_interface": "10.0.0.1--200GHL1/1/1",
        "label": "local",
        "link_side_ip_interface_map": {
            "local": "10.0.0.1--200GHL1/1/1",
            "remote": "10.0.0.2--400GHL1/2/2:1",
        },
        "rxpower": {"local": {"0": 20}, "remote": {"0": 40}},
        "vendor_sn": {"local": "secret-a", "remote": "secret-b"},
    }
    original = deepcopy(case)
    (source / "001.json").write_text(json.dumps(case), encoding="utf-8")
    report = prepare_dataset(source, output, "unit-test-secret", tmp_path / "archive" / "manifest.json")
    prepared = json.loads((output / "case_000001.json").read_text(encoding="utf-8"))

    assert report["output_file_count"] == 1
    assert prepared["label"] == "L2"
    assert prepared["rxpower"] == {"L1": {"0": 40}, "L2": {"0": 20}}
    assert prepared["link_side_ip_interface_map"] == {
        "L1": "L1_ENDPOINT--400G_PORT",
        "L2": "L2_ENDPOINT--200G_PORT",
    }
    assert prepared["alarm_ip_interface"] == "L2_ENDPOINT--200G_PORT"
    assert prepared["_meta"]["endpoint_values_swapped"] is True
    assert json.loads((source / "001.json").read_text(encoding="utf-8")) == original
    assert "secret-a" not in json.dumps(prepared)


def test_prepare_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "prepared"
    source.mkdir()
    output.mkdir()
    (source / "001.json").write_text("{}", encoding="utf-8")
    try:
        prepare_dataset(source, output, "secret", tmp_path / "manifest.json")
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing dataset must not be overwritten")
