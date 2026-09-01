import json
from pathlib import Path

from scripts.build_expanded_n5a_expert_review import build_report


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "20260816_expanded-pattern-conflict"


def test_weighted_n5a_review_report_is_locked_to_113_of_147(tmp_path: Path) -> None:
    html_path = tmp_path / "review.html"
    cases_path = tmp_path / "cases.json"

    summary = build_report(INPUT, html_path, cases_path)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")

    assert summary["weighted_n5a_count"] == 147
    assert summary["weighted_n5a_correct"] == 113
    assert summary["weighted_n5a_conflict_count"] == 34
    assert payload["conflict_count"] == 34
    assert len({row["test_case_id"] for row in payload["cases"]}) == 34
    assert html.count('class="expert-review"') == 34
    assert html.count('class="pattern"') == 34
    assert "expert_label_annotations_n5a_weighted.json" in html
    assert "rca-expert-annotations:20260816-expanded-n5a-weighted:v1" in html
    assert "S_weighted = 0.8×S_feature + 0.2×S_graph" in html
