"""用当前版本的可核对性检查重跑已保存的解释输出。

与 `replay_constraint_check.py` 同一个用途：检查逻辑的改动与模型无关，
不该为了重新评估而再占一次 GPU。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs
from rca_framework.expert import diagnose_many
from rca_framework.features import dictionary_for, extract_features, fit_feature_model
from rca_framework.llm.explain import check_explanation, parse_explanation, summarize_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--feature-profile", default="v3")
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    train_cases = cases_by_manifest_split(args.data_dir, "train")
    test_cases = cases_by_manifest_split(args.data_dir, "test")
    train_packs, test_packs = build_packs(train_cases), build_packs(test_cases)
    diagnoses = diagnose_many(test_packs)

    thresholds = fit_thresholds(train_cases)
    dictionary = dictionary_for(args.feature_profile)
    model = fit_feature_model(train_packs, dictionary=dictionary)
    tokens_by_id = {
        pack.case_id: extract_features(pack, thresholds, model, dictionary=dictionary).tokens
        for pack in test_packs
    }
    diag_by_id = {pack.case_id: diag for pack, diag in zip(test_packs, diagnoses)}

    reports = []
    changed: List[Tuple[str, bool, bool]] = []
    for record in records:
        case_id = record["case_id"]
        diagnosis = diag_by_id[case_id]
        rule_evidence = [
            (side.side, metric) for side in diagnosis.sides for metric, _ in side.anomalies
        ]
        report = check_explanation(
            parse_explanation(record["raw_output"]),
            available_tokens=tokens_by_id[case_id],
            rule_evidence=rule_evidence,
            rule_verdict=diagnosis.verdict,
        )
        reports.append(report)
        if report["all_pass"] != record["checks"]["all_pass"]:
            changed.append((case_id, record["checks"]["all_pass"], report["all_pass"]))

    print(json.dumps(summarize_checks(reports), ensure_ascii=False, indent=2))
    print(f"\n判定发生变化的 case：{len(changed)}")
    for case_id, before, after in changed:
        print(f"  {case_id}: {before} -> {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
