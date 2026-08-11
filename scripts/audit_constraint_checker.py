"""T6 审计：M7 约束校验器对已知 LLM 失效模式的拦截率。

为什么需要这个脚本：本机没有 GPU，跑不了真实模型，因此无法测「接上 LLM 之后准确率多少」。
但校验器**能不能拦住不合规输出**是一个独立于模型的性质，可以现在就量化。

做法是对每个 N5c case 构造 7 种回答：6 种对应已知的 LLM 失效模式，1 种是合规回答。
校验器必须拦住前 6 种、放行第 7 种。**放行率同样重要**——
一个把合规回答也拦下来的校验器会让系统陷入无限重写，比不校验更糟。

失效模式取自 legacy 阶段观察到的实际问题与 LLM 的常见毛病：

| 模式 | 说明 | 应由谁拦住 |
| --- | --- | --- |
| `fabricated_evidence` | 引用证据清单里没有的 token | M7 幻觉检测 |
| `fabricated_constraint` | 编造约束编号 | M7 幻觉检测 |
| `unsupported_step` | 只下结论不给依据 | M7 凭空断言检测 |
| `absolute_loss_claim` | 给出「链路损耗 x dB」 | C12 |
| `serdes_db_claim` | 把 serdes_snr 当 dB 讨论 | C13 |
| `host_snr_normal_claim` | 把未采集的 host_snr 说成正常 | C14 |
| `compliant` | 合规回答 | 不该被拦 |
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.branches.general import build_request  # noqa: E402
from rca_framework.constraints.checker import check_response  # noqa: E402
from rca_framework.data import load_cases  # noqa: E402
from rca_framework.evidence_graph import COVERAGE_POLICY, EvidenceGraph, match_many, route  # noqa: E402
from rca_framework.evidence_pack import build_packs, labels_of  # noqa: E402
from rca_framework.features.extractor import extract_features, fit_feature_model  # noqa: E402
from rca_framework.llm.protocol import parse_response  # noqa: E402


def make_response(mode: str, request: Any) -> str:
    tokens = list(request.evidence_tokens)
    real = tokens[:1] or ["level:L2:rxpower_mean:low_tail"]
    verdict = request.candidate_root_causes[0]
    step = {
        "claim": "当前 evidence token 作为该候选的支持线索",
        "cited_evidence": real,
        # 合规基线不强行引用约束：首个 token 可能属于 media_snr、
        # serdes 或 lane_direction，固定写 C7 会构成约束/token 错配。
        "cited_constraints": [],
        "effect": "support",
        "target": verdict,
    }

    if mode == "fabricated_evidence":
        step["cited_evidence"] = real + ["drop:L7:phantom_metric:all_lanes"]
    elif mode == "fabricated_constraint":
        step["cited_constraints"] = ["C42_invented_rule"]
    elif mode == "unsupported_step":
        step["cited_evidence"] = []
        step["cited_constraints"] = []
        step["claim"] = "综合来看应该是这一侧的问题"
    elif mode == "absolute_loss_claim":
        step["claim"] = "两端功率相减得到链路损耗约 4.7 dB，明显超标"
    elif mode == "serdes_db_claim":
        step["claim"] = "serdes_snr 只有 3.5 dB，信噪比严重不足"
    elif mode == "host_snr_normal_claim":
        step["claim"] = "host_snr 正常，因此电口侧没有问题"

    return json.dumps({
        "steps": [step],
        "verdict": verdict,
        "confidence": 0.75,
        "missing_information": [],
    }, ensure_ascii=False)


MODES = (
    "fabricated_evidence",
    "fabricated_constraint",
    "unsupported_step",
    "absolute_loss_claim",
    "serdes_db_claim",
    "host_snr_normal_claim",
    "compliant",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/organized_rca_v2_stratified_60_40_seed42"))
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.data_dir)
    train, test = cases[: args.train_size], cases[args.train_size :]
    thresholds = fit_thresholds(train)
    train_packs, test_packs = build_packs(train), build_packs(test)
    model = fit_feature_model(train_packs)
    train_features = [extract_features(pack, thresholds, model) for pack in train_packs]
    test_features = [extract_features(pack, thresholds, model) for pack in test_packs]
    graph = EvidenceGraph.build(train_features, labels_of(train), feature_model=model)
    results = match_many(graph, test_features, top_k=0)

    targets = [
        (result, pack) for result, pack in zip(results, test_packs)
        if route(result, COVERAGE_POLICY).branch == "N5c" and result.query_tokens
    ]
    print(f"N5c 有效证据 case：{len(targets)} 条\n")

    rows: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        blocked = 0
        kinds: Counter = Counter()
        for result, pack in targets:
            request = build_request(result, pack)
            response = parse_response(make_response(mode, request))
            report = check_response(
                response, pack, request.evidence_tokens,
                allowed_root_causes=request.candidate_root_causes,
            )
            if not report.ok:
                blocked += 1
                kinds.update(item.kind for item in report.fatal)
        rows[mode] = {
            "total": len(targets),
            "blocked": blocked,
            "block_rate": round(blocked / len(targets), 6) if targets else 0.0,
            "violation_kinds": dict(kinds),
        }

    print(f"{'失效模式':<26} {'应拦截':>6} {'实际拦截':>8} {'拦截率':>8}")
    for mode in MODES:
        row = rows[mode]
        expected = "否" if mode == "compliant" else "是"
        print(f"{mode:<26} {expected:>6} {row['blocked']:>5}/{row['total']:<3} {row['block_rate']:>8.2%}")
    print()

    failures = [
        mode for mode in MODES
        if (mode == "compliant" and rows[mode]["blocked"] > 0)
        or (mode != "compliant" and rows[mode]["blocked"] < rows[mode]["total"])
    ]
    if failures:
        print(f"未达标的模式：{failures}")
    else:
        print("全部达标：6 类违规全拦，合规回答全放行。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"n5c_cases": len(targets), "modes": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
