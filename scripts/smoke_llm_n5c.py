"""T6 冒烟：在少量 N5c case 上跑真实模型，打印原始输出与校验结果。

全量评估之前必须先跑这个。要看的不是准确率，而是三件事：

1. 模型有没有按 schema 输出，`parse_response` 能不能解析出来。
   推理型模型会先输出 `<think>` 段，解析器必须取思考之后的最终答案。
2. 校验器判了什么。如果大面积报同一类违规，多半是 prompt 没说清楚而不是模型不行。
3. 模型会不会用 `abstain`。这是 T6 最关心的行为问题：
   如果它从不弃权，那「不硬猜」这条设计就只在架构上成立、在行为上落空。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.branches.general import build_request  # noqa: E402
from rca_framework.data import load_cases  # noqa: E402
from rca_framework.evidence_graph import COVERAGE_POLICY, EvidenceGraph, match_many, route  # noqa: E402
from rca_framework.evidence_pack import build_packs, labels_of  # noqa: E402
from rca_framework.features.extractor import extract_features, fit_feature_model  # noqa: E402
from rca_framework.llm import ConstrainedReasoner, VLLMBackend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/organized_rca_v2_stratified_60_40_seed42"))
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="跳过 CUDA graph 捕获，启动更快，逐 token 略慢。")
    parser.add_argument("--disable-custom-all-reduce", action="store_true",
                        help="无 NVLink 的多卡机器上建议开启，改走标准 NCCL。")
    parser.add_argument("--guided-json", action="store_true",
                        help="强制 JSON 结构化解码。推理型模型不要开，会禁掉 <think> 段。")
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
    truth = dict(zip((item.case_id for item in test_packs), labels_of(test)))

    targets = [
        (result, pack) for result, pack in zip(results, test_packs)
        if route(result, COVERAGE_POLICY).branch == "N5c" and result.query_tokens
    ][: args.limit]
    requests = [build_request(result, pack) for result, pack in targets]
    packs = [pack for _, pack in targets]

    reasoner = ConstrainedReasoner(
        backend=VLLMBackend(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            guided_json=args.guided_json,
            enforce_eager=args.enforce_eager,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
        ),
        max_attempts=args.max_attempts,
    )
    traces = reasoner.reason_many(requests, packs)

    payload = []
    for request, trace in zip(requests, traces):
        print("=" * 100)
        print(f"case {request.case_id}  真值={truth.get(request.case_id)}  "
              f"可选根因={request.candidate_root_causes}")
        print(f"证据 ({len(request.evidence_tokens)}): {list(request.evidence_tokens)}")
        for attempt in trace.attempts:
            raw = attempt.raw_output
            print(f"\n--- 第 {attempt.index + 1} 轮  输出 {len(raw)} 字符  "
                  f"解析={'成功' if attempt.parsed else '失败'}  "
                  f"校验={'通过' if attempt.check.ok else '不通过'} ---")
            print(f"[原始输出前 700 字]\n{raw[:700]}")
            if len(raw) > 700:
                print(f"[原始输出末 500 字]\n{raw[-500:]}")
            for violation in attempt.check.fatal:
                print(f"  违规: [{violation.kind}] {violation.message}")
        accepted = trace.accepted
        print(f"\n结论: {accepted.verdict if accepted else None}   "
              f"（真值 {truth.get(request.case_id)}）")
        if not accepted:
            print(f"弃权原因: {trace.abstain_reason}")
        payload.append({"truth": truth.get(request.case_id), "trace": trace.to_dict()})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
