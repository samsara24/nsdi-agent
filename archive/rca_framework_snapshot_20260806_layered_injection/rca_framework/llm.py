from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .types import CaseEvidence, ROOT_CAUSES, normalize_scores, rank_scores


LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction": {"type": "string", "enum": list(ROOT_CAUSES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "path_ids": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prediction", "confidence", "path_ids", "reasoning", "missing_information"],
    "additionalProperties": False,
}

SUFFICIENCY_VALUES = ("sufficient", "insufficient")

# The layered schema adds one field so the model can separate "this is my best
# guess" from "the current telemetry supports this".  `prediction` stays
# mandatory and three-way so accuracy remains comparable to earlier runs.
LAYERED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction": {"type": "string", "enum": list(ROOT_CAUSES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_sufficiency": {"type": "string", "enum": list(SUFFICIENCY_VALUES)},
        "path_ids": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "prediction", "confidence", "evidence_sufficiency",
        "path_ids", "reasoning", "missing_information",
    ],
    "additionalProperties": False,
}

INJECTION_MODES = ("full", "layered")
SCORE_MODES = ("legacy", "llm_only")

ROOT_CAUSE_DEFINITIONS = {
    "L1": "400G 端口或该端设备侧根因",
    "L2": "200G 端口或该端设备侧根因",
    "fiber": "L1 与 L2 之间的光纤/链路介质根因",
}


def classify_kg_coverage(case: CaseEvidence, graph_result: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether the KG actually covers this case's anomaly combination.

    The regime is defined purely by the structure of the KG response, so it is
    reproducible and introduces no tuned threshold:

    - ``covered``: at least one learned KG feature rule matches, so the KG has
      seen this combination and its aggregated score is meaningful.
    - ``partial``: individual anomaly edges fire but no rule matches, so only
      atom-level statistics are transferable.
    - ``uncovered``: nothing fires, and the aggregated KG score degenerates to
      the training class prior.
    """
    paths = graph_result.get("paths") or []
    matched_rules = graph_result.get("matched_feature_rules") or {}
    matched_rule_count = sum(len(items) for items in matched_rules.values())
    retrieved = graph_result.get("retrieved_cases") or []
    max_similarity = max((float(row.get("similarity", 0.0)) for row in retrieved), default=0.0)
    path_count = int(graph_result.get("path_count", len(paths)))
    if matched_rule_count > 0:
        regime = "covered"
    elif path_count > 0:
        regime = "partial"
    else:
        regime = "uncovered"
    return {
        "regime": regime,
        "anomaly_count": len(case.anomalies),
        "path_count": path_count,
        "matched_rule_count": matched_rule_count,
        "max_retrieval_similarity": round(max_similarity, 8),
    }


_REGIME_GUIDANCE = {
    "covered": (
        "知识图谱已覆盖该异常组合：至少一条学到的 KG 规则命中。"
        "可以把匹配规则和候选路径分数作为主要依据。"
    ),
    "partial": (
        "知识图谱只覆盖到单个异常原子，没有覆盖当前这个组合。"
        "已屏蔽 KG 的聚合候选分数和 KG 结论，因为在组合未覆盖时该分数主要反映训练集类别先验。"
        "请只把 root_cause_paths 里的 training_count、precision、lift 当作原子级证据强度，"
        "组合层面的因果推理由你自己完成。"
    ),
    "uncovered": (
        "知识图谱对该 case 没有任何结构证据：没有异常节点命中，也没有规则匹配。"
        "已屏蔽 KG 的候选分数和 KG 结论，因为在无证据时它等于训练集类别先验，会把结论推向多数类。"
        "请只依据 target_case 的原始遥测和根因物理定义推理，"
        "并把 evidence_sufficiency 设为 insufficient，同时在 missing_information 中列出需要补采的测量。"
    ),
}

_SHARED_CONSTRAINTS = (
    "目标 case 的真实标签不可见，不得猜测或补造。",
    "正常行为不会形成知识图谱边。",
    "prediction 必须在 L1、L2、fiber 中选择一个，用于与既有评估口径对齐。",
    "若现有证据不足以支撑该选择，必须把 evidence_sufficiency 设为 insufficient，而不是抬高 confidence。",
)

_UNCERTAIN_CONSTRAINTS = (
    "字段缺失只代表未采集，不得当作该项正常。",
    "不得因为某一类在历史数据中更常见而倾向该类；类别先验未提供，也不应被推测。",
)


def build_path_prompt(case: CaseEvidence, graph_result: Dict[str, Any]) -> str:
    payload = {
        "task": "三分类光链路根因定界",
        "root_cause_definitions": {
            "L1": "400G 端口或该端设备侧根因",
            "L2": "200G 端口或该端设备侧根因",
            "fiber": "L1 与 L2 之间的光纤/链路介质根因",
        },
        "constraints": [
            "目标 case 的真实标签不可见，不得猜测或补造。",
            "只依据异常节点、图路径和训练集检索案例推理。",
            "正常行为不会形成知识图谱边。",
            "必须在 L1、L2、fiber 中选择一个结果。",
        ],
        "target_case": {
            "case_id": case.case_id,
            "summary": case.summary,
            "anomalies": [item.to_dict() for item in case.anomalies],
            "data_coverage": case.coverage,
            "missing_fields": case.missing_fields,
        },
        "candidate_path_scores": graph_result["scores"],
        "root_cause_paths": graph_result["paths"],
        "candidate_feature_profile_scores": graph_result.get("feature_profile_scores", {}),
        "matched_kg_feature_rules": graph_result.get("matched_feature_rules", {}),
        "retrieved_training_cases": graph_result["retrieved_cases"],
        "output_schema": {
            "prediction": "L1 | L2 | fiber",
            "confidence": "0 到 1",
            "path_ids": ["使用的 anomaly_id"],
            "reasoning": "基于路径的简短因果解释",
            "missing_information": ["会影响定界的缺失信息"],
        },
    }
    return "你是光链路 RCA 专家。只输出一个合法 JSON 对象。\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_layered_prompt(case: CaseEvidence, graph_result: Dict[str, Any]) -> str:
    """Inject only the KG evidence that is actually valid for this case.

    Under ``uncovered`` and ``partial`` regimes the aggregated KG score carries
    almost no case-specific signal and mostly restates the training class
    prior, so it is withheld instead of presented as a candidate distribution.
    """
    coverage = classify_kg_coverage(case, graph_result)
    regime = coverage["regime"]
    constraints = list(_SHARED_CONSTRAINTS)
    if regime == "covered":
        constraints.append("优先使用训练统计强度高的匹配规则和候选路径。")
    else:
        constraints.extend(_UNCERTAIN_CONSTRAINTS)

    payload: Dict[str, Any] = {
        "task": "三分类光链路根因定界",
        "root_cause_definitions": ROOT_CAUSE_DEFINITIONS,
        "kg_coverage": coverage,
        "kg_coverage_guidance": _REGIME_GUIDANCE[regime],
        "constraints": constraints,
        "target_case": {
            "case_id": case.case_id,
            "summary": case.summary,
            "anomalies": [item.to_dict() for item in case.anomalies],
            "data_coverage": case.coverage,
            "missing_fields": case.missing_fields,
        },
    }

    withheld: list[str] = []
    if regime == "covered":
        payload["candidate_path_scores"] = graph_result["scores"]
        payload["candidate_feature_profile_scores"] = graph_result.get("feature_profile_scores", {})
        payload["matched_kg_feature_rules"] = graph_result.get("matched_feature_rules", {})
        payload["root_cause_paths"] = graph_result["paths"]
    elif regime == "partial":
        payload["root_cause_paths"] = graph_result["paths"]
        withheld.append("candidate_path_scores：组合未被 KG 覆盖，聚合分数主要反映类别先验。")
        withheld.append("candidate_feature_profile_scores 与 matched_kg_feature_rules：本 case 无规则命中。")
    else:
        withheld.append("candidate_path_scores：无任何异常节点命中，聚合分数等于类别先验。")
        withheld.append("root_cause_paths 与 matched_kg_feature_rules：本 case 为空。")
    if withheld:
        payload["withheld_kg_fields"] = withheld

    retrieved = graph_result.get("retrieved_cases") or []
    payload["retrieved_training_cases"] = retrieved
    if regime != "covered":
        payload["retrieval_note"] = (
            f"检索到的训练 case 最高相似度仅 {coverage['max_retrieval_similarity']}，"
            "它们是当前训练集中最接近的样本，不代表同类，不得直接套用其标签。"
        )

    payload["output_schema"] = {
        "prediction": "L1 | L2 | fiber",
        "confidence": "0 到 1，表示你对该选择的把握",
        "evidence_sufficiency": "sufficient | insufficient，表示现有遥测是否足以支撑定界",
        "path_ids": ["使用的 anomaly_id，必须来自 target_case.anomalies"],
        "reasoning": "简短因果解释",
        "missing_information": ["需要补采、会改变结论的测量"],
    }
    return "你是光链路 RCA 专家。只输出一个合法 JSON 对象。\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    prediction = str(value.get("prediction", "")).strip()
    if prediction not in ROOT_CAUSES:
        return None
    try:
        confidence = min(1.0, max(0.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    sufficiency = str(value.get("evidence_sufficiency", "")).strip()
    return {
        "prediction": prediction,
        "confidence": confidence,
        "evidence_sufficiency": sufficiency if sufficiency in SUFFICIENCY_VALUES else "unreported",
        "path_ids": list(value.get("path_ids", [])),
        "reasoning": str(value.get("reasoning", "")),
        "missing_information": list(value.get("missing_information", [])),
        "raw": value,
    }


class PathLLMReasoner:
    def __init__(
        self,
        backend: str = "none",
        model_path: str = "",
        max_new_tokens: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        dtype: str = "auto",
        enforce_eager: bool = False,
        guided_json: bool = True,
        disable_custom_all_reduce: bool = False,
        injection_mode: str = "layered",
        score_mode: str = "llm_only",
        insufficient_confidence_scale: float = 1.0,
    ) -> None:
        self.backend = backend
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.guided_json = guided_json
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.injection_mode = injection_mode
        self.score_mode = score_mode
        self.insufficient_confidence_scale = insufficient_confidence_scale
        self._validate_modes()
        self._model: Any = None
        self._tokenizer: Any = None

    def _validate_modes(self) -> None:
        if self.injection_mode not in INJECTION_MODES:
            raise ValueError(f"unsupported injection mode: {self.injection_mode}")
        if self.score_mode not in SCORE_MODES:
            raise ValueError(f"unsupported score mode: {self.score_mode}")

    def configure(self, **overrides: Any) -> "PathLLMReasoner":
        """Change per-request settings without discarding the loaded model.

        Only reasoning-time settings may be changed here; anything that affects
        model loading must go through a new instance.
        """
        allowed = {"max_new_tokens", "guided_json", "injection_mode", "score_mode", "insufficient_confidence_scale"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"cannot reconfigure loading-time settings: {sorted(unknown)}")
        for key, value in overrides.items():
            setattr(self, key, value)
        self._validate_modes()
        return self

    @property
    def output_schema(self) -> Dict[str, Any]:
        return LLM_OUTPUT_SCHEMA if self.injection_mode == "full" else LAYERED_OUTPUT_SCHEMA

    def build_prompt(self, case: CaseEvidence, graph_result: Dict[str, Any]) -> str:
        if self.injection_mode == "full":
            return build_path_prompt(case, graph_result)
        return build_layered_prompt(case, graph_result)

    def reason(self, case: CaseEvidence, graph_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.reason_many([case], [graph_result])[0]

    def reason_many(self, cases: list[CaseEvidence], graph_results: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if len(cases) != len(graph_results):
            raise ValueError("cases and graph_results must have equal length")
        prompts = [self.build_prompt(case, graph_result) for case, graph_result in zip(cases, graph_results)]
        coverages = [classify_kg_coverage(case, graph_result) for case, graph_result in zip(cases, graph_results)]
        if self.backend == "none":
            return [
                self._fallback(graph_result, prompt, coverage, "LLM disabled; deterministic graph-path reasoning used")
                for graph_result, prompt, coverage in zip(graph_results, prompts, coverages)
            ]
        raw_outputs = self._generate_many(prompts)
        return [
            self._parse_or_fallback(graph_result, prompt, coverage, raw)
            for graph_result, prompt, coverage, raw in zip(graph_results, prompts, coverages, raw_outputs)
        ]

    def _score_llm_result(self, graph_result: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, float]:
        prediction, confidence = parsed["prediction"], parsed["confidence"]
        if self.score_mode == "legacy":
            graph_scores = normalize_scores(graph_result["scores"])
            scores = {label: graph_scores[label] * 0.35 for label in ROOT_CAUSES}
            scores[prediction] += 0.65 * confidence
            return normalize_scores(scores)
        # Read the self-reported confidence as an interpolation coefficient
        # between a uniform distribution and a one-hot vote, so the predicted
        # class always stays the argmax and confidence 0 degrades to uniform.
        # This keeps the LLM route independent of the KG score, which fusion
        # already weighs separately.
        uniform = 1.0 / len(ROOT_CAUSES)
        residual = (1.0 - confidence) * uniform
        scores = {label: residual for label in ROOT_CAUSES}
        scores[prediction] += confidence
        return normalize_scores(scores)

    def _parse_or_fallback(
        self, graph_result: Dict[str, Any], prompt: str, coverage: Dict[str, Any], raw: str,
    ) -> Dict[str, Any]:
        parsed = parse_llm_json(raw)
        if parsed is None:
            return self._fallback(
                graph_result, prompt, coverage,
                "invalid LLM JSON; deterministic graph-path reasoning used", raw,
            )
        reported_confidence = parsed["confidence"]
        fusion_confidence = reported_confidence
        if parsed["evidence_sufficiency"] == "insufficient":
            fusion_confidence = reported_confidence * self.insufficient_confidence_scale
        parsed.update({
            "scores": self._score_llm_result(graph_result, parsed),
            "reported_confidence": reported_confidence,
            "confidence": fusion_confidence,
            "prompt": prompt,
            "raw_output": raw,
            "reasoning_mode": "llm_path_reasoning",
            "graph_prediction": graph_result["prediction"],
            "kg_coverage": coverage,
            "injection_mode": self.injection_mode,
            "score_mode": self.score_mode,
        })
        return parsed

    def _fallback(
        self, graph_result: Dict[str, Any], prompt: str, coverage: Dict[str, Any], note: str, raw: str = "",
    ) -> Dict[str, Any]:
        ranking = rank_scores(graph_result["scores"])
        used_paths = [path["anomaly_id"] for path in graph_result["paths"] if path["root_cause"] == ranking[0][0]][:5]
        return {
            "prediction": graph_result["prediction"],
            "confidence": graph_result["confidence"],
            "reported_confidence": graph_result["confidence"],
            "evidence_sufficiency": "insufficient" if coverage["regime"] == "uncovered" else "unreported",
            "scores": graph_result["scores"],
            "path_ids": used_paths,
            "reasoning": note,
            "missing_information": [],
            "prompt": prompt,
            "raw_output": raw,
            "reasoning_mode": "deterministic_path_fallback",
            "graph_prediction": graph_result["prediction"],
            "kg_coverage": coverage,
            "injection_mode": self.injection_mode,
            "score_mode": self.score_mode,
        }

    def _render_prompt(self, prompt: str) -> str:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        template = getattr(self._tokenizer, "chat_template", "") or ""
        if "<｜User｜>" in template:
            return f"{self._tokenizer.bos_token or ''}<｜User｜>{prompt}<｜Assistant｜>"
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            )
        return prompt

    def _generate_many(self, prompts: list[str]) -> list[str]:
        if not self.model_path:
            raise ValueError("--model-path is required when LLM backend is enabled")
        if self.backend == "vllm":
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import GuidedDecodingParams

            if self._model is None:
                self._model = LLM(
                    model=self.model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    dtype=self.dtype,
                    enforce_eager=self.enforce_eager,
                    disable_custom_all_reduce=self.disable_custom_all_reduce,
                )
            rendered = [self._render_prompt(prompt) for prompt in prompts]
            guided = GuidedDecodingParams(json=self.output_schema) if self.guided_json else None
            outputs = self._model.generate(
                rendered,
                SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens, guided_decoding=guided),
            )
            return [output.outputs[0].text for output in outputs]
        if self.backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self._model is None:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
            outputs = []
            for prompt in prompts:
                rendered = self._render_prompt(prompt)
                inputs = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
                with torch.no_grad():
                    generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                outputs.append(self._tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
            return outputs
        raise ValueError(f"unsupported LLM backend: {self.backend}")
