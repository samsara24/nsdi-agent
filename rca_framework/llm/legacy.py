from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from ..types import CaseEvidence, ROOT_CAUSES, normalize_scores, rank_scores


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
    return {
        "prediction": prediction,
        "confidence": confidence,
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
        self._model: Any = None
        self._tokenizer: Any = None

    def reason(self, case: CaseEvidence, graph_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.reason_many([case], [graph_result])[0]

    def reason_many(self, cases: list[CaseEvidence], graph_results: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if len(cases) != len(graph_results):
            raise ValueError("cases and graph_results must have equal length")
        prompts = [build_path_prompt(case, graph_result) for case, graph_result in zip(cases, graph_results)]
        if self.backend == "none":
            return [
                self._fallback(graph_result, prompt, "LLM disabled; deterministic graph-path reasoning used")
                for graph_result, prompt in zip(graph_results, prompts)
            ]
        raw_outputs = self._generate_many(prompts)
        return [
            self._parse_or_fallback(graph_result, prompt, raw)
            for graph_result, prompt, raw in zip(graph_results, prompts, raw_outputs)
        ]

    def _parse_or_fallback(self, graph_result: Dict[str, Any], prompt: str, raw: str) -> Dict[str, Any]:
        parsed = parse_llm_json(raw)
        if parsed is None:
            return self._fallback(graph_result, prompt, "invalid LLM JSON; deterministic graph-path reasoning used", raw)
        graph_scores = normalize_scores(graph_result["scores"])
        llm_scores = {label: graph_scores[label] * 0.35 for label in ROOT_CAUSES}
        llm_scores[parsed["prediction"]] += 0.65 * parsed["confidence"]
        parsed.update({
            "scores": normalize_scores(llm_scores),
            "prompt": prompt,
            "raw_output": raw,
            "reasoning_mode": "llm_path_reasoning",
            "graph_prediction": graph_result["prediction"],
        })
        return parsed

    def _fallback(self, graph_result: Dict[str, Any], prompt: str, note: str, raw: str = "") -> Dict[str, Any]:
        ranking = rank_scores(graph_result["scores"])
        used_paths = [path["anomaly_id"] for path in graph_result["paths"] if path["root_cause"] == ranking[0][0]][:5]
        return {
            "prediction": graph_result["prediction"],
            "confidence": graph_result["confidence"],
            "scores": graph_result["scores"],
            "path_ids": used_paths,
            "reasoning": note,
            "missing_information": [],
            "prompt": prompt,
            "raw_output": raw,
            "reasoning_mode": "deterministic_path_fallback",
            "graph_prediction": graph_result["prediction"],
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
            guided = GuidedDecodingParams(json=LLM_OUTPUT_SCHEMA) if self.guided_json else None
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
