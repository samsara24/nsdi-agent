"""M8 推理后端。

后端只负责「给一批 prompt，返回一批文本」，不理解 RCA 语义。
这样约束校验、重写循环、弃权策略全部与后端无关，换模型不需要改推理逻辑。

`ScriptedBackend` 不是玩具：约束校验器和重写循环的行为必须能在没有 GPU 的环境里
逐条测出来，否则「不合规可回退或重写」这条验收无法验证。用真实模型跑测试既慢又不确定。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .protocol import DIAGNOSIS_OUTPUT_SCHEMA


CONTEXT_SAFETY_TOKENS = 32


def prompt_token_lengths(prompts: Sequence[str], tokenizer: Any) -> List[int]:
    """Count the exact rendered prompt tokens without loading model weights."""
    return [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]


def validate_context_window(
    prompts: Sequence[str],
    tokenizer: Any,
    *,
    max_model_len: int,
    max_new_tokens: int,
) -> List[int]:
    """Fail before model allocation when prompt + output cannot fit the window."""
    lengths = prompt_token_lengths(prompts, tokenizer)
    if not lengths:
        return lengths
    longest = max(lengths)
    required = longest + max_new_tokens + CONTEXT_SAFETY_TOKENS
    if required > max_model_len:
        index = lengths.index(longest)
        raise ValueError(
            "prompt context preflight failed: "
            f"request_index={index}, prompt_tokens={longest}, "
            f"max_new_tokens={max_new_tokens}, safety_tokens={CONTEXT_SAFETY_TOKENS}, "
            f"required_max_model_len>={required}, configured_max_model_len={max_model_len}"
        )
    return lengths


class Backend:
    """后端协议。`name` 会进 `run_manifest.json`。"""

    name: str = "base"

    def generate(self, prompts: Sequence[str]) -> List[str]:
        raise NotImplementedError

    def close(self) -> None:
        """Release optional backend resources. CPU-only backends are no-ops."""


class NoneBackend(Backend):
    """不调用任何模型。返回空串，上层按「解析失败」处理并最终弃权。

    它是默认后端：没有配置模型时，系统应当弃权而不是退回类别先验。
    """

    name = "none"

    def generate(self, prompts: Sequence[str]) -> List[str]:
        return ["" for _ in prompts]


@dataclass
class ScriptedBackend(Backend):
    """按脚本返回预设输出，用于测试重写循环与约束校验。

    `responses` 是一个列表的列表：第 i 次调用 `generate` 时返回 `responses[i]`。
    这样可以精确构造「第一轮违规、第二轮修好」这种场景。
    """

    responses: List[List[str]] = field(default_factory=list)
    name: str = "scripted"
    calls: int = 0
    prompts_seen: List[List[str]] = field(default_factory=list)

    def generate(self, prompts: Sequence[str]) -> List[str]:
        self.prompts_seen.append(list(prompts))
        if self.calls >= len(self.responses):
            outputs = ["" for _ in prompts]
        else:
            batch = self.responses[self.calls]
            outputs = [batch[index] if index < len(batch) else "" for index in range(len(prompts))]
        self.calls += 1
        return outputs


@dataclass
class VLLMBackend(Backend):
    """vLLM 后端。参数与 legacy `PathLLMReasoner` 保持一致，
    这样两条路径可以用同一份模型配置做对照实验。

    `guided_json` 保留为通用后端开关；要求单次调用的正式实验必须开启它，确保
    模型在一次生成内直接返回协议 JSON，而不是只输出 `<think>` 后提前结束。
    """

    model_path: str = ""
    max_new_tokens: int = 1024
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    dtype: str = "auto"
    enforce_eager: bool = False
    guided_json: bool = False
    disable_custom_all_reduce: bool = False
    temperature: float = 0.0
    seed: int = 42
    name: str = "vllm"
    _model: Any = None
    _tokenizer: Any = None

    def _render(self, prompt: str) -> str:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError):
                # Older remote tokenizers occasionally expose the method but no
                # usable template. Fall back only after the canonical renderer fails.
                pass
        template = getattr(self._tokenizer, "chat_template", "") or ""
        if "<｜User｜>" in template:
            return f"{self._tokenizer.bos_token or ''}<｜User｜>{prompt}<｜Assistant｜>"
        return prompt

    def _sampling_params(self) -> Any:
        """兼容 vLLM 0.11 的 `structured_outputs` 与更早版本的 `guided_decoding`。"""
        from vllm import SamplingParams

        common = {
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
        }
        if not self.guided_json:
            return SamplingParams(**common)
        try:
            from vllm.sampling_params import StructuredOutputsParams

            return SamplingParams(
                structured_outputs=StructuredOutputsParams(json=DIAGNOSIS_OUTPUT_SCHEMA), **common
            )
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams

            return SamplingParams(
                guided_decoding=GuidedDecodingParams(json=DIAGNOSIS_OUTPUT_SCHEMA), **common
            )

    def generate(self, prompts: Sequence[str]) -> List[str]:
        if not self.model_path:
            raise ValueError("model_path is required for the vllm backend")
        rendered = [self._render(prompt) for prompt in prompts]
        validate_context_window(
            rendered,
            self._tokenizer,
            max_model_len=self.max_model_len,
            max_new_tokens=self.max_new_tokens,
        )
        from vllm import LLM

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
        outputs = self._model.generate(
            rendered, self._sampling_params()
        )
        return [output.outputs[0].text for output in outputs]

    def close(self) -> None:
        """Best-effort deterministic teardown of vLLM and CUDA allocations."""
        model = self._model
        self._model = None
        self._tokenizer = None
        if model is not None:
            shutdown = getattr(model, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
            del model
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def backend_for(name: str, **kwargs: Any) -> Backend:
    if name == "none":
        return NoneBackend()
    if name == "vllm":
        return VLLMBackend(**kwargs)
    if name == "scripted":
        return ScriptedBackend(**kwargs)
    raise ValueError(f"unsupported LLM backend: {name}")
