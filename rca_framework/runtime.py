from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Mapping


_COERCERS: Dict[str, Callable[[Any], Any]] = {
    "llm_backend": str,
    "model_path": str,
    "max_new_tokens": int,
    "tensor_parallel_size": int,
    "gpu_memory_utilization": float,
    "max_model_len": int,
    "dtype": str,
    "enforce_eager": bool,
    "guided_json": bool,
    "disable_custom_all_reduce": bool,
}


@dataclass(frozen=True)
class RuntimeConfig:
    """推理期运行参数。默认值必须与 legacy CLI 默认值逐项一致。

    与 `PipelineConfig` 的区别：`PipelineConfig` 参与 `model.json` 并影响学到的知识，
    本类只影响一次推理如何执行，不进入模型产物。
    """

    llm_backend: str = "none"
    model_path: str = ""
    max_new_tokens: int = 512
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    dtype: str = "auto"
    enforce_eager: bool = False
    guided_json: bool = True
    disable_custom_all_reduce: bool = False

    @classmethod
    def from_kwargs(cls, values: Mapping[str, Any]) -> "RuntimeConfig":
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise TypeError(f"unsupported runtime settings: {unknown}")
        return cls(**{name: _COERCERS[name](value) for name, value in values.items()})

    def to_dict(self) -> Dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}
