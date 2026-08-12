"""证据 token 的语义归约：把不同异常判据族映射到同一个「侧 + 指标」坐标。

迭代 3 的实测动机（见 Progress.md §9.33）：477 条 fatal 违规里 298 条是同一件事——
模型引用 `level:L1:txpower_mean:low_tail` 去支撑一条契约写着 `expert:L1:txpower:`
的约束。两个 token 说的是**同一侧的同一个指标**，只是一个用数据集分位数判异常、
另一个用工程阈值判异常。校验器按前缀字符串比对，于是把物理上正确的一步判废。

这里的等价类只放宽「哪一族判据认定了异常」，`(side, metric)` 必须完全一致。
这一点是安全性的全部依据：C23-C25 这类方向约束的正确性只取决于
「症状出现在哪一侧的哪个指标」，与用哪套阈值发现它无关；而 `allowed_targets`
仍然独立地把方向锁死。因此放宽 token 族不会放宽任何物理断言。

`side=None` 表示该族不区分两端（例如 `serdes:` 契约），按通配处理。
`metric=None` 表示 token 是侧级别的整体判断（`expert:pattern:*`、`expert:points_to:*`），
它们不参与指标等价，只能靠字面前缀匹配。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

Scope = Tuple[Optional[str], Optional[str]]

#: `level` 家族在指标名后追加统计量后缀，归约时剥掉。
_LEVEL_SUFFIXES: Tuple[str, ...] = ("_mean", "_min", "_max", "_p10", "_p90")

#: 端口状态位度量的是哪条光路。RxLOL/RxLOS 是接收侧现象，TxLOS 是发送侧现象。
_STATUS_METRIC = {
    "RxLOL": "rxpower",
    "RxLOS": "rxpower",
    "TxLOS": "txpower",
    "TxFault": "txpower",
}

_SIDES = ("L1", "L2")


def _strip_level_suffix(metric: str) -> str:
    for suffix in _LEVEL_SUFFIXES:
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric


def token_scope(token: str) -> Optional[Scope]:
    """把一个证据 token 归约到 `(side, metric)`；无法归约时返回 None。"""
    parts = token.split(":")
    if not parts:
        return None
    family = parts[0]

    if family in ("drop", "expert") and len(parts) >= 3 and parts[1] in _SIDES:
        # drop:L1:rxpower:single_lane / expert:L1:rxpower:lane_down
        return (parts[1], parts[2])
    if family == "expert" and len(parts) >= 3 and parts[1] in ("pattern", "points_to"):
        # 侧级别整体判断，没有单一指标。
        return (parts[2] if len(parts) > 2 and parts[2] in _SIDES else None, None)
    if family == "level" and len(parts) >= 3 and parts[1] in _SIDES:
        return (parts[1], _strip_level_suffix(parts[2]))
    if family == "imbalance" and len(parts) >= 3 and parts[1] in _SIDES:
        return (parts[1], parts[2])
    if family == "serdes" and len(parts) >= 2 and parts[1] in _SIDES:
        # serdes:L1:valid 描述的是该侧 SerDes 的状态，与 serdes_snr 同坐标。
        return (parts[1], "serdes_snr")
    if family == "status" and len(parts) >= 3 and parts[1] in _SIDES:
        metric = _STATUS_METRIC.get(parts[2])
        return (parts[1], metric) if metric else None
    return None


def prefix_scope(prefix: str) -> Optional[Scope]:
    """把约束契约里的前缀归约到同一坐标系。

    契约前缀通常以 `:` 结尾（`expert:L1:txpower:`），也可能是完整 token
    （`expert:pattern:L1:port_down`）。裸家族名（`serdes:`）视为该指标的通配。
    """
    trimmed = prefix.rstrip(":")
    parts = trimmed.split(":")
    if len(parts) == 1:
        # 裸家族前缀：只有 serdes 能确定指标，其余交给字面匹配。
        return (None, "serdes_snr") if parts[0] == "serdes" else None
    return token_scope(trimmed)


def matches_scope(token: str, prefixes: Sequence[str]) -> bool:
    """token 是否落在契约的适用范围内：字面前缀命中，或语义坐标相同。"""
    if not prefixes:
        return False
    if token.startswith(tuple(prefixes)):
        return True
    scope = token_scope(token)
    if scope is None or scope[1] is None:
        # 侧级别 token 与无法归约的 token 一律只认字面前缀，避免过度放宽。
        return False
    for prefix in prefixes:
        other = prefix_scope(prefix)
        if other is None or other[1] is None:
            continue
        if other[1] != scope[1]:
            continue
        if other[0] is None or other[0] == scope[0]:
            return True
    return False
