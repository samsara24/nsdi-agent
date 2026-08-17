"""Measurement contracts for the current RCA telemetry.

Measurement contracts are not physical evidence for a root cause.  They are
veto rules: if a reasoning step violates one, the conclusion is not trustworthy
for this telemetry snapshot and should become evidence request or abstention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple


MEASUREMENT_CONTRACT_LIBRARY_VERSION = "measurement-contracts-v1"
CONTRACT_KIND = "veto"


@dataclass(frozen=True)
class MeasurementContract:
    contract_id: str
    title: str
    statement: str
    veto_reason: str
    prompt_text: str
    source_constraint_ids: Tuple[str, ...]
    applies_to_token_prefixes: Tuple[str, ...] = ()
    kind: str = CONTRACT_KIND
    review_status: str = "pending_expert_review"

    def __post_init__(self) -> None:
        if self.kind != CONTRACT_KIND:
            raise ValueError(f"measurement contract kind must be {CONTRACT_KIND!r}: {self.kind!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementContractLibrary:
    version: str
    contracts: Tuple[MeasurementContract, ...]

    def get(self, contract_id: str) -> MeasurementContract:
        for item in self.contracts:
            if item.contract_id == contract_id:
                return item
        raise KeyError(f"unknown measurement contract: {contract_id}")

    def ids(self) -> Tuple[str, ...]:
        return tuple(item.contract_id for item in self.contracts)

    def by_source(self, old_constraint_id: str) -> Tuple[MeasurementContract, ...]:
        return tuple(item for item in self.contracts if old_constraint_id in item.source_constraint_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "contracts": [item.to_dict() for item in self.contracts],
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


MEASUREMENT_CONTRACTS: Tuple[MeasurementContract, ...] = (
    MeasurementContract(
        contract_id="M1_no_absolute_link_loss",
        title="本数据不能用两端功率相减计算链路损耗",
        statement="两端 lane 编号或标定口径不可靠，功率相减会产生物理上不可能的负损耗。",
        veto_reason="出现绝对链路损耗数值或两端功率相减时，该推理步骤作废。",
        prompt_text="不要根据两端功率相减给出链路损耗或衰减数值；只使用同侧相对量和有光/无光判断。",
        source_constraint_ids=("C12_no_absolute_link_loss", "C21_healthy_band_tx_level_is_not_attribution_evidence"),
        applies_to_token_prefixes=("lane:", "level:L1:txpower_mean:", "level:L2:txpower_mean:"),
    ),
    MeasurementContract(
        contract_id="M2_serdes_snr_unit_unknown",
        title="serdes_snr 量纲未知",
        statement="serdes_snr 的健康取值量级不是 dB，当前只能按有效/失效二值信号使用。",
        veto_reason="把 serdes_snr 当 dB 信噪比解释时，该推理步骤作废。",
        prompt_text="serdes_snr 量纲未确认，不要按 dB 比较或解释，只能作有效/失效判断。",
        source_constraint_ids=("C13_serdes_snr_unit_unknown",),
        applies_to_token_prefixes=("serdes:", "expert:L1:serdes_snr:", "expert:L2:serdes_snr:"),
    ),
    MeasurementContract(
        contract_id="M3_missing_host_snr_is_unknown",
        title="host_snr 缺失不等于正常",
        statement="host_snr 在多数 case 没有采集，看不到该字段只能说明未知。",
        veto_reason="把缺失 host_snr 描述为正常或健康时，该推理步骤作废。",
        prompt_text="host_snr 未采集时必须写成未知，不要推断它正常。",
        source_constraint_ids=("C14_host_snr_mostly_missing",),
        applies_to_token_prefixes=("telemetry:partial_telemetry", "telemetry:no_telemetry"),
    ),
    MeasurementContract(
        contract_id="M4_blackout_sentinel_is_no_reading",
        title="全链路 blackout 时哨兵表示读不到数",
        statement="两端收发光功率同时触底且 TxLOS 仍 Normal 时，哨兵含义翻转为遥测失效。",
        veto_reason="命中全链路 blackout 后，任何基于断光哨兵推出未发光或排除 fiber 的步骤作废。",
        prompt_text="全链路 blackout 时不要断言某端未发光，也不要据此排除光纤，应请求现场确认。",
        source_constraint_ids=("C15_blackout_sentinel_is_not_laser_off",),
        applies_to_token_prefixes=("drop:L1:txpower:all_lanes", "drop:L2:txpower:all_lanes", "drop:L1:rxpower:all_lanes", "drop:L2:rxpower:all_lanes"),
    ),
    MeasurementContract(
        contract_id="M5_population_prior_is_not_case_evidence",
        title="群体先验不是当前 case 物理证据",
        statement="SOP 叶节点分布、历史标签投票和类别先验只描述训练群体，不描述当前链路发生了什么。",
        veto_reason="把先验、历史标签分布或 SOP 叶节点统计作为 support 步骤时，该推理步骤作废。",
        prompt_text="类别先验、历史标签投票、决策树叶节点统计只能作上下文，不能作为当前 case 的 cited_evidence。",
        source_constraint_ids=("C19_population_prior_is_not_case_evidence",),
    ),
    MeasurementContract(
        contract_id="M6_fiber_not_identifiable_without_field_evidence",
        title="当前遥测不能确认 fiber 根因",
        statement="介质根因需要 OTDR、端面镜检或双向功率标定；当前遥测无法单独确认。",
        veto_reason="输出 fiber 自动结论时应改为候选 fiber + 补采清单。",
        prompt_text="不要给出自动 fiber 结论；怀疑介质时输出候选 fiber 并请求 OTDR、端面镜检或双向功率标定。",
        source_constraint_ids=("C20_fiber_not_identifiable_from_current_telemetry",),
    ),
)


MEASUREMENT_CONTRACT_LIBRARY = MeasurementContractLibrary(
    version=MEASUREMENT_CONTRACT_LIBRARY_VERSION,
    contracts=MEASUREMENT_CONTRACTS,
)


def render_measurement_prompt_block(
    library: MeasurementContractLibrary = MEASUREMENT_CONTRACT_LIBRARY,
    *,
    contracts: Sequence[MeasurementContract] | None = None,
) -> str:
    selected = list(library.contracts if contracts is None else contracts)
    lines = [f"# 量测契约（{library.version}，hash {library.content_hash()}）", ""]
    for item in selected:
        token_scope = "、".join(item.applies_to_token_prefixes) if item.applies_to_token_prefixes else "无需绑定当前 token"
        lines.append(
            f"- [{item.contract_id}] {item.prompt_text}\n"
            f"  契约类型：veto；可用 token 前缀={token_scope}；违反后该推理步骤或结论作废。"
        )
    return "\n".join(lines).strip() + "\n"


def iter_measurement_contracts(
    library: MeasurementContractLibrary = MEASUREMENT_CONTRACT_LIBRARY,
) -> Iterable[MeasurementContract]:
    return iter(library.contracts)
