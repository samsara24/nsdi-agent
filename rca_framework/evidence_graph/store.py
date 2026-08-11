"""M2 证据图存储与索引。

图的节点只有两类：case 与 feature token。边是「某 case 具备某 token」。
标签挂在 case 节点上，但检索路径默认不读它——`EvidenceGraph.signature_of` 与
`token_index` 都不接触标签，只有在调用方显式要求时才通过 `label_of` 取。
这样「检索阶段泄漏训练标签」就需要一次显式调用，而不会顺手发生。

为什么不复用 legacy `AnomalyKnowledgeGraph`：那个类把索引、路径打分、feature rule
学习和 RAG 检索揉在一起，且它的 `scores` 是 58/85 回归锚点的一部分，动不得。
这里只做索引与检索，打分交给 N5 分支。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..features.dictionary import FEATURE_DICTIONARY, FeatureDictionary
from ..features.extractor import CaseFeatures, FeatureModel
from ..types import ROOT_CAUSES


EVIDENCE_GRAPH_SCHEMA = "evidence-graph-v1"
EVIDENCE_GRAPH_V2_SCHEMA = "evidence-graph-v2"


@dataclass(frozen=True)
class GraphCase:
    """图里的一个历史 case 节点。"""

    case_id: str
    label: str
    tokens: Tuple[str, ...]
    telemetry_status: str = "full_telemetry"
    confirmed_by: str = "dataset"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "tokens": list(self.tokens),
            "telemetry_status": self.telemetry_status,
            "confirmed_by": self.confirmed_by,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphCase":
        return cls(
            case_id=value["case_id"],
            label=value.get("label", ""),
            tokens=tuple(value.get("tokens", [])),
            telemetry_status=value.get("telemetry_status", "full_telemetry"),
            confirmed_by=value.get("confirmed_by", "dataset"),
        )


@dataclass(frozen=True)
class DiagnosisNode:
    node_id: str
    node_type: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.node_id, "type": self.node_type, "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DiagnosisNode":
        return cls(node_id=str(value["id"]), node_type=str(value["type"]), attrs=dict(value.get("attrs", {})))


@dataclass(frozen=True)
class DiagnosisEdge:
    src: str
    dst: str
    edge_type: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "type": self.edge_type, "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DiagnosisEdge":
        return cls(
            src=str(value["src"]),
            dst=str(value["dst"]),
            edge_type=str(value["type"]),
            attrs=dict(value.get("attrs", {})),
        )


@dataclass(frozen=True)
class CaseDiagnosis:
    case_id: str
    sop_version: str
    constraint_library_version: str
    nodes: Tuple[DiagnosisNode, ...] = ()
    edges: Tuple[DiagnosisEdge, ...] = ()
    confirmed_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "sop_version": self.sop_version,
            "constraint_library_version": self.constraint_library_version,
            "confirmed_by": self.confirmed_by,
            "content_hash": self.content_hash(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CaseDiagnosis":
        return cls(
            case_id=str(value["case_id"]),
            sop_version=str(value.get("sop_version", "")),
            constraint_library_version=str(value.get("constraint_library_version", "")),
            confirmed_by=str(value.get("confirmed_by", "")),
            nodes=tuple(DiagnosisNode.from_dict(item) for item in value.get("nodes", [])),
            edges=tuple(DiagnosisEdge.from_dict(item) for item in value.get("edges", [])),
        )

    def content_hash(self) -> str:
        payload = {
            "case_id": self.case_id,
            "sop_version": self.sop_version,
            "constraint_library_version": self.constraint_library_version,
            "confirmed_by": self.confirmed_by,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@dataclass
class EvidenceGraph:
    """按特征字典 v1 建立的历史 case 索引。

    `version` 是证据图版本号，必须写进 `run_manifest.json`。它由图内容、特征字典
    指纹和 `FeatureModel` 指纹三者共同决定：任何一项变了，历史匹配结果都不可比。
    """

    cases: Tuple[GraphCase, ...] = ()
    idf: Dict[str, float] = field(default_factory=dict)
    dictionary_version: str = FEATURE_DICTIONARY.version
    dictionary_hash: str = ""
    feature_model: Optional[Dict[str, Any]] = None
    source_dataset: str = ""
    schema_version: str = EVIDENCE_GRAPH_SCHEMA
    case_diagnoses: Tuple[CaseDiagnosis, ...] = ()

    # --- 构建 ---------------------------------------------------------------

    @classmethod
    def build(
        cls,
        features: Sequence[CaseFeatures],
        labels: Sequence[str],
        *,
        feature_model: Optional[FeatureModel] = None,
        dictionary: FeatureDictionary = FEATURE_DICTIONARY,
        source_dataset: str = "",
        confirmed_by: str = "dataset",
    ) -> "EvidenceGraph":
        if len(features) != len(labels):
            raise ValueError("features and labels must be the same length")
        unknown = sorted({label for label in labels if label not in ROOT_CAUSES})
        if unknown:
            raise ValueError(f"unsupported labels in evidence graph: {unknown}")

        cases = tuple(
            GraphCase(
                case_id=item.case_id,
                label=label,
                tokens=item.tokens,
                telemetry_status=item.telemetry_status,
                confirmed_by=confirmed_by,
            )
            for item, label in zip(features, labels)
        )
        return cls(
            cases=cases,
            idf=_idf(tuple(case.tokens for case in cases)),
            dictionary_version=dictionary.version,
            dictionary_hash=dictionary.content_hash(),
            feature_model=feature_model.to_dict() if feature_model is not None else None,
            source_dataset=source_dataset,
        )

    def extend(self, cases: Sequence[GraphCase]) -> "EvidenceGraph":
        """N8 回灌入口：追加人工确认过的 case 并重算 IDF。

        返回新对象而不是就地修改，这样每个证据图版本都是一个不可变快照，
        实验产物里记录的 `version` 才真的能定位到当时用的那份图。
        """
        known = {case.case_id for case in self.cases}
        duplicates = sorted(case.case_id for case in cases if case.case_id in known)
        if duplicates:
            raise ValueError(f"case already in graph: {duplicates}")
        merged = self.cases + tuple(cases)
        return EvidenceGraph(
            cases=merged,
            idf=_idf(tuple(case.tokens for case in merged)),
            dictionary_version=self.dictionary_version,
            dictionary_hash=self.dictionary_hash,
            feature_model=self.feature_model,
            source_dataset=self.source_dataset,
            schema_version=self.schema_version,
            case_diagnoses=self.case_diagnoses,
        )

    def with_case_diagnoses(self, diagnoses: Sequence[CaseDiagnosis]) -> "EvidenceGraph":
        by_case = {item.case_id: item for item in self.case_diagnoses}
        for diagnosis in diagnoses:
            by_case[diagnosis.case_id] = diagnosis
        return EvidenceGraph(
            cases=self.cases,
            idf=self.idf,
            dictionary_version=self.dictionary_version,
            dictionary_hash=self.dictionary_hash,
            feature_model=self.feature_model,
            source_dataset=self.source_dataset,
            schema_version=EVIDENCE_GRAPH_V2_SCHEMA,
            case_diagnoses=tuple(by_case[key] for key in sorted(by_case)),
        )

    # --- 查询 ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cases)

    def signature_of(self, case_id: str) -> Tuple[str, ...]:
        for case in self.cases:
            if case.case_id == case_id:
                return case.tokens
        raise KeyError(case_id)

    def label_of(self, case_id: str) -> str:
        """显式取标签。检索路径不调用它；只有 N5a 复用历史结论时才调用。"""
        for case in self.cases:
            if case.case_id == case_id:
                return case.label
        raise KeyError(case_id)

    def token_index(self) -> Dict[str, Tuple[str, ...]]:
        """token -> 拥有它的 case id。倒排索引，不含标签。"""
        index: Dict[str, List[str]] = defaultdict(list)
        for case in self.cases:
            for token in case.tokens:
                index[token].append(case.case_id)
        return {token: tuple(ids) for token, ids in sorted(index.items())}

    def label_distribution(self) -> Dict[str, int]:
        counts = Counter(case.label for case in self.cases)
        return {label: counts[label] for label in ROOT_CAUSES}

    def signature_groups(self) -> Dict[str, Tuple[str, ...]]:
        """signature 字面量 -> case id。用于 N5a 的桶纯净度统计。"""
        groups: Dict[str, List[str]] = defaultdict(list)
        for case in self.cases:
            groups["|".join(case.tokens)].append(case.case_id)
        return {key: tuple(ids) for key, ids in sorted(groups.items())}

    def purity_report(self) -> Dict[str, Any]:
        """N5a 必报的 signature 纯净度指标。"""
        groups = self.signature_groups()
        labels = {case.case_id: case.label for case in self.cases}
        mixed = {
            key: ids for key, ids in groups.items()
            if len({labels[case_id] for case_id in ids}) > 1
        }
        mixed_cases = sum(len(ids) for ids in mixed.values())
        total = len(self.cases)
        return {
            "case_count": total,
            "signature_group_count": len(groups),
            "mixed_label_group_count": len(mixed),
            "mixed_label_case_count": mixed_cases,
            "mixed_label_case_ratio": round(mixed_cases / total, 6) if total else None,
            "singleton_group_count": sum(1 for ids in groups.values() if len(ids) == 1),
            "empty_signature_case_count": len(groups.get("", ())),
        }

    # --- 版本与持久化 -------------------------------------------------------

    def content_hash(self) -> str:
        value = {
            "cases": [case.to_dict() for case in sorted(self.cases, key=lambda item: item.case_id)],
            "dictionary_hash": self.dictionary_hash,
            "feature_model": self.feature_model,
        }
        if self.case_diagnoses:
            value["case_diagnoses"] = [item.to_dict() for item in self.case_diagnoses]
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def version(self) -> str:
        return f"{EVIDENCE_GRAPH_SCHEMA}:{len(self.cases)}:{self.content_hash()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "content_hash": self.content_hash(),
            "source_dataset": self.source_dataset,
            "dictionary_version": self.dictionary_version,
            "dictionary_hash": self.dictionary_hash,
            "feature_model": self.feature_model,
            "case_count": len(self.cases),
            "label_distribution": self.label_distribution(),
            "idf": dict(sorted(self.idf.items())),
            "cases": [case.to_dict() for case in self.cases],
            "case_diagnoses": [item.to_dict() for item in self.case_diagnoses],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceGraph":
        return cls(
            cases=tuple(GraphCase.from_dict(item) for item in value.get("cases", [])),
            idf=dict(value.get("idf", {})),
            dictionary_version=value.get("dictionary_version", FEATURE_DICTIONARY.version),
            dictionary_hash=value.get("dictionary_hash", ""),
            feature_model=value.get("feature_model"),
            source_dataset=value.get("source_dataset", ""),
            schema_version=value.get("schema_version", EVIDENCE_GRAPH_SCHEMA),
            case_diagnoses=tuple(CaseDiagnosis.from_dict(item) for item in value.get("case_diagnoses", [])),
        )


def _idf(signatures: Sequence[Iterable[str]]) -> Dict[str, float]:
    """与 legacy `AnomalyKnowledgeGraph.fit` 相同的 IDF 定义。

    键序固定为字典序、数值固定舍入到 8 位，否则集合迭代顺序会让相似度出现
    浮点级抖动，产物不可复现——阶段 0 已经踩过这个坑。
    """
    document_count = len(signatures)
    counts: Counter[str] = Counter()
    for signature in signatures:
        counts.update(set(signature))
    return {
        token: round(math.log((document_count + 1) / (count + 1)) + 1.0, 8)
        for token, count in sorted(counts.items())
    }
