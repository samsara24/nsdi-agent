"""Train-only offline knowledge bundle for the non-evolution RCA experiment.

The bundle is the hard boundary between training and evaluation: thresholds,
feature-model parameters, historical vectors, the evidence graph, learned SOP,
and confidence calibration are fitted once on the manifest train split, then
loaded read-only while evaluating the test split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from .anomaly import ThresholdModel, fit_thresholds
from .branches import fit_calibration, handle_many
from .branches.base import BranchCalibration
from .constraints.library import CONSTRAINT_LIBRARY
from .constraints.measurement import MEASUREMENT_CONTRACT_LIBRARY
from .constraints.physics import PHYSICS_LIBRARY
from .decision_tree import (
    NumericDecisionTree,
    fit_numeric_decision_tree,
    numeric_features_from_pack,
    numeric_features_from_packs,
)
from .decision import (
    DEFAULT_DECISION_POLICY,
    DecisionPolicy,
    LLMCalibration,
    apply_llm_calibration,
    build_candidates,
    decide_many,
    fit_decision_policy,
)
from .evidence_graph import EvidenceGraph, RoutingPolicy, match_many
from .evidence_pack import EvidencePack, build_packs, labels_of
from .expert import ExpertCalibration, diagnose_many
from .features.dictionary import dictionary_for
from .features.extractor import CaseFeatures, FeatureModel, extract_features, fit_feature_model
from .feedback import build_case_diagnosis
from .sop import LearnedSOP, learn_sop
from .topology import SOURCE_TOPOLOGIES


KNOWLEDGE_BUNDLE_SCHEMA = "offline-knowledge-bundle-v1"


def _active_constraint_version(pack: EvidencePack) -> str:
    if pack.source_dataset in SOURCE_TOPOLOGIES:
        return f"{PHYSICS_LIBRARY.version}+{MEASUREMENT_CONTRACT_LIBRARY.version}"
    return CONSTRAINT_LIBRARY.version


def _model_from_dict(value: Mapping[str, Any]) -> Any:
    version = str(value.get("version", ""))
    if version.startswith("numeric-decision-tree"):
        return NumericDecisionTree.from_dict(value)
    return LearnedSOP.from_dict(value)


@dataclass(frozen=True)
class TrainingKnowledgeArtifacts:
    """LLM-assisted train-only build logs; raw traces live outside the bundle."""

    summary: Mapping[str, Any] = field(default_factory=dict)
    traces: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class OfflineKnowledgeBundle:
    schema_version: str
    source_dataset: str
    split_manifest_hash: str
    feature_profile: str
    thresholds: ThresholdModel
    feature_model: FeatureModel
    graph: EvidenceGraph
    sop: Any
    training_features: Tuple[CaseFeatures, ...]
    branch_calibrations: Mapping[str, BranchCalibration]
    llm_calibrations: Mapping[str, LLMCalibration] = field(default_factory=dict)
    #: 专家规则各分组在训练集上的实测可靠性。规则本身来自现网经验、不含拟合参数，
    #: 但「这条规则有多可信」必须留在训练边界内，因此它和分支标定一样进知识包。
    expert_calibration: Optional[ExpertCalibration] = None
    #: 每个路由策略在训练留一法上反解出的 M9 工作点。测试阶段只读加载，不重新拟合。
    decision_policies: Mapping[str, DecisionPolicy] = field(default_factory=dict)
    build_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def train_case_ids(self) -> Tuple[str, ...]:
        return tuple(item.case_id for item in self.training_features)

    def content_hash(self) -> str:
        payload = self._payload()
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_dataset": self.source_dataset,
            "split_manifest_hash": self.split_manifest_hash,
            "feature_profile": self.feature_profile,
            "thresholds": self.thresholds.to_dict(),
            "feature_model": self.feature_model.to_dict(),
            "graph": self.graph.to_dict(),
            "sop": self.sop.to_dict(),
            "training_features": [item.to_dict() for item in self.training_features],
            "branch_calibrations": {
                name: calibration.to_dict()
                for name, calibration in sorted(self.branch_calibrations.items())
            },
            "llm_calibrations": {
                name: calibration.to_dict()
                for name, calibration in sorted(self.llm_calibrations.items())
            },
            "expert_calibration": (
                self.expert_calibration.to_dict() if self.expert_calibration is not None else None
            ),
            "decision_policies": {
                name: policy.to_dict()
                for name, policy in sorted(self.decision_policies.items())
            },
            "build_metadata": dict(self.build_metadata),
        }

    def to_dict(self) -> Dict[str, Any]:
        value = self._payload()
        value["content_hash"] = self.content_hash()
        value["train_case_ids"] = list(self.train_case_ids)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OfflineKnowledgeBundle":
        schema = str(value.get("schema_version", ""))
        if schema != KNOWLEDGE_BUNDLE_SCHEMA:
            raise ValueError(
                f"unsupported knowledge bundle schema: {schema!r}; "
                f"expected {KNOWLEDGE_BUNDLE_SCHEMA!r}"
            )
        bundle = cls(
            schema_version=schema,
            source_dataset=str(value.get("source_dataset", "")),
            split_manifest_hash=str(value.get("split_manifest_hash", "")),
            feature_profile=str(value["feature_profile"]),
            thresholds=ThresholdModel.from_dict(dict(value["thresholds"])),
            feature_model=FeatureModel.from_dict(dict(value["feature_model"])),
            graph=EvidenceGraph.from_dict(value["graph"]),
            sop=_model_from_dict(value["sop"]),
            training_features=tuple(
                CaseFeatures.from_dict(dict(item))
                for item in value.get("training_features", ())
            ),
            branch_calibrations={
                str(name): BranchCalibration.from_dict(item)
                for name, item in value.get("branch_calibrations", {}).items()
            },
            llm_calibrations={
                str(name): LLMCalibration.from_dict(item)
                for name, item in value.get("llm_calibrations", {}).items()
            },
            expert_calibration=(
                ExpertCalibration.from_dict(value["expert_calibration"])
                if value.get("expert_calibration")
                else None
            ),
            decision_policies={
                str(name): DecisionPolicy(
                    version=str(item.get("version", DEFAULT_DECISION_POLICY.version)),
                    final_lower_bound=float(item.get("final_lower_bound", 0.5)),
                    minimum_support=int(item.get("minimum_support", 10)),
                    candidate_order=tuple(item.get("candidate_order", ("branch",))),
                    target_selective_risk=item.get("target_selective_risk"),
                    fitted_on=str(item.get("fitted_on", "")),
                    non_identifiable_labels=tuple(item.get("non_identifiable_labels", ())),
                    non_identifiable_evidence={
                        str(key): tuple(entry)
                        for key, entry in item.get("non_identifiable_evidence", {}).items()
                    },
                    per_label_lower_bound={
                        str(key): float(entry)
                        for key, entry in item.get("per_label_lower_bound", {}).items()
                    },
                )
                for name, item in value.get("decision_policies", {}).items()
            },
            build_metadata=dict(value.get("build_metadata", {})),
        )
        expected_hash = str(value.get("content_hash", ""))
        if expected_hash and bundle.content_hash() != expected_hash:
            raise ValueError(
                "knowledge bundle content hash mismatch: "
                f"stored={expected_hash}, computed={bundle.content_hash()}"
            )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        dictionary = dictionary_for(self.feature_profile)
        if self.graph.dictionary_hash != dictionary.content_hash():
            raise ValueError("knowledge bundle feature dictionary hash does not match current code")
        if self.feature_model.dictionary_hash != self.graph.dictionary_hash:
            raise ValueError("feature model and evidence graph dictionary hashes differ")
        if (
            not str(getattr(self.sop, "version", "")).startswith("numeric-decision-tree")
            and self.sop.dictionary_hash != self.graph.dictionary_hash
        ):
            raise ValueError("learned SOP and evidence graph dictionary hashes differ")
        feature_ids = self.train_case_ids
        graph_ids = tuple(item.case_id for item in self.graph.cases)
        if feature_ids != graph_ids:
            raise ValueError("training vector order does not match evidence graph case order")
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("duplicate case ids in knowledge bundle")
        if self.sop.training_case_count != len(feature_ids):
            raise ValueError("learned SOP training count does not match historical vectors")
        if not self.branch_calibrations:
            raise ValueError("knowledge bundle has no train-only branch calibration")

    def save(self, path: Path) -> Path:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "OfflineKnowledgeBundle":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def extract_test_features(
        self,
        cases: Sequence[Dict[str, Any]],
        *,
        source_dataset: Optional[str] = None,
    ) -> Tuple[Tuple[EvidencePack, ...], Tuple[CaseFeatures, ...]]:
        """Transform test inputs using frozen train-only parameters."""
        dictionary = dictionary_for(self.feature_profile)
        packs = tuple(
            build_packs(cases, source_dataset=source_dataset or self.source_dataset)
        )
        features = tuple(
            extract_features(
                pack,
                self.thresholds,
                self.feature_model,
                dictionary=dictionary,
            )
            for pack in packs
        )
        return packs, features

    def expert_predictions(
        self,
        packs: Sequence[EvidencePack],
    ) -> Tuple[Optional[Dict[str, Any]], ...]:
        """用训练集标定给测试 case 的专家裁决打分。

        规则在测试期照常运行——它没有参数需要冻结；被冻结的是可靠性统计。
        知识包里没有专家标定时全部返回 `None`，M9 于是看不到专家候选，
        而不是拿一个没有标定的下界去闯门禁。
        """
        if self.expert_calibration is None:
            return tuple(None for _ in packs)
        return tuple(
            self.expert_calibration.prediction(diagnosis)
            for diagnosis in diagnose_many(packs)
        )


def out_of_fold_expert_predictions(
    packs: Sequence[EvidencePack],
    labels: Sequence[str],
    *,
    folds: int = 5,
) -> Tuple[Optional[Dict[str, Any]], ...]:
    """折外的专家规则打分，供门限反解使用。

    这里折外的只有**可靠性统计**，规则输出本身与折划分无关——这正是专家规则
    与 SOP 的结构性差异：SOP 折外要重新学一棵树，专家规则只要重新数一遍
    「同组里有多少条判对了」。因此这里不存在留一法那种叶内反序问题
    （见 `_out_of_fold_sop_predictions`），折外与全量的差值就是纯粹的乐观量。
    """
    diagnoses = diagnose_many(packs)
    assignments = stratified_folds(labels, folds)
    out: list[Optional[Dict[str, Any]]] = [None] * len(packs)
    for held_out in assignments:
        if not held_out:
            continue
        held = set(held_out)
        keep = [index for index in range(len(packs)) if index not in held]
        if not keep:
            continue
        calibration = ExpertCalibration.fit(
            [diagnoses[index] for index in keep],
            [labels[index] for index in keep],
            source=f"train-oof{folds}",
        )
        for index in held_out:
            out[index] = calibration.prediction(diagnoses[index])
    return tuple(out)


def _loo_sop_predictions(
    features: Sequence[Any],
    labels: Sequence[str],
    *,
    sop: Any,
) -> Tuple[Optional[Dict[str, Any]], ...]:
    """对每条训练 case 用「去掉它自己」重拟合的 SOP 给出预测。

    树本身很浅，161 次重拟合的代价可以忽略。这一步不是形式主义：
    叶节点的支持数与 Wilson 下界正是 M9 门禁要用的量，如果它包含被评估的
    那条 case，反解出来的阈值在测试集上就会失效。

    **不要用它的置信度去反解门限**，改用 `_out_of_fold_sop_predictions`。
    留一法确实排除了自身标签的正向泄漏，但引入了一个反向的构造性相关：
    叶节点纯度 = (符合该叶结论的样本数 - [自己符合]) / (叶大小 - 1)，
    于是同一个叶子里，**符合结论的 case 拿到的置信度必然低于不符合的 case**。
    在 rca_v2_l2fixed 上三个叶子无一例外（详见 Progress.md 迭代 2）：
    叶 `root.absent.present.absent` 判 L2，真值为 L2 的 28 条下界 0.6320，
    真值为 L1 的下界 0.6666~0.7149，真值为 fiber 的高达 0.7510。
    置信度与正确性在叶内是完全反序的，按它反解出的门限会系统性地
    「放行反例、拦下正例」——这正是迭代 1 平衡召回只有 0.2596 的成因。
    """
    out: list[Optional[Dict[str, Any]]] = []
    kept_features = list(features)
    kept_labels = list(labels)
    for index in range(len(kept_features)):
        subset_features = kept_features[:index] + kept_features[index + 1 :]
        subset_labels = kept_labels[:index] + kept_labels[index + 1 :]
        if not subset_features:
            out.append(None)
            continue
        if str(getattr(sop, "version", "")).startswith("numeric-decision-tree"):
            model = fit_numeric_decision_tree(
                subset_features,
                subset_labels,
                max_depth=sop.max_depth,
                min_leaf_size=sop.min_leaf_size,
                source=f"{sop.source}:loo",
            )
        else:
            model = learn_sop(
                subset_features,
                subset_labels,
                max_depth=sop.max_depth,
                min_leaf_size=sop.min_leaf_size,
                source=f"{sop.source}:loo",
            )
        out.append(model.predict(kept_features[index]).to_dict())
    return tuple(out)


def stratified_folds(labels: Sequence[str], folds: int) -> Tuple[Tuple[int, ...], ...]:
    """按标签分层地把下标轮转分配到 `folds` 折。

    不使用随机数：给定标签序列结果唯一，实验可以逐字节复现。
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    buckets: list[list[int]] = [[] for _ in range(folds)]
    by_label: Dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    cursor = 0
    for label in sorted(by_label):
        for index in by_label[label]:
            buckets[cursor % folds].append(index)
            cursor += 1
    return tuple(tuple(sorted(bucket)) for bucket in buckets)


def _out_of_fold_sop_predictions(
    features: Sequence[Any],
    labels: Sequence[str],
    *,
    sop: Any,
    folds: int = 5,
) -> Tuple[Optional[Dict[str, Any]], ...]:
    """用分层 K 折的折外模型给每条训练 case 出预测，供门限反解使用。

    与留一法的区别只有一处，但决定了门限是否可用：模型在「去掉整整一折」
    的数据上拟合，被评估 case 的标签只通过它所在的那一折（约 20% 的数据）
    影响叶节点纯度，而不再是唯一的扰动源。留一法下「叶内符合者置信度更低」
    是恒等式，K 折下它被同折其余样本冲淡，置信度重新变成模型属性
    而不是被留出标签的函数。

    折外预测仍然严格无自身泄漏：预测某条 case 的模型从未见过它。
    """
    assignments = stratified_folds(labels, folds)
    out: list[Optional[Dict[str, Any]]] = [None] * len(features)
    for held_out in assignments:
        if not held_out:
            continue
        keep = [index for index in range(len(features)) if index not in set(held_out)]
        if not keep:
            continue
        if str(getattr(sop, "version", "")).startswith("numeric-decision-tree"):
            model = fit_numeric_decision_tree(
                [features[index] for index in keep],
                [labels[index] for index in keep],
                max_depth=sop.max_depth,
                min_leaf_size=sop.min_leaf_size,
                source=f"{sop.source}:oof{folds}",
            )
        else:
            model = learn_sop(
                [features[index] for index in keep],
                [labels[index] for index in keep],
                max_depth=sop.max_depth,
                min_leaf_size=sop.min_leaf_size,
                source=f"{sop.source}:oof{folds}",
            )
        for index in held_out:
            out[index] = model.predict(features[index]).to_dict()
    return tuple(out)


def fit_offline_knowledge(
    train_cases: Sequence[Dict[str, Any]],
    *,
    source_dataset: str,
    split_manifest_hash: str,
    feature_profile: str,
    policies: Sequence[RoutingPolicy],
    reasoner: Optional[Any] = None,
    top_k: int = 0,
    build_metadata: Optional[Mapping[str, Any]] = None,
    target_selective_risk: Optional[float] = None,
    decision_minimum_support: int = 10,
    decision_candidate_order: Tuple[str, ...] = ("branch",),
    decision_non_identifiable_labels: Tuple[str, ...] = (),
    decision_non_identifiable_evidence: Optional[Mapping[str, Tuple[str, ...]]] = None,
    decision_class_conditional: bool = False,
) -> Tuple[OfflineKnowledgeBundle, TrainingKnowledgeArtifacts]:
    """Fit and optionally LLM-enrich all train-only artifacts.

    When a reasoner is supplied, every train case that enters an LLM-capable
    branch is processed in leave-one-out mode. The validated SOP+LLM chain is
    attached to the historical case as EvidenceGraph v2 diagnosis knowledge.
    """
    labels = labels_of(train_cases)
    dictionary = dictionary_for(feature_profile)
    thresholds = fit_thresholds(train_cases)
    packs = tuple(build_packs(train_cases, source_dataset=source_dataset))
    numeric_rows = numeric_features_from_packs(packs)
    model = fit_feature_model(packs, dictionary=dictionary)
    features = tuple(
        extract_features(pack, thresholds, model, dictionary=dictionary)
        for pack in packs
    )
    sop = fit_numeric_decision_tree(
        numeric_rows,
        labels,
        source=f"{Path(source_dataset).name}:manifest-train",
    )
    graph = EvidenceGraph.build(
        features,
        labels,
        feature_model=model,
        dictionary=dictionary,
        source_dataset=source_dataset,
        confirmed_by="dataset:manifest-train",
    )
    train_results = match_many(graph, features, top_k=top_k, leave_one_out=True)
    # 专家规则不学参数，因此这里只统计各规则组在训练集上的实测可靠性。
    # 测试期用的就是这张表；门限反解另用折外版本，两者之差即为乐观量。
    expert_calibration = ExpertCalibration.fit(
        diagnose_many(packs), labels, source="manifest-train"
    )

    branch_calibrations: Dict[str, BranchCalibration] = {}
    llm_calibrations: Dict[str, LLMCalibration] = {}
    decision_policies: Dict[str, DecisionPolicy] = {}
    decision_fits: Dict[str, Any] = {}
    trace_artifacts: Dict[str, Dict[str, Any]] = {}
    policy_summaries: Dict[str, Any] = {}
    diagnosis_by_case: MutableMapping[str, Any] = {}

    for policy in policies:
        calibration = fit_calibration(
            train_results,
            packs,
            labels,
            policy=policy,
            source="manifest-train-loo",
        )
        branch_calibrations[policy.name] = calibration
        traces: Dict[str, Any] = {}
        paired = handle_many(
            train_results,
            packs,
            calibration,
            policy=policy,
            reasoner=reasoner,
            trace_collector=traces,
            features=features,
            sop_model=sop,
        )
        outcomes = [item[1] for item in paired]
        if reasoner is not None:
            llm_calibration = LLMCalibration.fit(
                outcomes,
                [traces.get(outcome.case_id) for outcome in outcomes],
                labels,
                source=f"manifest-train-loo-llm:{policy.name}",
            )
            llm_calibrations[policy.name] = llm_calibration
            outcomes = [
                apply_llm_calibration(outcome, traces.get(outcome.case_id), llm_calibration)
                for outcome in outcomes
            ]

        # SOP / expert 预测仍然计算并写入报告字段，但正式默认 candidate_order 只有 branch。
        # 只有显式消融把它们加入 candidate_order 时，下面的折外预测才会进入 M9 级联。
        # 这样保留历史 i3-i5 对照价值，同时不让训练集统计先验或专家规则顶替证据图主干。
        oof_sop = _out_of_fold_sop_predictions(
            numeric_rows if str(getattr(sop, "version", "")).startswith("numeric-decision-tree") else features,
            labels,
            sop=sop,
        )
        oof_expert = out_of_fold_expert_predictions(packs, labels)
        decision_policy = DEFAULT_DECISION_POLICY
        decision_fit: Optional[Dict[str, Any]] = None
        if target_selective_risk is not None:
            probe_policy = DecisionPolicy(
                final_lower_bound=0.0,
                minimum_support=decision_minimum_support,
                candidate_order=decision_candidate_order,
                non_identifiable_labels=decision_non_identifiable_labels,
                non_identifiable_evidence=dict(decision_non_identifiable_evidence or {}),
            )
            rows = [
                (
                    build_candidates(
                        outcome,
                        sop_prediction=sop_pred,
                        expert_prediction=expert_pred,
                        policy=probe_policy,
                    ),
                    label,
                )
                for outcome, sop_pred, expert_pred, label in zip(
                    outcomes, oof_sop, oof_expert, labels
                )
            ]
            decision_policy, decision_fit = fit_decision_policy(
                rows,
                target_selective_risk=target_selective_risk,
                minimum_support=decision_minimum_support,
                candidate_order=decision_candidate_order,
                non_identifiable_labels=decision_non_identifiable_labels,
                non_identifiable_evidence=decision_non_identifiable_evidence,
                source=f"manifest-train-loo:{policy.name}",
                class_conditional=decision_class_conditional,
            )
            decision_policies[policy.name] = decision_policy

        final_decisions = decide_many(
            outcomes,
            decision_policy,
            sop_predictions=oof_sop,
            expert_predictions=oof_expert,
        )
        for pack, feature, outcome, final_decision, confirmed_label in zip(
            packs, features, outcomes, final_decisions, labels
        ):
            diagnosis_by_case[pack.case_id] = build_case_diagnosis(
                pack,
                feature,
                outcome,
                final_decision,
                sop_version=sop.version,
                constraint_library_version=_active_constraint_version(pack),
                confirmed_by="dataset:manifest-train",
                confirmed_label=confirmed_label,
            )
        trace_artifacts[policy.name] = {
            case_id: trace.to_dict() for case_id, trace in sorted(traces.items())
        }
        policy_summaries[policy.name] = {
            "case_count": len(features),
            "llm_trace_count": len(traces),
            "llm_accepted_count": sum(
                getattr(trace, "accepted", None) is not None for trace in traces.values()
            ),
            "prediction_matches_training_label": sum(
                outcome.verdict == label for outcome, label in zip(outcomes, labels)
            ),
            "answered_count": sum(outcome.verdict is not None for outcome in outcomes),
            "branch_calibration": calibration.to_dict(),
            "llm_calibration": (
                llm_calibrations[policy.name].to_dict()
                if policy.name in llm_calibrations
                else None
            ),
            "decision_policy": decision_policy.to_dict(),
            "decision_policy_fit": decision_fit,
            "oof_sop_answered": sum(
                1 for item in oof_sop if item is not None and item.get("verdict") is not None
            ),
            "oof_sop_correct": sum(
                (item or {}).get("verdict") == label for item, label in zip(oof_sop, labels)
            ),
            "oof_expert_answered": sum(
                1 for item in oof_expert if item is not None and item.get("verdict") is not None
            ),
            "oof_expert_correct": sum(
                (item or {}).get("verdict") == label for item, label in zip(oof_expert, labels)
            ),
        }
        if decision_fit is not None:
            decision_fits[policy.name] = decision_fit

    if diagnosis_by_case:
        graph = graph.with_case_diagnoses(
            [diagnosis_by_case[case_id] for case_id in sorted(diagnosis_by_case)]
        )

    bundle = OfflineKnowledgeBundle(
        schema_version=KNOWLEDGE_BUNDLE_SCHEMA,
        source_dataset=source_dataset,
        split_manifest_hash=split_manifest_hash,
        feature_profile=feature_profile,
        thresholds=thresholds,
        feature_model=model,
        graph=graph,
        sop=sop,
        training_features=features,
        branch_calibrations=branch_calibrations,
        llm_calibrations=llm_calibrations,
        expert_calibration=expert_calibration,
        decision_policies=decision_policies,
        build_metadata=dict(build_metadata or {}),
    )
    bundle.validate()
    return bundle, TrainingKnowledgeArtifacts(
        summary={
            "schema_version": "training-knowledge-summary-v1",
            "train_case_count": len(train_cases),
            "historical_vector_count": len(features),
            "evidence_graph_case_count": len(graph.cases),
            "evidence_graph_diagnosis_count": len(graph.case_diagnoses),
            "graph_version": graph.version,
            "graph_purity": graph.purity_report(),
            "sop": sop.to_dict(),
            "expert_calibration": expert_calibration.to_dict(),
            "policies": policy_summaries,
        },
        traces=trace_artifacts,
    )
