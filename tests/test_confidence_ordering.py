"""迭代 2 守护：门限赖以工作的「置信度越高越可能对」必须成立。

迭代 1 的 66% 覆盖率是假的，成因不在门限而在置信度：SOP 候选的置信度取自
「去掉自己重拟合」的叶节点纯度，而

    纯度(去掉 case i) = (符合结论的样本数 - [i 符合]) / (叶大小 - 1)

于是同一叶子内，**符合该叶结论的 case 拿到的置信度必然比不符合的低**。
置信度与正确性在叶内完全反序，按它反解出的门限专挑反例放行、把正例挡在外面。

这类错误不会让任何断言失败、也不会让指标变难看——恰恰相反，它让指标变好看。
所以必须有一个直接量测「排序方向」的测试盯着它。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.features.dictionary import dictionary_for
from rca_framework.features.extractor import extract_features, fit_feature_model
from rca_framework.knowledge import (
    _loo_sop_predictions,
    _out_of_fold_sop_predictions,
    stratified_folds,
)
from rca_framework.sop import learn_sop
from scripts.probe_per_label_operating_points import confidence_auc

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "rca_v2_l2fixed"


def test_confidence_auc_reports_direction_not_just_magnitude():
    assert confidence_auc([(0.9, 1), (0.1, 0)]) == 1.0
    assert confidence_auc([(0.1, 1), (0.9, 0)]) == 0.0
    assert confidence_auc([(0.5, 1), (0.5, 0)]) == 0.5
    # 只有一类时无法定义方向，必须返回 None 而不是悄悄给 0.5。
    assert confidence_auc([(0.9, 1), (0.8, 1)]) is None


def test_stratified_folds_are_balanced_deterministic_and_exhaustive():
    labels = ["L2"] * 10 + ["L1"] * 5 + ["fiber"] * 2
    folds = stratified_folds(labels, 5)
    assert stratified_folds(labels, 5) == folds
    flat = sorted(index for fold in folds for index in fold)
    assert flat == list(range(len(labels)))
    assert len({index for fold in folds for index in fold}) == len(labels)
    # 每折的多数类样本数不应相差超过 1。
    counts = [sum(1 for index in fold if labels[index] == "L2") for fold in folds]
    assert max(counts) - min(counts) <= 1
    with pytest.raises(ValueError):
        stratified_folds(labels, 1)


@pytest.fixture(scope="module")
def train_split():
    if not DATA_DIR.exists():
        pytest.skip(f"dataset not available: {DATA_DIR}")
    cases = cases_by_manifest_split(DATA_DIR, "train")
    labels = labels_of(cases)
    dictionary = dictionary_for("v2")
    thresholds = fit_thresholds(cases)
    packs = build_packs(cases, source_dataset=str(DATA_DIR))
    model = fit_feature_model(packs, dictionary=dictionary)
    features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in packs]
    sop = learn_sop(features, labels, source=f"{DATA_DIR.name}:manifest-train")
    return features, labels, sop


def within_leaf_aucs(predictions, labels):
    """按叶子分组算「置信度 vs 判对」的 AUC。

    必须分组内看：不同叶子的真实纯度本来就不同，跨叶比较会把反序掩盖掉，
    而门限是跨叶的一条线，所以叶内反序会直接变成门限的反向筛选。
    """
    grouped: dict[str, list[tuple[float, int]]] = {}
    for item, truth in zip(predictions, labels):
        if item is None or item.get("verdict") is None:
            continue
        grouped.setdefault(str(item["path"]), []).append(
            (float(item["confidence_lower_bound"]), int(item["verdict"] == truth))
        )
    return {
        path: confidence_auc(pairs)
        for path, pairs in grouped.items()
        if confidence_auc(pairs) is not None and len(pairs) >= 10
    }


def test_leave_one_out_confidence_is_inverted_within_every_leaf(train_split):
    """记录被修掉的那个缺陷本身：留一法置信度在叶内是反序的。

    这个断言不是在保护 `_loo_sop_predictions` 的行为，而是在保证
    「为什么不能用它反解门限」这条理由始终可复现。一旦哪天它不再反序，
    说明上游拟合逻辑变了，迭代 2 的结论需要重新审。
    """
    features, labels, sop = train_split
    aucs = within_leaf_aucs(_loo_sop_predictions(features, labels, sop=sop), labels)
    assert aucs, "expected at least one leaf with both correct and incorrect cases"
    for path, auc in aucs.items():
        assert auc < 0.5, f"leaf {path} unexpectedly not inverted under LOO: AUC={auc}"


def test_out_of_fold_confidence_is_not_inverted(train_split):
    """折外置信度必须至少不反序：被留出 case 的标签不再是唯一扰动源。"""
    features, labels, sop = train_split
    predictions = _out_of_fold_sop_predictions(features, labels, sop=sop, folds=5)
    assert all(item is not None for item in predictions)
    aucs = within_leaf_aucs(predictions, labels)
    assert aucs
    loo_aucs = within_leaf_aucs(_loo_sop_predictions(features, labels, sop=sop), labels)
    assert max(aucs.values()) > max(loo_aucs.values())
    # 合并口径下必须站在 0.5 的正确一侧，否则门限依然是反向筛选。
    pooled = confidence_auc(
        [
            (float(item["confidence_lower_bound"]), int(item["verdict"] == truth))
            for item, truth in zip(predictions, labels)
            if item is not None and item.get("verdict") is not None
        ]
    )
    assert pooled is not None and pooled > 0.5


def test_out_of_fold_predictions_never_come_from_a_model_that_saw_the_case(train_split):
    """折外的无泄漏性来自 fold 划分，这里直接核对划分被真的用上了。

    如果实现退化成「用全量模型预测」，同一条 case 的置信度会与全量叶纯度一致；
    折外实现下至少要有一部分 case 的置信度与全量模型不同。
    """
    features, labels, sop = train_split
    predictions = _out_of_fold_sop_predictions(features, labels, sop=sop, folds=5)
    full = [sop.predict(feature).to_dict() for feature in features]
    differing = sum(
        1
        for oof, whole in zip(predictions, full)
        if oof is not None
        and abs(float(oof["confidence_lower_bound"]) - float(whole["confidence_lower_bound"]))
        > 1e-9
    )
    assert differing > 0
