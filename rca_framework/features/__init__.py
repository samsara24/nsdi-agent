"""M1 可解释特征层：特征字典 v1 与证据到稀疏特征向量的抽取器。

本子包不被 legacy 路径引用，因此不影响 `rca_framework.cli` 的 58/85 回归锚点。
"""

from .dictionary import (
    FEATURE_DICTIONARY,
    FEATURE_DICTIONARY_VERSION,
    FeatureDictionary,
    FeatureFamily,
    dictionary_for,
)
from .extractor import (
    CaseFeatures,
    FeatureModel,
    detect_token_conflicts,
    extract_feature_tokens,
    extract_features,
    fit_feature_model,
)

__all__ = [
    "FEATURE_DICTIONARY",
    "FEATURE_DICTIONARY_VERSION",
    "CaseFeatures",
    "FeatureDictionary",
    "FeatureFamily",
    "FeatureModel",
    "detect_token_conflicts",
    "dictionary_for",
    "extract_feature_tokens",
    "extract_features",
    "fit_feature_model",
]
