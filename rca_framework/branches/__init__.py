"""N5a / N5b / N5c 三分支处理器与 N6 弃权出口。

三个分支的共同输出是 `BranchOutcome`，它带一个**标定过的**置信度：
不是常数，而是训练集留一法上该分组的实测正确率，另附 Wilson 95% 置信下界。
"""

from .base import (
    BranchCalibration,
    BranchOutcome,
    EvidenceLink,
    majority_label,
    wilson_lower_bound,
)
from .dispatch import (
    calibration_group_of,
    fit_calibration,
    handle,
    handle_many,
)
from .general import DiagnosisRequest, Exclusion, build_request

__all__ = [
    "BranchCalibration",
    "BranchOutcome",
    "DiagnosisRequest",
    "EvidenceLink",
    "Exclusion",
    "build_request",
    "calibration_group_of",
    "fit_calibration",
    "handle",
    "handle_many",
    "majority_label",
    "wilson_lower_bound",
]
