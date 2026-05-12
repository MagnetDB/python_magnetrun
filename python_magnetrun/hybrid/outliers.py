"""
Backward-compatibility shim.

The outlier detection module has been promoted to the package root:
    python_magnetrun.outliers

All public names are re-exported here so that existing imports of
``python_magnetrun.hybrid.outliers`` continue to work unchanged.
"""

from ..outliers import (  # noqa: F401
    OUTLIER_DEFAULTS,
    OutlierConfig,
    OutlierDetector,
    OutlierMethod,
    OutlierResult,
    analyze_outliers,
    detect_outliers,
    find_outlier_segments,
    get_outlier_summary,
    remove_outliers,
)
