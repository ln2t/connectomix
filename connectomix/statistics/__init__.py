"""Statistical analysis for group-level connectivity.

This module provides tools for group-level statistical analysis including:
- Second-level GLM fitting and contrast computation
- Permutation testing for FWE correction
- Multiple comparison correction (uncorrected, FDR, FWE, cluster)
- Cluster analysis and anatomical labeling
"""

from connectomix.statistics.glm import (
    build_design_matrix,
    fit_second_level_model,
    compute_contrast,
    save_design_matrix,
)
from connectomix.statistics.permutation import (
    run_permutation_test,
    compute_fwe_threshold,
)
from connectomix.statistics.thresholding import (
    apply_threshold,
    apply_uncorrected_threshold,
    apply_fdr_threshold,
    apply_fwe_threshold,
    apply_cluster_threshold,
    fdr_threshold,
    fwe_threshold,
)
from connectomix.statistics.clustering import (
    get_cluster_table,
    add_anatomical_labels,
    label_clusters,
    save_cluster_table,
)

__all__ = [
    # GLM
    "build_design_matrix",
    "fit_second_level_model",
    "compute_contrast",
    "save_design_matrix",
    # Permutation
    "run_permutation_test",
    "compute_fwe_threshold",
    # Thresholding
    "apply_threshold",
    "apply_uncorrected_threshold",
    "apply_fdr_threshold",
    "apply_fwe_threshold",
    "apply_cluster_threshold",
    "fdr_threshold",
    "fwe_threshold",
    # Clustering
    "get_cluster_table",
    "add_anatomical_labels",
    "label_clusters",
    "save_cluster_table",
]
