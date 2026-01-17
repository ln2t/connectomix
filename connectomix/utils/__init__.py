"""Utility functions for Connectomix."""

from connectomix.utils.logging import setup_logging, timer
from connectomix.utils.validation import validate_alpha, validate_positive, validate_file_exists
from connectomix.utils.matrix import sym_matrix_to_vec, vec_to_sym_matrix
from connectomix.utils.visualization import (
    plot_design_matrix,
    plot_connectivity_matrix,
    plot_stat_map,
    plot_glass_brain,
    plot_seeds,
    plot_cluster_locations,
    plot_qc_metrics,
    close_all_figures,
)
from connectomix.utils.reports import (
    HTMLReportGenerator,
    generate_participant_report,
    generate_group_report,
)

__all__ = [
    # Logging
    "setup_logging",
    "timer",
    # Validation
    "validate_alpha",
    "validate_positive",
    "validate_file_exists",
    # Matrix utilities
    "sym_matrix_to_vec",
    "vec_to_sym_matrix",
    # Visualization
    "plot_design_matrix",
    "plot_connectivity_matrix",
    "plot_stat_map",
    "plot_glass_brain",
    "plot_seeds",
    "plot_cluster_locations",
    "plot_qc_metrics",
    "close_all_figures",
    # Reports
    "HTMLReportGenerator",
    "generate_participant_report",
    "generate_group_report",
]
