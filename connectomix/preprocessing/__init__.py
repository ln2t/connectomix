"""Preprocessing functions for functional images."""

from connectomix.preprocessing.resampling import (
    resample_to_reference,
    check_geometric_consistency,
    save_geometry_info,
    validate_group_geometry,
)
from connectomix.preprocessing.denoising import denoise_image
from connectomix.preprocessing.canica import run_canica_atlas

__all__ = [
    "resample_to_reference",
    "check_geometric_consistency",
    "save_geometry_info",
    "validate_group_geometry",
    "denoise_image",
    "run_canica_atlas",
]
