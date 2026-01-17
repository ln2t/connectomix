"""Atlas data management for Connectomix."""

from connectomix.data.atlases import (
    load_atlas,
    list_available_atlases,
    load_custom_atlas,
    get_atlas_info,
    get_atlas_labels,
    get_atlas_coords,
    clear_atlas_cache,
    validate_atlas,
    get_atlas_resolution,
    ATLAS_REGISTRY,
)

__all__ = [
    "load_atlas",
    "list_available_atlases",
    "load_custom_atlas",
    "get_atlas_info",
    "get_atlas_labels",
    "get_atlas_coords",
    "clear_atlas_cache",
    "validate_atlas",
    "get_atlas_resolution",
    "ATLAS_REGISTRY",
]
