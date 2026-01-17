"""Atlas management for Connectomix.

This module provides centralized atlas loading and caching functionality.
Atlases are used for ROI-based connectivity analysis and anatomical labeling.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


# Registry of supported atlases with their nilearn identifiers and properties
ATLAS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "aal": {
        "name": "AAL (Automated Anatomical Labeling)",
        "source": "nilearn",
        "function": "fetch_atlas_aal",
        "n_rois": 116,
        "resolution": "2mm",
        "description": "Anatomical parcellation based on sulcal boundaries",
    },
    "schaefer_100": {
        "name": "Schaefer 100 Parcels (7 Networks)",
        "source": "nilearn",
        "function": "fetch_atlas_schaefer_2018",
        "kwargs": {"n_rois": 100, "yeo_networks": 7},
        "n_rois": 100,
        "resolution": "2mm",
        "description": "Functional parcellation, 7 network version",
    },
    "schaefer_200": {
        "name": "Schaefer 200 Parcels (7 Networks)",
        "source": "nilearn",
        "function": "fetch_atlas_schaefer_2018",
        "kwargs": {"n_rois": 200, "yeo_networks": 7},
        "n_rois": 200,
        "resolution": "2mm",
        "description": "Functional parcellation, 7 network version",
    },
    "schaefer_400": {
        "name": "Schaefer 400 Parcels (7 Networks)",
        "source": "nilearn",
        "function": "fetch_atlas_schaefer_2018",
        "kwargs": {"n_rois": 400, "yeo_networks": 7},
        "n_rois": 400,
        "resolution": "2mm",
        "description": "Functional parcellation, 7 network version",
    },
    "schaefer_100_17": {
        "name": "Schaefer 100 Parcels (17 Networks)",
        "source": "nilearn",
        "function": "fetch_atlas_schaefer_2018",
        "kwargs": {"n_rois": 100, "yeo_networks": 17},
        "n_rois": 100,
        "resolution": "2mm",
        "description": "Functional parcellation, 17 network version",
    },
    "schaefer_200_17": {
        "name": "Schaefer 200 Parcels (17 Networks)",
        "source": "nilearn",
        "function": "fetch_atlas_schaefer_2018",
        "kwargs": {"n_rois": 200, "yeo_networks": 17},
        "n_rois": 200,
        "resolution": "2mm",
        "description": "Functional parcellation, 17 network version",
    },
    "harvard_oxford_cort": {
        "name": "Harvard-Oxford Cortical",
        "source": "nilearn",
        "function": "fetch_atlas_harvard_oxford",
        "kwargs": {"atlas_name": "cort-maxprob-thr25-2mm"},
        "n_rois": 48,
        "resolution": "2mm",
        "description": "Probabilistic cortical atlas (25% threshold)",
    },
    "harvard_oxford_sub": {
        "name": "Harvard-Oxford Subcortical",
        "source": "nilearn",
        "function": "fetch_atlas_harvard_oxford",
        "kwargs": {"atlas_name": "sub-maxprob-thr25-2mm"},
        "n_rois": 21,
        "resolution": "2mm",
        "description": "Probabilistic subcortical atlas (25% threshold)",
    },
    "destrieux": {
        "name": "Destrieux Atlas",
        "source": "nilearn",
        "function": "fetch_atlas_destrieux_2009",
        "n_rois": 148,
        "resolution": "2mm",
        "description": "Sulcal-based parcellation (FreeSurfer)",
    },
    "difumo_64": {
        "name": "DiFuMo 64 Components",
        "source": "nilearn",
        "function": "fetch_atlas_difumo",
        "kwargs": {"dimension": 64},
        "n_rois": 64,
        "resolution": "2mm",
        "description": "Dictionary-based functional atlas",
    },
    "difumo_128": {
        "name": "DiFuMo 128 Components",
        "source": "nilearn",
        "function": "fetch_atlas_difumo",
        "kwargs": {"dimension": 128},
        "n_rois": 128,
        "resolution": "2mm",
        "description": "Dictionary-based functional atlas",
    },
    "msdl": {
        "name": "MSDL Atlas",
        "source": "nilearn",
        "function": "fetch_atlas_msdl",
        "n_rois": 39,
        "resolution": "2mm",
        "description": "Multi-Subject Dictionary Learning atlas",
    },
}


def list_available_atlases() -> List[Dict[str, Any]]:
    """List all available atlases.
    
    Returns:
        List of dictionaries with atlas information.
    
    Example:
        >>> atlases = list_available_atlases()
        >>> for atlas in atlases:
        ...     print(f"{atlas['id']}: {atlas['name']} ({atlas['n_rois']} ROIs)")
    """
    atlases = []
    
    for atlas_id, info in ATLAS_REGISTRY.items():
        atlases.append({
            "id": atlas_id,
            "name": info["name"],
            "n_rois": info["n_rois"],
            "resolution": info["resolution"],
            "description": info["description"],
        })
    
    return atlases


def get_atlas_info(atlas_name: str) -> Dict[str, Any]:
    """Get information about a specific atlas.
    
    Args:
        atlas_name: Atlas identifier from ATLAS_REGISTRY.
    
    Returns:
        Dictionary with atlas metadata.
    
    Raises:
        ValueError: If atlas not found.
    """
    if atlas_name not in ATLAS_REGISTRY:
        available = ", ".join(ATLAS_REGISTRY.keys())
        raise ValueError(
            f"Unknown atlas: {atlas_name}. Available atlases: {available}"
        )
    
    return ATLAS_REGISTRY[atlas_name].copy()


@lru_cache(maxsize=16)
def load_atlas(atlas_name: str) -> Tuple[nib.Nifti1Image, List[str]]:
    """Load atlas image and labels.
    
    Uses LRU cache to avoid reloading frequently used atlases.
    
    Args:
        atlas_name: Atlas identifier from ATLAS_REGISTRY.
    
    Returns:
        Tuple of (atlas_img, labels) where atlas_img is a NIfTI image
        and labels is a list of region names.
    
    Raises:
        ValueError: If atlas not found or loading fails.
    
    Example:
        >>> atlas_img, labels = load_atlas("schaefer_100")
        >>> print(f"Loaded atlas with {len(labels)} regions")
    """
    if atlas_name not in ATLAS_REGISTRY:
        available = ", ".join(ATLAS_REGISTRY.keys())
        raise ValueError(
            f"Unknown atlas: {atlas_name}. Available atlases: {available}"
        )
    
    info = ATLAS_REGISTRY[atlas_name]
    
    logger.info(f"Loading atlas: {info['name']}")
    
    try:
        from nilearn import datasets
        
        # Get the fetch function
        fetch_func = getattr(datasets, info["function"])
        kwargs = info.get("kwargs", {})
        
        # Fetch atlas
        atlas_data = fetch_func(**kwargs)
        
        # Extract image and labels based on atlas type
        if info["function"] == "fetch_atlas_aal":
            atlas_img = nib.load(atlas_data.maps)
            labels = atlas_data.labels
            
        elif info["function"] == "fetch_atlas_schaefer_2018":
            atlas_img = nib.load(atlas_data.maps)
            labels = atlas_data.labels
            # Decode bytes to strings if necessary
            labels = [l.decode() if isinstance(l, bytes) else l for l in labels]
            
        elif info["function"] == "fetch_atlas_harvard_oxford":
            atlas_img = nib.load(atlas_data.maps)
            labels = atlas_data.labels
            
        elif info["function"] == "fetch_atlas_destrieux_2009":
            atlas_img = nib.load(atlas_data.maps)
            labels = atlas_data.labels
            labels = [l.decode() if isinstance(l, bytes) else l for l in labels]
            
        elif info["function"] == "fetch_atlas_difumo":
            atlas_img = nib.load(atlas_data.maps)
            labels = atlas_data.labels
            
        elif info["function"] == "fetch_atlas_msdl":
            # MSDL returns multiple maps, need special handling
            atlas_img = nib.load(atlas_data.maps)
            labels = atlas_data.labels
            
        else:
            raise ValueError(f"Unsupported fetch function: {info['function']}")
        
        # Ensure labels is a list of strings
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        
        logger.info(f"Loaded {len(labels)} regions from {atlas_name}")
        
        return atlas_img, labels
        
    except Exception as e:
        logger.error(f"Failed to load atlas {atlas_name}: {e}")
        raise ValueError(f"Failed to load atlas {atlas_name}: {e}") from e


def load_custom_atlas(
    atlas_path: Path,
    labels_path: Optional[Path] = None,
) -> Tuple[nib.Nifti1Image, List[str]]:
    """Load a custom atlas from file.
    
    Args:
        atlas_path: Path to NIfTI atlas image.
        labels_path: Optional path to labels file (one label per line).
    
    Returns:
        Tuple of (atlas_img, labels).
    
    Example:
        >>> atlas_img, labels = load_custom_atlas(
        ...     "my_atlas.nii.gz",
        ...     "my_labels.txt"
        ... )
    """
    atlas_path = Path(atlas_path)
    
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    
    logger.info(f"Loading custom atlas from: {atlas_path}")
    
    atlas_img = nib.load(atlas_path)
    
    # Extract unique labels from atlas
    atlas_data = atlas_img.get_fdata()
    unique_values = np.unique(atlas_data[atlas_data > 0]).astype(int)
    n_regions = len(unique_values)
    
    # Load labels file if provided
    if labels_path:
        labels_path = Path(labels_path)
        if not labels_path.exists():
            logger.warning(f"Labels file not found: {labels_path}")
            labels = [f"Region_{i}" for i in unique_values]
        else:
            with open(labels_path, "r") as f:
                labels = [line.strip() for line in f if line.strip()]
            
            if len(labels) != n_regions:
                logger.warning(
                    f"Number of labels ({len(labels)}) doesn't match "
                    f"number of regions ({n_regions}). Using generic labels."
                )
                labels = [f"Region_{i}" for i in unique_values]
    else:
        labels = [f"Region_{i}" for i in unique_values]
    
    logger.info(f"Loaded custom atlas with {n_regions} regions")
    
    return atlas_img, labels


def get_atlas_labels(atlas_name: str) -> List[str]:
    """Get just the labels for an atlas (without loading the image).
    
    Args:
        atlas_name: Atlas identifier.
    
    Returns:
        List of region labels.
    """
    _, labels = load_atlas(atlas_name)
    return labels


def get_atlas_coords(atlas_name: str) -> np.ndarray:
    """Get atlas region coordinates (centers of mass).
    
    Args:
        atlas_name: Atlas identifier.
    
    Returns:
        Array of shape (N, 3) with MNI coordinates.
    """
    from nilearn.image import math_img
    from nilearn.plotting import find_parcellation_cut_coords
    
    atlas_img, _ = load_atlas(atlas_name)
    
    # Get coordinates
    coords = find_parcellation_cut_coords(atlas_img)
    
    return coords


def clear_atlas_cache() -> None:
    """Clear the atlas cache to free memory."""
    load_atlas.cache_clear()
    logger.info("Cleared atlas cache")


def validate_atlas(atlas_name: str) -> bool:
    """Check if an atlas name is valid.
    
    Args:
        atlas_name: Atlas identifier to check.
    
    Returns:
        True if atlas is available, False otherwise.
    """
    return atlas_name in ATLAS_REGISTRY


def get_atlas_resolution(atlas_name: str) -> str:
    """Get the resolution of an atlas.
    
    Args:
        atlas_name: Atlas identifier.
    
    Returns:
        Resolution string (e.g., "2mm").
    """
    if atlas_name not in ATLAS_REGISTRY:
        raise ValueError(f"Unknown atlas: {atlas_name}")
    
    return ATLAS_REGISTRY[atlas_name]["resolution"]
