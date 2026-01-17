"""Cluster analysis and anatomical labeling.

This module provides functions for extracting cluster information from
thresholded statistical maps and adding anatomical labels.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import get_data
from nilearn.reporting import get_clusters_table

from connectomix.utils.exceptions import StatisticalError

logger = logging.getLogger(__name__)


def get_cluster_table(
    stat_map: Union[str, Path, nib.Nifti1Image],
    threshold: float,
    min_cluster_size: int = 10,
    two_sided: bool = True,
) -> pd.DataFrame:
    """Extract cluster information from a statistical map.
    
    Uses nilearn's get_clusters_table to identify clusters and extract
    their properties including size, peak coordinates, and peak values.
    
    Args:
        stat_map: Statistical map (path or NIfTI image).
        threshold: Cluster-forming threshold (applied before extraction).
        min_cluster_size: Minimum cluster size in voxels (default 10).
        two_sided: If True, include both positive and negative clusters.
    
    Returns:
        DataFrame with columns:
            - Cluster ID: Unique cluster identifier
            - X, Y, Z: Peak coordinates in MNI space
            - Peak Stat: Peak statistic value
            - Cluster Size (mm3): Cluster volume
            - Cluster Size (voxels): Number of voxels
    
    Example:
        >>> clusters = get_cluster_table(t_map, threshold=3.0)
        >>> print(clusters)
           Cluster ID    X    Y    Z  Peak Stat  Cluster Size (mm3)
        0           1  -45  -70   35       5.23                2500
        1           2    0   52    0       4.87                1800
    """
    # Load image if path
    if isinstance(stat_map, (str, Path)):
        stat_map = nib.load(str(stat_map))
    
    # Get cluster table from nilearn
    # Note: nilearn's get_clusters_table returns different column names
    # depending on version, so we'll standardize them
    try:
        clusters_df = get_clusters_table(
            stat_map,
            stat_threshold=threshold,
            cluster_threshold=min_cluster_size,
            two_sided=two_sided,
        )
    except Exception as e:
        logger.warning(f"Could not extract clusters: {e}")
        return pd.DataFrame(columns=[
            "Cluster ID", "X", "Y", "Z", "Peak Stat",
            "Cluster Size (mm3)", "Cluster Size (voxels)"
        ])
    
    if clusters_df.empty:
        logger.info("No clusters found above threshold")
        return pd.DataFrame(columns=[
            "Cluster ID", "X", "Y", "Z", "Peak Stat",
            "Cluster Size (mm3)", "Cluster Size (voxels)"
        ])
    
    # Standardize column names
    clusters_df = _standardize_cluster_columns(clusters_df, stat_map)
    
    logger.info(f"Found {len(clusters_df)} clusters above threshold")
    
    return clusters_df


def _standardize_cluster_columns(
    df: pd.DataFrame,
    stat_map: nib.Nifti1Image,
) -> pd.DataFrame:
    """Standardize column names from nilearn's get_clusters_table.
    
    Different versions of nilearn use different column names, so we
    standardize them here.
    """
    # Create standardized output
    output = pd.DataFrame()
    
    # Cluster ID
    if "Cluster ID" in df.columns:
        output["Cluster ID"] = df["Cluster ID"]
    else:
        output["Cluster ID"] = range(1, len(df) + 1)
    
    # Coordinates (X, Y, Z)
    coord_cols = [c for c in df.columns if c.lower() in ["x", "y", "z"]]
    if len(coord_cols) == 3:
        output["X"] = df[coord_cols[0]]
        output["Y"] = df[coord_cols[1]]
        output["Z"] = df[coord_cols[2]]
    elif "Peak coordinates" in df.columns:
        # Some versions return coordinates as a single column
        coords = df["Peak coordinates"].apply(eval)
        output["X"] = coords.apply(lambda x: x[0])
        output["Y"] = coords.apply(lambda x: x[1])
        output["Z"] = coords.apply(lambda x: x[2])
    
    # Peak statistic
    stat_cols = [c for c in df.columns if "peak" in c.lower() and "stat" in c.lower()]
    if stat_cols:
        output["Peak Stat"] = df[stat_cols[0]]
    elif "Peak Stat" in df.columns:
        output["Peak Stat"] = df["Peak Stat"]
    
    # Cluster size
    size_cols = [c for c in df.columns if "size" in c.lower() or "volume" in c.lower()]
    if size_cols:
        output["Cluster Size (mm3)"] = df[size_cols[0]]
    
    # Calculate voxel count if not provided
    if "Cluster Size (voxels)" not in output.columns and "Cluster Size (mm3)" in output.columns:
        voxel_volume = np.prod(stat_map.header.get_zooms()[:3])
        output["Cluster Size (voxels)"] = (
            output["Cluster Size (mm3)"] / voxel_volume
        ).astype(int)
    
    return output


def add_anatomical_labels(
    cluster_table: pd.DataFrame,
    atlas_name: str = "aal",
) -> pd.DataFrame:
    """Add anatomical region labels to cluster table.
    
    Uses atlas data to identify the anatomical region at each peak coordinate.
    
    Args:
        cluster_table: DataFrame from get_cluster_table().
        atlas_name: Name of atlas to use for labeling.
            Options: "aal", "harvardoxford", "destrieux", "juelich"
    
    Returns:
        DataFrame with additional "Region" column containing anatomical labels.
    
    Example:
        >>> clusters = get_cluster_table(t_map, threshold=3.0)
        >>> clusters = add_anatomical_labels(clusters, atlas_name="aal")
        >>> print(clusters["Region"])
        0    Precuneus_L
        1    Frontal_Med_Orb_L
    """
    if cluster_table.empty:
        cluster_table["Region"] = pd.Series(dtype=str)
        return cluster_table
    
    # Load atlas
    atlas_img, atlas_labels = _load_labeling_atlas(atlas_name)
    
    if atlas_img is None:
        logger.warning(f"Could not load atlas '{atlas_name}' for labeling")
        cluster_table["Region"] = "Unknown"
        return cluster_table
    
    # Get coordinates
    coords = cluster_table[["X", "Y", "Z"]].values
    
    # Label each peak coordinate
    labels = []
    for x, y, z in coords:
        label = _get_region_label_at_coord(
            x, y, z, atlas_img, atlas_labels
        )
        labels.append(label)
    
    cluster_table = cluster_table.copy()
    cluster_table["Region"] = labels
    
    return cluster_table


def _load_labeling_atlas(
    atlas_name: str,
) -> Tuple[Optional[nib.Nifti1Image], Optional[Dict[int, str]]]:
    """Load atlas for anatomical labeling.
    
    Args:
        atlas_name: Name of atlas.
    
    Returns:
        Tuple of (atlas_image, labels_dict) or (None, None) if not available.
    """
    try:
        if atlas_name.lower() == "aal":
            from nilearn.datasets import fetch_atlas_aal
            atlas = fetch_atlas_aal()
            atlas_img = nib.load(atlas["maps"])
            # AAL labels are 1-indexed
            labels = {i + 1: name for i, name in enumerate(atlas["labels"])}
            labels[0] = "Background"
            
        elif atlas_name.lower() == "harvardoxford":
            from nilearn.datasets import fetch_atlas_harvard_oxford
            atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
            atlas_img = atlas["maps"]
            if isinstance(atlas_img, str):
                atlas_img = nib.load(atlas_img)
            labels = {i: name for i, name in enumerate(atlas["labels"])}
            
        elif atlas_name.lower() == "destrieux":
            from nilearn.datasets import fetch_atlas_destrieux_2009
            atlas = fetch_atlas_destrieux_2009()
            atlas_img = nib.load(atlas["maps"])
            labels = {i: name for i, name in enumerate(atlas["labels"])}
            
        else:
            logger.warning(f"Unknown atlas: {atlas_name}")
            return None, None
        
        logger.debug(f"Loaded {atlas_name} atlas with {len(labels)} regions")
        return atlas_img, labels
        
    except Exception as e:
        logger.warning(f"Error loading atlas {atlas_name}: {e}")
        return None, None


def _get_region_label_at_coord(
    x: float,
    y: float,
    z: float,
    atlas_img: nib.Nifti1Image,
    labels: Dict[int, str],
) -> str:
    """Get anatomical label at a coordinate.
    
    Args:
        x, y, z: MNI coordinates.
        atlas_img: Atlas NIfTI image.
        labels: Dictionary mapping label indices to names.
    
    Returns:
        Region name or "Unknown" if not in atlas.
    """
    from nilearn.image import coord_transform
    
    try:
        # Convert MNI coordinates to voxel indices
        affine = atlas_img.affine
        inv_affine = np.linalg.inv(affine)
        
        # Apply inverse affine to get voxel coords
        coord_mni = np.array([x, y, z, 1])
        voxel = inv_affine.dot(coord_mni)[:3]
        
        # Round to nearest voxel
        i, j, k = [int(round(c)) for c in voxel]
        
        # Check bounds
        atlas_data = get_data(atlas_img)
        if (0 <= i < atlas_data.shape[0] and
            0 <= j < atlas_data.shape[1] and
            0 <= k < atlas_data.shape[2]):
            
            label_idx = int(atlas_data[i, j, k])
            return labels.get(label_idx, f"Region_{label_idx}")
        else:
            return "Outside atlas"
            
    except Exception as e:
        logger.debug(f"Error getting label at ({x}, {y}, {z}): {e}")
        return "Unknown"


def label_clusters(
    stat_map: Union[str, Path, nib.Nifti1Image],
    threshold: float,
    atlas_name: str = "aal",
    min_cluster_size: int = 10,
    two_sided: bool = True,
) -> pd.DataFrame:
    """Extract clusters and add anatomical labels in one step.
    
    Convenience function that combines get_cluster_table and
    add_anatomical_labels.
    
    Args:
        stat_map: Statistical map.
        threshold: Cluster-forming threshold.
        atlas_name: Atlas for labeling (default "aal").
        min_cluster_size: Minimum cluster size in voxels.
        two_sided: Include positive and negative clusters.
    
    Returns:
        DataFrame with cluster info and anatomical labels.
    
    Example:
        >>> clusters = label_clusters(t_map, threshold=3.0, atlas_name="aal")
    """
    # Get cluster table
    cluster_table = get_cluster_table(
        stat_map,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        two_sided=two_sided,
    )
    
    # Add anatomical labels
    if not cluster_table.empty:
        cluster_table = add_anatomical_labels(cluster_table, atlas_name)
    
    return cluster_table


def save_cluster_table(
    cluster_table: pd.DataFrame,
    output_path: Path,
    metadata: Optional[Dict] = None,
) -> Path:
    """Save cluster table to TSV with JSON sidecar.
    
    Args:
        cluster_table: DataFrame with cluster information.
        output_path: Output path for TSV file.
        metadata: Additional metadata for sidecar.
    
    Returns:
        Path to saved TSV file.
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save TSV
    cluster_table.to_csv(output_path, sep="\t", index=False)
    
    # Prepare sidecar
    sidecar = {
        "Description": "Cluster table from thresholded statistical map",
        "NumberOfClusters": len(cluster_table),
        "Columns": {
            "Cluster ID": "Unique cluster identifier",
            "X": "Peak coordinate X (MNI space, mm)",
            "Y": "Peak coordinate Y (MNI space, mm)",
            "Z": "Peak coordinate Z (MNI space, mm)",
            "Peak Stat": "Peak statistic value in cluster",
            "Cluster Size (mm3)": "Cluster volume in cubic millimeters",
            "Cluster Size (voxels)": "Number of voxels in cluster",
        },
    }
    
    if "Region" in cluster_table.columns:
        sidecar["Columns"]["Region"] = "Anatomical region at peak coordinate"
    
    if metadata:
        sidecar.update(metadata)
    
    sidecar_path = output_path.with_suffix(".json")
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2)
    
    logger.info(f"Saved cluster table to {output_path}")
    
    return output_path
