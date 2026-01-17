"""Multiple comparison correction and thresholding.

This module provides functions for applying various thresholding strategies
to statistical maps, including uncorrected thresholds, FDR correction,
FWE correction, and cluster-based thresholding.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
from nilearn.glm import threshold_stats_img
from nilearn.image import get_data, new_img_like, math_img
from scipy import stats

from connectomix.utils.exceptions import StatisticalError

logger = logging.getLogger(__name__)


def apply_threshold(
    stat_map: Union[str, Path, nib.Nifti1Image],
    threshold: float,
    two_sided: bool = True,
    cluster_threshold: Optional[int] = None,
) -> nib.Nifti1Image:
    """Apply a simple threshold to a statistical map.
    
    Args:
        stat_map: Statistical map (path or NIfTI image).
        threshold: Threshold value. Voxels with |stat| > threshold are kept
            for two-sided tests, or stat > threshold for one-sided.
        two_sided: If True, threshold both positive and negative values.
        cluster_threshold: Minimum cluster size in voxels. Clusters smaller
            than this are removed. None for no cluster thresholding.
    
    Returns:
        Thresholded NIfTI image with sub-threshold voxels set to 0.
    
    Example:
        >>> thresholded = apply_threshold(t_map, threshold=3.0)
    """
    # Load image if path
    if isinstance(stat_map, (str, Path)):
        stat_map = nib.load(str(stat_map))
    
    data = get_data(stat_map).copy()
    
    if two_sided:
        # Keep voxels where |stat| > threshold
        mask = np.abs(data) > threshold
    else:
        # Keep voxels where stat > threshold (positive only)
        mask = data > threshold
    
    # Apply threshold
    thresholded_data = np.where(mask, data, 0)
    
    # Apply cluster threshold if requested
    if cluster_threshold is not None and cluster_threshold > 0:
        thresholded_data = _remove_small_clusters(
            thresholded_data, cluster_threshold
        )
    
    thresholded_img = new_img_like(stat_map, thresholded_data)
    
    n_surviving = np.sum(thresholded_data != 0)
    logger.debug(
        f"Applied threshold {threshold:.3f}: "
        f"{n_surviving} voxels survive"
    )
    
    return thresholded_img


def apply_uncorrected_threshold(
    stat_map: Union[str, Path, nib.Nifti1Image],
    alpha: float = 0.001,
    two_sided: bool = True,
) -> Tuple[nib.Nifti1Image, float]:
    """Apply uncorrected p-value threshold to statistical map.
    
    Converts p-value threshold to t/z threshold and applies it.
    
    Args:
        stat_map: Statistical map (t or z statistics).
        alpha: P-value threshold (default 0.001).
        two_sided: If True, use two-tailed threshold.
    
    Returns:
        Tuple of (thresholded image, threshold value used).
    
    Example:
        >>> thresholded, threshold = apply_uncorrected_threshold(t_map, alpha=0.001)
        >>> print(f"Threshold used: {threshold:.3f}")
    """
    if not 0 < alpha < 1:
        raise StatisticalError(
            f"alpha must be between 0 and 1, got {alpha}"
        )
    
    # Compute threshold from normal distribution
    if two_sided:
        threshold = stats.norm.ppf(1 - alpha / 2)
    else:
        threshold = stats.norm.ppf(1 - alpha)
    
    logger.info(
        f"Uncorrected threshold: alpha={alpha}, "
        f"{'two-sided' if two_sided else 'one-sided'}, "
        f"threshold={threshold:.3f}"
    )
    
    thresholded_img = apply_threshold(stat_map, threshold, two_sided)
    
    return thresholded_img, threshold


def apply_fdr_threshold(
    stat_map: Union[str, Path, nib.Nifti1Image],
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Tuple[nib.Nifti1Image, float]:
    """Apply FDR (False Discovery Rate) correction to statistical map.
    
    Uses the Benjamini-Hochberg procedure for FDR correction.
    
    Args:
        stat_map: Statistical map (t or z statistics).
        alpha: FDR level (default 0.05).
        two_sided: If True, use two-tailed test.
    
    Returns:
        Tuple of (thresholded image, threshold value used).
        If no voxels survive FDR correction, threshold is np.inf.
    
    Example:
        >>> thresholded, threshold = apply_fdr_threshold(t_map, alpha=0.05)
    """
    if not 0 < alpha < 1:
        raise StatisticalError(
            f"alpha must be between 0 and 1, got {alpha}"
        )
    
    # Load image if path
    if isinstance(stat_map, (str, Path)):
        stat_map = nib.load(str(stat_map))
    
    try:
        # Use nilearn's built-in FDR thresholding
        thresholded_img, threshold = threshold_stats_img(
            stat_map,
            alpha=alpha,
            height_control="fdr",
            two_sided=two_sided,
        )
        
        logger.info(
            f"FDR threshold: alpha={alpha}, "
            f"{'two-sided' if two_sided else 'one-sided'}, "
            f"threshold={threshold:.3f}"
        )
        
    except Exception as e:
        # No voxels survive FDR correction
        logger.warning(
            f"No voxels survive FDR correction at alpha={alpha}. "
            f"Returning empty map. Error: {e}"
        )
        data = get_data(stat_map)
        thresholded_img = new_img_like(stat_map, np.zeros_like(data))
        threshold = np.inf
    
    return thresholded_img, threshold


def apply_fwe_threshold(
    stat_map: Union[str, Path, nib.Nifti1Image],
    null_distribution: np.ndarray,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Tuple[nib.Nifti1Image, float]:
    """Apply FWE (Family-Wise Error) correction using permutation null.
    
    Uses the null distribution from permutation testing to determine
    the FWE-corrected threshold.
    
    Args:
        stat_map: Statistical map (t or z statistics).
        null_distribution: Max statistics from permutation test
            (from run_permutation_test).
        alpha: FWE level (default 0.05).
        two_sided: If True, use absolute values.
    
    Returns:
        Tuple of (thresholded image, threshold value used).
    
    Example:
        >>> results = run_permutation_test(...)
        >>> thresholded, threshold = apply_fwe_threshold(
        ...     t_map, results['null_distribution'], alpha=0.05
        ... )
    """
    if not 0 < alpha < 1:
        raise StatisticalError(
            f"alpha must be between 0 and 1, got {alpha}"
        )
    
    # Compute threshold from null distribution
    percentile = 100 * (1 - alpha)
    threshold = np.percentile(null_distribution, percentile)
    
    logger.info(
        f"FWE threshold: alpha={alpha}, "
        f"{'two-sided' if two_sided else 'one-sided'}, "
        f"threshold={threshold:.3f} "
        f"(from {len(null_distribution)} permutations)"
    )
    
    thresholded_img = apply_threshold(stat_map, threshold, two_sided)
    
    return thresholded_img, threshold


def apply_cluster_threshold(
    stat_map: Union[str, Path, nib.Nifti1Image],
    cluster_forming_alpha: float = 0.01,
    min_cluster_size: int = 50,
    two_sided: bool = True,
) -> Tuple[nib.Nifti1Image, float, int]:
    """Apply cluster-based thresholding.
    
    First applies a cluster-forming threshold, then removes clusters
    smaller than the specified minimum size.
    
    Args:
        stat_map: Statistical map (t or z statistics).
        cluster_forming_alpha: P-value for cluster-forming threshold.
        min_cluster_size: Minimum cluster size in voxels.
        two_sided: If True, use two-tailed threshold.
    
    Returns:
        Tuple of (thresholded image, threshold value, number of surviving clusters).
    
    Example:
        >>> thresholded, threshold, n_clusters = apply_cluster_threshold(
        ...     t_map, cluster_forming_alpha=0.01, min_cluster_size=50
        ... )
    """
    # Load image if path
    if isinstance(stat_map, (str, Path)):
        stat_map = nib.load(str(stat_map))
    
    # Compute cluster-forming threshold
    if two_sided:
        threshold = stats.norm.ppf(1 - cluster_forming_alpha / 2)
    else:
        threshold = stats.norm.ppf(1 - cluster_forming_alpha)
    
    data = get_data(stat_map).copy()
    
    # Apply cluster-forming threshold
    if two_sided:
        mask = np.abs(data) > threshold
    else:
        mask = data > threshold
    
    thresholded_data = np.where(mask, data, 0)
    
    # Remove small clusters
    thresholded_data, n_clusters = _remove_small_clusters(
        thresholded_data, min_cluster_size, return_n_clusters=True
    )
    
    thresholded_img = new_img_like(stat_map, thresholded_data)
    
    logger.info(
        f"Cluster threshold: alpha={cluster_forming_alpha}, "
        f"threshold={threshold:.3f}, "
        f"min_size={min_cluster_size}, "
        f"surviving_clusters={n_clusters}"
    )
    
    return thresholded_img, threshold, n_clusters


def _remove_small_clusters(
    data: np.ndarray,
    min_size: int,
    return_n_clusters: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Remove clusters smaller than minimum size.
    
    Args:
        data: 3D array with non-zero values indicating activation.
        min_size: Minimum cluster size in voxels.
        return_n_clusters: If True, also return number of surviving clusters.
    
    Returns:
        Filtered data array (and optionally number of clusters).
    """
    from scipy import ndimage
    
    # Create binary mask of non-zero voxels
    mask = data != 0
    
    # Label connected components (6-connectivity for 3D)
    labeled, n_features = ndimage.label(mask)
    
    # Find cluster sizes
    cluster_sizes = ndimage.sum(mask, labeled, range(1, n_features + 1))
    
    # Create mask of clusters to keep
    keep_clusters = np.array([size >= min_size for size in cluster_sizes])
    
    # Build output mask
    output_mask = np.zeros_like(mask)
    for i, keep in enumerate(keep_clusters, start=1):
        if keep:
            output_mask |= (labeled == i)
    
    # Apply mask to data
    filtered_data = np.where(output_mask, data, 0)
    
    n_surviving = np.sum(keep_clusters)
    
    if return_n_clusters:
        return filtered_data, n_surviving
    return filtered_data


def threshold_stats_to_binary(
    stat_map: Union[str, Path, nib.Nifti1Image],
    threshold: float,
    two_sided: bool = True,
) -> nib.Nifti1Image:
    """Create binary mask from thresholded statistical map.
    
    Args:
        stat_map: Statistical map.
        threshold: Threshold value.
        two_sided: If True, include both positive and negative tails.
    
    Returns:
        Binary NIfTI image (1 = significant, 0 = not significant).
    """
    if isinstance(stat_map, (str, Path)):
        stat_map = nib.load(str(stat_map))
    
    data = get_data(stat_map)
    
    if two_sided:
        binary = (np.abs(data) > threshold).astype(np.int8)
    else:
        binary = (data > threshold).astype(np.int8)
    
    return new_img_like(stat_map, binary)


# Aliases for backwards compatibility with __init__.py
def fdr_threshold(
    stat_map: Union[str, Path, nib.Nifti1Image],
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Tuple[nib.Nifti1Image, float]:
    """Alias for apply_fdr_threshold."""
    return apply_fdr_threshold(stat_map, alpha, two_sided)


def fwe_threshold(
    stat_map: Union[str, Path, nib.Nifti1Image],
    null_distribution: np.ndarray,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Tuple[nib.Nifti1Image, float]:
    """Alias for apply_fwe_threshold."""
    return apply_fwe_threshold(stat_map, null_distribution, alpha, two_sided)
