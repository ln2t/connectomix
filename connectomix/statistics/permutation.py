"""Permutation testing for family-wise error (FWE) correction.

This module implements non-parametric permutation testing for computing
FWE-corrected p-values using the maximum statistic approach.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import get_data, new_img_like
from nilearn.masking import apply_mask, unmask

from connectomix.utils.exceptions import StatisticalError
from connectomix.utils.logging import timer

logger = logging.getLogger(__name__)


def run_permutation_test(
    stat_maps: List[Union[str, Path, nib.Nifti1Image]],
    design_matrix: pd.DataFrame,
    contrast: Union[str, List[float], np.ndarray],
    n_permutations: int = 10000,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
    two_sided: bool = True,
    smoothing_fwhm: Optional[float] = None,
    mask_img: Optional[Union[str, Path, nib.Nifti1Image]] = None,
) -> Dict[str, Union[np.ndarray, float, nib.Nifti1Image]]:
    """Run permutation test for FWE-corrected inference.
    
    Performs non-parametric permutation testing using the max-T approach.
    For each permutation, subject labels are shuffled, the GLM is re-fitted,
    and the maximum absolute t-statistic is recorded. This builds a null
    distribution for FWE correction.
    
    Args:
        stat_maps: List of paths to NIfTI files or Nifti1Image objects.
            One map per subject in the same order as design matrix rows.
        design_matrix: Design matrix from build_design_matrix().
        contrast: Contrast definition (string or vector).
        n_permutations: Number of permutations to run. Higher values give
            more accurate p-values but take longer. Default 10000.
        n_jobs: Number of parallel jobs. -1 uses all available cores.
        random_state: Random seed for reproducibility.
        two_sided: If True, use absolute values for max statistic.
        smoothing_fwhm: FWHM in mm for spatial smoothing (applied once).
        mask_img: Brain mask for analysis.
    
    Returns:
        Dictionary containing:
            - 'null_distribution': Array of max statistics from permutations
            - 'observed_stat_map': Original (unpermuted) t-statistic map
            - 'p_values_map': Voxel-wise corrected p-values (NIfTI)
            - 'threshold_95': FWE threshold at alpha=0.05
            - 'threshold_99': FWE threshold at alpha=0.01
    
    Raises:
        StatisticalError: If inputs are invalid.
    
    Example:
        >>> results = run_permutation_test(
        ...     maps, design_matrix, contrast="group_patient",
        ...     n_permutations=5000, n_jobs=4, random_state=42
        ... )
        >>> fwe_threshold = results['threshold_95']
    """
    from connectomix.statistics.glm import (
        _string_to_contrast_vector,
        fit_second_level_model,
    )
    
    # Set random seed
    rng = np.random.RandomState(random_state)
    
    # Convert paths to strings
    stat_maps_list = [
        str(m) if isinstance(m, Path) else m for m in stat_maps
    ]
    
    n_subjects = len(stat_maps_list)
    
    logger.info(
        f"Running permutation test: {n_permutations} permutations, "
        f"{n_subjects} subjects, {n_jobs} jobs"
    )
    
    # Convert string contrast to vector if needed
    if isinstance(contrast, str):
        contrast_vector = _string_to_contrast_vector(contrast, design_matrix)
    else:
        contrast_vector = np.array(contrast)
    
    # Fit original model to get observed statistic
    with timer(logger, "Fitting original model"):
        original_model = fit_second_level_model(
            stat_maps_list,
            design_matrix,
            smoothing_fwhm=smoothing_fwhm,
            mask_img=mask_img,
        )
        observed_stat_map = original_model.compute_contrast(
            contrast_vector, stat_type="t", output_type="stat"
        )
    
    # Get mask from fitted model
    mask = original_model.masker_.mask_img_
    
    # Extract observed statistics as 1D array
    observed_stats = apply_mask(observed_stat_map, mask)
    
    if two_sided:
        observed_max = np.abs(observed_stats).max()
    else:
        observed_max = observed_stats.max()
    
    logger.info(f"Observed max statistic: {observed_max:.3f}")
    
    # Generate permutation indices
    permutation_indices = [
        rng.permutation(n_subjects) for _ in range(n_permutations)
    ]
    
    # Run permutations in parallel
    with timer(logger, f"Running {n_permutations} permutations"):
        if n_jobs == 1:
            # Sequential execution
            null_distribution = []
            for i, perm_idx in enumerate(permutation_indices):
                max_stat = _compute_permutation_max_stat(
                    stat_maps_list,
                    design_matrix,
                    contrast_vector,
                    perm_idx,
                    mask,
                    two_sided,
                )
                null_distribution.append(max_stat)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"  Completed {i + 1}/{n_permutations} permutations")
        else:
            # Parallel execution
            null_distribution = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_compute_permutation_max_stat)(
                    stat_maps_list,
                    design_matrix,
                    contrast_vector,
                    perm_idx,
                    mask,
                    two_sided,
                )
                for perm_idx in permutation_indices
            )
    
    null_distribution = np.array(null_distribution)
    
    # Compute thresholds
    threshold_95 = np.percentile(null_distribution, 95)
    threshold_99 = np.percentile(null_distribution, 99)
    
    logger.info(f"FWE thresholds: 95%={threshold_95:.3f}, 99%={threshold_99:.3f}")
    
    # Compute voxel-wise corrected p-values
    p_values = _compute_corrected_p_values(
        observed_stats, null_distribution, two_sided
    )
    
    # Convert back to NIfTI
    p_values_map = unmask(p_values, mask)
    
    return {
        "null_distribution": null_distribution,
        "observed_stat_map": observed_stat_map,
        "p_values_map": p_values_map,
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
    }


def _compute_permutation_max_stat(
    stat_maps: List[str],
    design_matrix: pd.DataFrame,
    contrast_vector: np.ndarray,
    permutation_indices: np.ndarray,
    mask: nib.Nifti1Image,
    two_sided: bool,
) -> float:
    """Compute max statistic for a single permutation.
    
    Internal function used by run_permutation_test.
    
    Args:
        stat_maps: List of stat map paths.
        design_matrix: Original design matrix.
        contrast_vector: Contrast vector.
        permutation_indices: Indices for permuting subjects.
        mask: Brain mask.
        two_sided: Whether to use absolute values.
    
    Returns:
        Maximum statistic for this permutation.
    """
    # Permute the stat maps (not the design matrix)
    permuted_maps = [stat_maps[i] for i in permutation_indices]
    
    # Fit model with permuted data
    model = SecondLevelModel(mask_img=mask)
    model.fit(permuted_maps, design_matrix=design_matrix)
    
    # Compute contrast
    stat_map = model.compute_contrast(
        contrast_vector, stat_type="t", output_type="stat"
    )
    
    # Get max statistic
    stats = apply_mask(stat_map, mask)
    
    if two_sided:
        return np.abs(stats).max()
    else:
        return stats.max()


def _compute_corrected_p_values(
    observed_stats: np.ndarray,
    null_distribution: np.ndarray,
    two_sided: bool,
) -> np.ndarray:
    """Compute FWE-corrected p-values for each voxel.
    
    Args:
        observed_stats: Observed t-statistics (1D array, one per voxel).
        null_distribution: Max statistics from permutations.
        two_sided: Whether test is two-sided.
    
    Returns:
        Array of corrected p-values (same shape as observed_stats).
    """
    n_permutations = len(null_distribution)
    
    if two_sided:
        # Count permutations where max |T| exceeds observed |t|
        observed_abs = np.abs(observed_stats)
        p_values = np.array([
            np.sum(null_distribution >= obs) / n_permutations
            for obs in observed_abs
        ])
    else:
        # Count permutations where max T exceeds observed t
        p_values = np.array([
            np.sum(null_distribution >= obs) / n_permutations
            for obs in observed_stats
        ])
    
    return p_values


def compute_fwe_threshold(
    null_distribution: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute FWE threshold from null distribution.
    
    Args:
        null_distribution: Max statistics from permutation test.
        alpha: Significance level (default 0.05).
    
    Returns:
        Threshold value for FWE correction at given alpha.
    
    Example:
        >>> threshold = compute_fwe_threshold(null_dist, alpha=0.05)
        >>> # Voxels with |t| > threshold are significant at FWE p < 0.05
    """
    percentile = 100 * (1 - alpha)
    threshold = np.percentile(null_distribution, percentile)
    
    logger.debug(f"FWE threshold at alpha={alpha}: {threshold:.3f}")
    
    return threshold
