"""Group-level connectivity analysis using tangent space.

This module implements tangent space connectivity analysis for group-level
studies. The tangent space approach provides:
- A group mean connectivity (geometric mean of covariances)
- Individual subject deviations from the group mean
- Better statistical properties for group comparisons

Reference:
    Varoquaux et al., "Detection of brain functional-connectivity difference 
    in post-stroke patients using group-level covariance modeling", MICCAI 2010.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from nilearn.connectome import ConnectivityMeasure

from connectomix.utils.exceptions import ConnectomixError

logger = logging.getLogger(__name__)


def discover_participant_timeseries(
    derivatives_dir: Path,
    atlas: str,
    task: Optional[str] = None,
    session: Optional[str] = None,
    subjects: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Discover time series files from participant-level outputs.
    
    Searches the derivatives directory for ROI time series files that match
    the specified criteria.
    
    Args:
        derivatives_dir: Path to connectomix derivatives directory
        atlas: Atlas name to filter by
        task: Task name to filter by (optional)
        session: Session to filter by (optional)
        subjects: List of subjects to include (optional, None = all)
    
    Returns:
        Dictionary mapping subject IDs to time series file paths
    
    Raises:
        ConnectomixError: If no matching files are found
    """
    timeseries_files = {}
    
    # Build glob pattern
    # Files are like: sub-01/[ses-X/]connectivity_data/sub-01_..._timeseries.npy
    pattern = "sub-*/connectivity_data/*_timeseries.npy"
    if session:
        pattern = f"sub-*/ses-{session}/connectivity_data/*_timeseries.npy"
    
    logger.debug(f"Searching for time series files with pattern: {pattern}")
    
    for ts_file in derivatives_dir.glob(pattern):
        filename = ts_file.name
        
        # Check atlas match
        if f"atlas-{atlas}" not in filename:
            continue
        
        # Check task match if specified
        if task and f"task-{task}" not in filename:
            continue
        
        # Extract subject ID from filename
        parts = filename.split("_")
        sub_id = None
        for part in parts:
            if part.startswith("sub-"):
                sub_id = part.replace("sub-", "")
                break
        
        if sub_id is None:
            logger.warning(f"Could not extract subject ID from: {filename}")
            continue
        
        # Filter by subject list if provided
        if subjects and sub_id not in subjects:
            continue
        
        # Check for duplicates (multiple runs/sessions)
        if sub_id in timeseries_files:
            logger.warning(
                f"Multiple time series found for subject {sub_id}. "
                f"Using first found: {timeseries_files[sub_id].name}"
            )
            continue
        
        timeseries_files[sub_id] = ts_file
        logger.debug(f"Found time series for sub-{sub_id}: {ts_file.name}")
    
    if not timeseries_files:
        raise ConnectomixError(
            f"No time series files found in {derivatives_dir} "
            f"matching atlas={atlas}, task={task}, session={session}"
        )
    
    logger.info(f"Discovered {len(timeseries_files)} subject time series files")
    return timeseries_files


def load_timeseries(
    timeseries_files: Dict[str, Path],
) -> Tuple[List[str], List[np.ndarray]]:
    """Load time series arrays from files.
    
    Args:
        timeseries_files: Dictionary mapping subject IDs to file paths
    
    Returns:
        Tuple of (subject_ids, timeseries_list) where timeseries_list contains
        arrays of shape (n_timepoints, n_regions) for each subject
    
    Raises:
        ConnectomixError: If files cannot be loaded or have inconsistent shapes
    """
    subject_ids = []
    timeseries_list = []
    n_regions = None
    
    for sub_id in sorted(timeseries_files.keys()):
        ts_file = timeseries_files[sub_id]
        
        try:
            ts = np.load(ts_file)
        except Exception as e:
            raise ConnectomixError(f"Failed to load {ts_file}: {e}")
        
        # Validate shape
        if ts.ndim != 2:
            raise ConnectomixError(
                f"Time series for sub-{sub_id} has wrong dimensions: "
                f"expected 2D (timepoints x regions), got {ts.ndim}D"
            )
        
        # Check consistent number of regions
        if n_regions is None:
            n_regions = ts.shape[1]
        elif ts.shape[1] != n_regions:
            raise ConnectomixError(
                f"Inconsistent number of regions: sub-{sub_id} has {ts.shape[1]} "
                f"regions, expected {n_regions}"
            )
        
        subject_ids.append(sub_id)
        timeseries_list.append(ts)
        logger.debug(
            f"Loaded sub-{sub_id}: {ts.shape[0]} timepoints, {ts.shape[1]} regions"
        )
    
    logger.info(
        f"Loaded {len(timeseries_list)} subjects, "
        f"{n_regions} regions each"
    )
    
    return subject_ids, timeseries_list


def compute_tangent_connectivity(
    timeseries_list: List[np.ndarray],
    subject_ids: List[str],
    vectorize: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute tangent space connectivity for a group of subjects.
    
    Uses nilearn's ConnectivityMeasure with kind='tangent' to compute:
    - Group mean connectivity (geometric mean of covariance matrices)
    - Individual tangent vectors (deviations from group mean)
    
    The tangent space approach is recommended for group studies as it:
    - Captures both correlations and partial correlations information
    - Provides better statistical properties (Euclidean space)
    - Handles subject variability in a principled way
    
    Args:
        timeseries_list: List of time series arrays, one per subject.
            Each array has shape (n_timepoints, n_regions).
        subject_ids: List of subject IDs corresponding to timeseries_list.
        vectorize: If True, return vectorized (flattened) connectivity.
    
    Returns:
        Dictionary containing:
            - 'group_mean': Group mean connectivity matrix (n_regions x n_regions)
            - 'whitening': Whitening matrix used for tangent projection
            - 'tangent_matrices': Dict mapping subject ID to tangent matrix
            - 'subject_ids': List of subject IDs in order
            - 'n_regions': Number of ROI regions
            - 'n_subjects': Number of subjects
    
    Raises:
        ConnectomixError: If fewer than 2 subjects provided or computation fails
    """
    n_subjects = len(timeseries_list)
    
    if n_subjects < 2:
        raise ConnectomixError(
            f"Tangent space analysis requires at least 2 subjects, "
            f"got {n_subjects}"
        )
    
    logger.info(f"Computing tangent space connectivity for {n_subjects} subjects")
    
    # Create tangent space connectivity measure
    tangent_measure = ConnectivityMeasure(
        kind='tangent',
        vectorize=vectorize,
        standardize='zscore_sample',
    )
    
    try:
        # Fit and transform: computes group mean and individual tangent matrices
        tangent_matrices = tangent_measure.fit_transform(timeseries_list)
    except Exception as e:
        raise ConnectomixError(f"Failed to compute tangent connectivity: {e}")
    
    # Extract results
    group_mean = tangent_measure.mean_
    whitening = tangent_measure.whitening_
    n_regions = group_mean.shape[0]
    
    logger.info(f"Group mean connectivity: {n_regions} x {n_regions}")
    logger.info(f"Tangent matrices shape: {tangent_matrices.shape}")
    
    # Build result dictionary
    tangent_by_subject = {}
    for i, sub_id in enumerate(subject_ids):
        tangent_by_subject[sub_id] = tangent_matrices[i]
    
    results = {
        'group_mean': group_mean,
        'whitening': whitening,
        'tangent_matrices': tangent_by_subject,
        'subject_ids': subject_ids,
        'n_regions': n_regions,
        'n_subjects': n_subjects,
    }
    
    # Log some statistics
    mean_upper = group_mean[np.triu_indices_from(group_mean, k=1)]
    logger.info(
        f"Group mean connectivity range: [{mean_upper.min():.3f}, {mean_upper.max():.3f}]"
    )
    
    return results


def compute_group_correlation_mean(
    timeseries_list: List[np.ndarray],
    subject_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Compute simple mean correlation across subjects.
    
    This is a simpler alternative to tangent space that computes:
    - Individual correlation matrices
    - Arithmetic mean across subjects
    
    Note: The tangent space approach is generally preferred for group studies.
    
    Args:
        timeseries_list: List of time series arrays, one per subject.
        subject_ids: List of subject IDs.
    
    Returns:
        Dictionary containing:
            - 'group_mean': Mean correlation matrix
            - 'correlation_matrices': Dict mapping subject ID to correlation matrix
            - 'subject_ids': List of subject IDs
    """
    logger.info(f"Computing mean correlation for {len(timeseries_list)} subjects")
    
    correlation_measure = ConnectivityMeasure(
        kind='correlation',
        vectorize=False,
        standardize='zscore_sample',
    )
    
    correlation_matrices = correlation_measure.fit_transform(timeseries_list)
    group_mean = correlation_measure.mean_
    
    # Build per-subject dictionary
    corr_by_subject = {}
    for i, sub_id in enumerate(subject_ids):
        corr_by_subject[sub_id] = correlation_matrices[i]
    
    return {
        'group_mean': group_mean,
        'correlation_matrices': corr_by_subject,
        'subject_ids': subject_ids,
        'n_regions': group_mean.shape[0],
        'n_subjects': len(subject_ids),
    }


def project_to_tangent_space(
    connectivity_matrix: np.ndarray,
    group_mean: np.ndarray,
    whitening: np.ndarray,
) -> np.ndarray:
    """Project a single connectivity matrix into tangent space.
    
    This can be used to project new subjects into an existing tangent space
    defined by a group mean and whitening matrix.
    
    Args:
        connectivity_matrix: Individual covariance/connectivity matrix
        group_mean: Group mean covariance matrix
        whitening: Whitening matrix from the group fit
    
    Returns:
        Tangent vector (matrix) representing deviation from group mean
    """
    from scipy import linalg
    
    # Whiten the connectivity matrix
    whitened = whitening @ connectivity_matrix @ whitening.T
    
    # Project to tangent space via matrix logarithm
    tangent = linalg.logm(whitened)
    
    # Ensure symmetric
    tangent = (tangent + tangent.T) / 2
    
    return tangent


def inverse_tangent_transform(
    tangent_matrix: np.ndarray,
    group_mean: np.ndarray,
    whitening: np.ndarray,
) -> np.ndarray:
    """Transform a tangent matrix back to covariance space.
    
    Args:
        tangent_matrix: Tangent space representation
        group_mean: Group mean covariance matrix
        whitening: Whitening matrix from the group fit
    
    Returns:
        Reconstructed covariance matrix
    """
    from scipy import linalg
    
    # Inverse whitening
    inv_whitening = linalg.inv(whitening)
    
    # Exponentiate to get back to SPD manifold
    exp_tangent = linalg.expm(tangent_matrix)
    
    # Un-whiten
    covariance = inv_whitening @ exp_tangent @ inv_whitening.T
    
    return covariance
