"""ROI-to-ROI connectivity analysis."""

from pathlib import Path
from typing import List, Optional
import logging
import numpy as np
import nibabel as nib

from connectomix.connectivity.extraction import extract_roi_timeseries
from connectomix.io.writers import save_matrix_with_sidecar
from connectomix.utils.matrix import compute_connectivity_matrix
from connectomix.utils.exceptions import ConnectivityError


def compute_roi_to_roi(
    func_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    atlas_name: str,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
    kind: str = "correlation",
    roi_names: Optional[List[str]] = None
) -> Path:
    """Compute ROI-to-ROI connectivity matrix.
    
    Creates an N×N correlation matrix showing connectivity between all
    atlas regions.
    
    Args:
        func_img: Functional image (4D)
        atlas_img: Atlas image with labeled regions
        atlas_name: Name of atlas (for metadata)
        output_path: Path for output correlation matrix (.npy)
        logger: Optional logger instance
        kind: Type of connectivity ('correlation' or 'covariance')
        roi_names: Optional list of ROI names (if not provided, use indices)
    
    Returns:
        Path to saved correlation matrix
    
    Raises:
        ConnectivityError: If analysis fails
    """
    if logger:
        logger.info(f"Computing ROI-to-ROI connectivity (atlas: {atlas_name})")
    
    try:
        # Extract ROI time series
        time_series = extract_roi_timeseries(
            func_img,
            atlas_img,
            logger
        )
        
        n_regions = time_series.shape[1]
        
        if logger:
            logger.debug(f"  Time series shape: {time_series.shape}")
            logger.debug(f"  Number of regions: {n_regions}")
        
        # Generate ROI names if not provided
        if roi_names is None:
            roi_names = [f"ROI_{i+1}" for i in range(n_regions)]
        elif len(roi_names) != n_regions:
            logger.warning(
                f"Number of ROI names ({len(roi_names)}) doesn't match "
                f"number of regions ({n_regions}). Using indices instead."
            )
            roi_names = [f"ROI_{i+1}" for i in range(n_regions)]
        
        # Compute connectivity matrix
        if logger:
            logger.debug(f"  Computing {kind} matrix...")
        
        connectivity_matrix = compute_connectivity_matrix(time_series, kind=kind)
        
        # Save matrix
        metadata = {
            'AtlasName': atlas_name,
            'ROINames': roi_names,
            'NumberOfRegions': n_regions,
            'AnalysisMethod': 'roiToRoi',
            'ConnectivityKind': kind,
            'MatrixShape': list(connectivity_matrix.shape),
            'Description': f'ROI-to-ROI {kind} matrix using {atlas_name} atlas'
        }
        
        save_matrix_with_sidecar(connectivity_matrix, output_path, metadata)
        
        if logger:
            logger.info(f"  Saved {kind} matrix: {output_path.name}")
            logger.debug(f"  Matrix range: [{connectivity_matrix.min():.3f}, {connectivity_matrix.max():.3f}]")
        
        return output_path
    
    except Exception as e:
        raise ConnectivityError(f"ROI-to-ROI analysis failed: {e}")


def get_atlas_labels(atlas_img: nib.Nifti1Image) -> List[int]:
    """Get list of unique labels in atlas.
    
    Args:
        atlas_img: Atlas image with labeled regions
    
    Returns:
        List of unique integer labels (excluding 0/background)
    """
    atlas_data = atlas_img.get_fdata()
    unique_labels = np.unique(atlas_data)
    
    # Remove background (0)
    unique_labels = unique_labels[unique_labels > 0]
    
    return unique_labels.astype(int).tolist()
