"""ROI-to-voxel connectivity analysis using GLM."""

from pathlib import Path
from typing import List, Optional
import logging
import numpy as np
import nibabel as nib
from nilearn import glm

from connectomix.connectivity.extraction import extract_single_region_timeseries
from connectomix.io.writers import save_nifti_with_sidecar
from connectomix.utils.exceptions import ConnectivityError


def compute_roi_to_voxel(
    func_img: nib.Nifti1Image,
    roi_mask: nib.Nifti1Image,
    roi_name: str,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
    t_r: Optional[float] = None
) -> Path:
    """Compute ROI-to-voxel connectivity using GLM.
    
    Creates a 3D brain map showing correlation strength between the ROI
    and every voxel in the brain.
    
    Args:
        func_img: Functional image (4D)
        roi_mask: Binary mask defining the ROI
        roi_name: Name of ROI region (for metadata)
        output_path: Path for output effect size map
        logger: Optional logger instance
        t_r: Repetition time in seconds
    
    Returns:
        Path to saved effect size map
    
    Raises:
        ConnectivityError: If analysis fails
    """
    if logger:
        logger.info(f"Computing ROI-to-voxel connectivity: {roi_name}")
    
    try:
        # Extract ROI time series (average across voxels in mask)
        roi_timeseries = extract_single_region_timeseries(
            func_img,
            roi_mask,
            logger
        )
        
        if logger:
            logger.debug(f"  ROI time series shape: {roi_timeseries.shape}")
        
        # Create design matrix with ROI time series as regressor
        n_scans = len(roi_timeseries)
        design_matrix = np.column_stack([
            roi_timeseries,
            np.ones(n_scans)  # Intercept
        ])
        
        # Fit GLM
        if logger:
            logger.debug("  Fitting GLM...")
        
        glm_model = glm.FirstLevelModel(
            t_r=t_r,
            high_pass=None,  # Already filtered
            smoothing_fwhm=None,  # No additional smoothing
            standardize=False,  # Already standardized
            minimize_memory=False
        )
        
        glm_model.fit(func_img, design_matrices=design_matrix)
        
        # Compute contrast for ROI regressor (first column)
        if logger:
            logger.debug("  Computing effect size contrast...")
        
        contrast = np.zeros(design_matrix.shape[1])
        contrast[0] = 1  # Effect of ROI regressor
        
        effect_size_map = glm_model.compute_contrast(
            contrast,
            output_type='effect_size'
        )
        
        # Save effect size map
        metadata = {
            'ROIName': roi_name,
            'AnalysisMethod': 'roiToVoxel',
            'ContrastType': 'effect_size',
            'Description': f'ROI-to-voxel connectivity map for {roi_name}'
        }
        
        save_nifti_with_sidecar(effect_size_map, output_path, metadata)
        
        if logger:
            logger.info(f"  Saved effect size map: {output_path.name}")
        
        return output_path
    
    except Exception as e:
        raise ConnectivityError(f"ROI-to-voxel analysis failed for {roi_name}: {e}")


def compute_multiple_rois_to_voxel(
    func_img: nib.Nifti1Image,
    roi_masks: List[nib.Nifti1Image],
    roi_names: List[str],
    output_dir: Path,
    output_pattern: str,
    logger: Optional[logging.Logger] = None,
    t_r: Optional[float] = None
) -> List[Path]:
    """Compute ROI-to-voxel connectivity for multiple ROIs.
    
    Args:
        func_img: Functional image (4D)
        roi_masks: List of binary mask images
        roi_names: List of ROI region names
        output_dir: Directory for output maps
        output_pattern: Filename pattern with {roi_name} placeholder
        logger: Optional logger instance
        t_r: Repetition time in seconds
    
    Returns:
        List of paths to saved effect size maps
    
    Raises:
        ConnectivityError: If analysis fails
    """
    if len(roi_names) != len(roi_masks):
        raise ConnectivityError(
            f"Number of ROI names ({len(roi_names)}) doesn't match "
            f"number of masks ({len(roi_masks)})"
        )
    
    output_paths = []
    
    for roi_name, roi_mask in zip(roi_names, roi_masks):
        # Build output path
        output_filename = output_pattern.format(roi_name=roi_name)
        output_path = output_dir / output_filename
        
        # Compute connectivity
        result_path = compute_roi_to_voxel(
            func_img=func_img,
            roi_mask=roi_mask,
            roi_name=roi_name,
            output_path=output_path,
            logger=logger,
            t_r=t_r
        )
        
        output_paths.append(result_path)
    
    return output_paths
