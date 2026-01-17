"""Time series extraction from functional images."""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import nibabel as nib
from nilearn import maskers
import logging

from connectomix.io.readers import load_seeds_file
from connectomix.utils.exceptions import ConnectivityError


def extract_seeds_timeseries(
    func_img: nib.Nifti1Image,
    seeds_coords: np.ndarray,
    radius: float,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """Extract time series from spherical seed regions.
    
    Args:
        func_img: Functional image (4D)
        seeds_coords: Array of seed coordinates, shape (n_seeds, 3)
        radius: Sphere radius in mm
        logger: Optional logger instance
    
    Returns:
        Time series array of shape (n_timepoints, n_seeds)
    
    Raises:
        ConnectivityError: If extraction fails
    """
    if logger:
        logger.debug(f"Extracting time series from {len(seeds_coords)} seed(s), radius={radius}mm")
    
    try:
        # Create spheres masker
        masker = maskers.NiftiSpheresMasker(
            seeds=seeds_coords,
            radius=radius,
            standardize=True,  # Standardize signal
            detrend=False,     # Already detrended in preprocessing
            low_pass=None,     # Already filtered in preprocessing
            high_pass=None,
            t_r=None,          # Will be read from image
            verbose=0
        )
        
        # Extract time series
        time_series = masker.fit_transform(func_img)
        
        if logger:
            logger.debug(f"  Extracted shape: {time_series.shape}")
        
        return time_series
    
    except Exception as e:
        raise ConnectivityError(f"Failed to extract seed time series: {e}")


def extract_roi_timeseries(
    func_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """Extract time series from atlas-defined ROIs.
    
    Args:
        func_img: Functional image (4D)
        atlas_img: Atlas image with labeled regions
        logger: Optional logger instance
    
    Returns:
        Time series array of shape (n_timepoints, n_regions)
    
    Raises:
        ConnectivityError: If extraction fails
    """
    if logger:
        atlas_data = atlas_img.get_fdata()
        n_regions = len(np.unique(atlas_data)) - 1  # Exclude background (0)
        logger.debug(f"Extracting time series from {n_regions} ROI(s)")
    
    try:
        # Create labels masker
        masker = maskers.NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize=True,  # Standardize signal
            detrend=False,     # Already detrended in preprocessing
            low_pass=None,     # Already filtered in preprocessing
            high_pass=None,
            t_r=None,          # Will be read from image
            verbose=0
        )
        
        # Extract time series
        time_series = masker.fit_transform(func_img)
        
        if logger:
            logger.debug(f"  Extracted shape: {time_series.shape}")
        
        return time_series
    
    except Exception as e:
        raise ConnectivityError(f"Failed to extract ROI time series: {e}")


def extract_single_region_timeseries(
    func_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """Extract time series from a single mask region.
    
    Args:
        func_img: Functional image (4D)
        mask_img: Binary mask image
        logger: Optional logger instance
    
    Returns:
        Time series array of shape (n_timepoints,) - averaged across voxels
    
    Raises:
        ConnectivityError: If extraction fails
    """
    if logger:
        mask_data = mask_img.get_fdata()
        n_voxels = np.sum(mask_data > 0)
        logger.debug(f"Extracting time series from mask ({n_voxels} voxel(s))")
    
    try:
        # Create masker
        masker = maskers.NiftiMasker(
            mask_img=mask_img,
            standardize=True,  # Standardize signal
            detrend=False,     # Already detrended in preprocessing
            low_pass=None,     # Already filtered in preprocessing
            high_pass=None,
            t_r=None,          # Will be read from image
            verbose=0
        )
        
        # Extract time series (average across voxels)
        time_series = masker.fit_transform(func_img)
        
        # Should be shape (n_timepoints, 1), flatten to (n_timepoints,)
        time_series = time_series.flatten()
        
        if logger:
            logger.debug(f"  Extracted shape: {time_series.shape}")
        
        return time_series
    
    except Exception as e:
        raise ConnectivityError(f"Failed to extract mask time series: {e}")


def load_and_extract_seeds(
    func_img_path: Path,
    seeds_file_path: Path,
    radius: float,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[str], np.ndarray]:
    """Load seeds file and extract time series.
    
    Convenience function combining seed loading and extraction.
    
    Args:
        func_img_path: Path to functional image
        seeds_file_path: Path to seeds TSV file
        radius: Sphere radius in mm
        logger: Optional logger instance
    
    Returns:
        Tuple of (seed_names, time_series)
        - seed_names: List of seed region names
        - time_series: Array of shape (n_timepoints, n_seeds)
    """
    # Load seeds
    seed_names, seed_coords = load_seeds_file(seeds_file_path)
    
    # Load functional image
    func_img = nib.load(func_img_path)
    
    # Extract time series
    time_series = extract_seeds_timeseries(func_img, seed_coords, radius, logger)
    
    return seed_names, time_series


def load_and_extract_rois(
    func_img_path: Path,
    atlas_img_path: Path,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """Load atlas and extract ROI time series.
    
    Convenience function combining atlas loading and extraction.
    
    Args:
        func_img_path: Path to functional image
        atlas_img_path: Path to atlas image
        logger: Optional logger instance
    
    Returns:
        Time series array of shape (n_timepoints, n_regions)
    """
    # Load images
    func_img = nib.load(func_img_path)
    atlas_img = nib.load(atlas_img_path)
    
    # Extract time series
    time_series = extract_roi_timeseries(func_img, atlas_img, logger)
    
    return time_series
