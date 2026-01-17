"""Seed-to-voxel connectivity analysis using GLM."""

from pathlib import Path
from typing import List, Optional
import logging
import numpy as np
import nibabel as nib
from nilearn import glm

from connectomix.connectivity.extraction import extract_seeds_timeseries
from connectomix.io.writers import save_nifti_with_sidecar
from connectomix.utils.exceptions import ConnectivityError


def compute_seed_to_voxel(
    func_img: nib.Nifti1Image,
    seed_coords: np.ndarray,
    seed_name: str,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
    radius: float = 5.0,
    t_r: Optional[float] = None
) -> Path:
    """Compute seed-to-voxel connectivity using GLM.
    
    Creates a 3D brain map showing correlation strength between the seed
    region and every voxel in the brain.
    
    Args:
        func_img: Functional image (4D)
        seed_coords: Seed coordinates, shape (3,) - single seed
        seed_name: Name of seed region (for metadata)
        output_path: Path for output effect size map
        logger: Optional logger instance
        radius: Sphere radius in mm
        t_r: Repetition time in seconds
    
    Returns:
        Path to saved effect size map
    
    Raises:
        ConnectivityError: If analysis fails
    """
    if logger:
        logger.info(f"Computing seed-to-voxel connectivity: {seed_name}")
    
    try:
        # Ensure seed_coords is 2D array for masker
        if seed_coords.ndim == 1:
            seed_coords = seed_coords.reshape(1, -1)
        
        # Extract seed time series
        seed_timeseries = extract_seeds_timeseries(
            func_img,
            seed_coords,
            radius,
            logger
        )
        
        # Should be shape (n_timepoints, 1), flatten to (n_timepoints,)
        seed_timeseries = seed_timeseries.flatten()
        
        if logger:
            logger.debug(f"  Seed time series shape: {seed_timeseries.shape}")
        
        # Create design matrix with seed time series as regressor
        n_scans = len(seed_timeseries)
        design_matrix = np.column_stack([
            seed_timeseries,
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
        
        # Compute contrast for seed regressor (first column)
        if logger:
            logger.debug("  Computing effect size contrast...")
        
        contrast = np.zeros(design_matrix.shape[1])
        contrast[0] = 1  # Effect of seed regressor
        
        effect_size_map = glm_model.compute_contrast(
            contrast,
            output_type='effect_size'
        )
        
        # Save effect size map
        metadata = {
            'SeedName': seed_name,
            'SeedCoordinates_mm': seed_coords.flatten().tolist(),
            'SeedRadius_mm': radius,
            'AnalysisMethod': 'seedToVoxel',
            'ContrastType': 'effect_size',
            'Description': f'Seed-to-voxel connectivity map for {seed_name}'
        }
        
        save_nifti_with_sidecar(effect_size_map, output_path, metadata)
        
        if logger:
            logger.info(f"  Saved effect size map: {output_path.name}")
        
        return output_path
    
    except Exception as e:
        raise ConnectivityError(f"Seed-to-voxel analysis failed for {seed_name}: {e}")


def compute_multiple_seeds_to_voxel(
    func_img: nib.Nifti1Image,
    seed_coords_array: np.ndarray,
    seed_names: List[str],
    output_dir: Path,
    output_pattern: str,
    logger: Optional[logging.Logger] = None,
    radius: float = 5.0,
    t_r: Optional[float] = None
) -> List[Path]:
    """Compute seed-to-voxel connectivity for multiple seeds.
    
    Args:
        func_img: Functional image (4D)
        seed_coords_array: Array of seed coordinates, shape (n_seeds, 3)
        seed_names: List of seed region names
        output_dir: Directory for output maps
        output_pattern: Filename pattern with {seed_name} placeholder
        logger: Optional logger instance
        radius: Sphere radius in mm
        t_r: Repetition time in seconds
    
    Returns:
        List of paths to saved effect size maps
    
    Raises:
        ConnectivityError: If analysis fails
    """
    if len(seed_names) != len(seed_coords_array):
        raise ConnectivityError(
            f"Number of seed names ({len(seed_names)}) doesn't match "
            f"number of coordinates ({len(seed_coords_array)})"
        )
    
    output_paths = []
    
    for seed_name, seed_coords in zip(seed_names, seed_coords_array):
        # Build output path
        output_filename = output_pattern.format(seed_name=seed_name)
        output_path = output_dir / output_filename
        
        # Compute connectivity
        result_path = compute_seed_to_voxel(
            func_img=func_img,
            seed_coords=seed_coords,
            seed_name=seed_name,
            output_path=output_path,
            logger=logger,
            radius=radius,
            t_r=t_r
        )
        
        output_paths.append(result_path)
    
    return output_paths
