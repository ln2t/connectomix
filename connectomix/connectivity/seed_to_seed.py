"""Seed-to-seed connectivity analysis."""

from pathlib import Path
from typing import List, Optional
import logging
import numpy as np
import nibabel as nib

from connectomix.connectivity.extraction import extract_seeds_timeseries
from connectomix.io.writers import save_matrix_with_sidecar
from connectomix.utils.matrix import compute_connectivity_matrix
from connectomix.utils.exceptions import ConnectivityError


def compute_seed_to_seed(
    func_img: nib.Nifti1Image,
    seed_coords: np.ndarray,
    seed_names: List[str],
    output_path: Path,
    logger: Optional[logging.Logger] = None,
    radius: float = 5.0,
    kind: str = "correlation"
) -> Path:
    """Compute seed-to-seed connectivity matrix.
    
    Creates an N×N correlation matrix showing connectivity between all
    seed regions.
    
    Args:
        func_img: Functional image (4D)
        seed_coords: Array of seed coordinates, shape (n_seeds, 3)
        seed_names: List of seed region names
        output_path: Path for output correlation matrix (.npy)
        logger: Optional logger instance
        radius: Sphere radius in mm
        kind: Type of connectivity ('correlation' or 'covariance')
    
    Returns:
        Path to saved correlation matrix
    
    Raises:
        ConnectivityError: If analysis fails
    """
    if logger:
        logger.info(f"Computing seed-to-seed connectivity ({len(seed_names)} seeds)")
    
    try:
        # Extract seed time series
        time_series = extract_seeds_timeseries(
            func_img,
            seed_coords,
            radius,
            logger
        )
        
        if logger:
            logger.debug(f"  Time series shape: {time_series.shape}")
        
        # Compute connectivity matrix
        if logger:
            logger.debug(f"  Computing {kind} matrix...")
        
        connectivity_matrix = compute_connectivity_matrix(time_series, kind=kind)
        
        # Save matrix
        metadata = {
            'SeedNames': seed_names,
            'SeedCoordinates_mm': seed_coords.tolist(),
            'SeedRadius_mm': radius,
            'AnalysisMethod': 'seedToSeed',
            'ConnectivityKind': kind,
            'MatrixShape': list(connectivity_matrix.shape),
            'Description': f'Seed-to-seed {kind} matrix'
        }
        
        save_matrix_with_sidecar(connectivity_matrix, output_path, metadata)
        
        if logger:
            logger.info(f"  Saved {kind} matrix: {output_path.name}")
            logger.debug(f"  Matrix range: [{connectivity_matrix.min():.3f}, {connectivity_matrix.max():.3f}]")
        
        return output_path
    
    except Exception as e:
        raise ConnectivityError(f"Seed-to-seed analysis failed: {e}")
