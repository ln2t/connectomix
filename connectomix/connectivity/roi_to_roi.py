"""ROI-to-ROI connectivity analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import nibabel as nib

from connectomix.connectivity.extraction import extract_roi_timeseries
from connectomix.io.writers import save_matrix_with_sidecar
from connectomix.utils.matrix import compute_connectivity_matrix, compute_all_connectivity_matrices, CONNECTIVITY_KINDS
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
        kind: Type of connectivity ('correlation', 'covariance', 'partial correlation', 'precision')
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


def compute_roi_to_roi_all_measures(
    func_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    atlas_name: str,
    output_dir: Path,
    base_filename: str,
    logger: Optional[logging.Logger] = None,
    roi_names: Optional[List[str]] = None,
    roi_coords: Optional[np.ndarray] = None,
    coordinate_space: str = "MNI152NLin2009cAsym",
    save_timeseries: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Path], Optional[Path], List[str]]:
    """Compute all ROI-to-ROI connectivity matrices and save timeseries.
    
    Extracts ROI time series once and computes all connectivity measures:
    correlation, covariance, partial correlation, and precision.
    
    Args:
        func_img: Functional image (4D)
        atlas_img: Atlas image with labeled regions
        atlas_name: Name of atlas (for metadata)
        output_dir: Directory for output files
        base_filename: Base filename for outputs (without extension)
        logger: Optional logger instance
        roi_names: Optional list of ROI names
        roi_coords: Optional array of ROI centroid coordinates (N, 3) in MNI space
        coordinate_space: Name of the MNI coordinate space (for metadata)
        save_timeseries: Whether to save the raw time series
    
    Returns:
        Tuple of:
            - time_series: Extracted time series array (n_timepoints, n_regions)
            - matrices: Dict mapping connectivity kind to matrix
            - output_paths: Dict mapping connectivity kind to output path
            - timeseries_path: Path to saved timeseries (or None if not saved)
            - roi_names: List of ROI names used
    
    Raises:
        ConnectivityError: If analysis fails
    """
    if logger:
        logger.info(f"Computing all ROI-to-ROI connectivity measures (atlas: {atlas_name})")
    
    try:
        # Extract ROI time series
        time_series = extract_roi_timeseries(
            func_img,
            atlas_img,
            logger
        )
        
        n_timepoints, n_regions = time_series.shape
        
        if logger:
            logger.info(f"  Extracted time series: {n_timepoints} timepoints × {n_regions} regions")
        
        # Generate ROI names if not provided
        if roi_names is None:
            roi_names = [f"ROI_{i+1}" for i in range(n_regions)]
        elif len(roi_names) != n_regions:
            if logger:
                logger.warning(
                    f"Number of ROI names ({len(roi_names)}) doesn't match "
                    f"number of regions ({n_regions}). Using indices instead."
                )
            roi_names = [f"ROI_{i+1}" for i in range(n_regions)]
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save timeseries if requested
        timeseries_path = None
        if save_timeseries:
            timeseries_path = output_dir / f"{base_filename}_timeseries.npy"
            np.save(timeseries_path, time_series)
            
            # Save JSON sidecar for timeseries
            import json
            ts_sidecar = {
                'Description': f'ROI time series extracted from {atlas_name} atlas',
                'AtlasName': atlas_name,
                'ROINames': roi_names,
                'NumberOfRegions': n_regions,
                'NumberOfTimepoints': n_timepoints,
                'Shape': [n_timepoints, n_regions],
            }
            # Add coordinates if available
            if roi_coords is not None:
                ts_sidecar['ROICoordinates'] = roi_coords.tolist()
                ts_sidecar['CoordinateSpace'] = coordinate_space
            
            with open(timeseries_path.with_suffix('.json'), 'w') as f:
                json.dump(ts_sidecar, f, indent=2)
            
            if logger:
                logger.info(f"  Saved time series: {timeseries_path.name}")
        
        # Compute all connectivity matrices
        if logger:
            logger.info(f"  Computing connectivity matrices...")
        
        matrices = compute_all_connectivity_matrices(time_series)
        
        # Save each matrix
        output_paths = {}
        for kind, matrix in matrices.items():
            # Create safe filename (replace spaces)
            kind_safe = kind.replace(' ', '-')
            output_path = output_dir / f"{base_filename}_desc-{kind_safe}_connectivity.npy"
            
            metadata = {
                'AtlasName': atlas_name,
                'ROINames': roi_names,
                'NumberOfRegions': n_regions,
                'AnalysisMethod': 'roiToRoi',
                'ConnectivityKind': kind,
                'MatrixShape': list(matrix.shape),
                'Description': f'ROI-to-ROI {kind} matrix using {atlas_name} atlas'
            }
            
            # Add coordinates if available
            if roi_coords is not None:
                metadata['ROICoordinates'] = roi_coords.tolist()
                metadata['CoordinateSpace'] = coordinate_space
            
            save_matrix_with_sidecar(matrix, output_path, metadata)
            output_paths[kind] = output_path
            
            if logger:
                logger.info(f"    Saved {kind}: {output_path.name}")
                logger.debug(f"      Range: [{matrix.min():.3f}, {matrix.max():.3f}]")
        
        return time_series, matrices, output_paths, timeseries_path, roi_names
    
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
