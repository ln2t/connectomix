"""CanICA data-driven atlas generation."""

from pathlib import Path
from typing import List, Optional
import logging
import numpy as np
import nibabel as nib
from nilearn.decomposition import CanICA
from nilearn.regions import RegionExtractor

from connectomix.utils.exceptions import PreprocessingError


def run_canica_atlas(
    func_files: List[Path],
    n_components: int,
    output_components_path: Path,
    output_regions_path: Path,
    logger: logging.Logger,
    threshold: float = 1.0,
    min_region_size: int = 50,
    mask_img: Optional[nib.Nifti1Image] = None
) -> Path:
    """Generate data-driven atlas using CanICA.
    
    Args:
        func_files: List of paths to denoised functional images
        n_components: Number of ICA components to extract
        output_components_path: Path for ICA components output
        output_regions_path: Path for extracted regions output
        logger: Logger instance
        threshold: Threshold for extracting regions from components
        min_region_size: Minimum region size in voxels
        mask_img: Optional brain mask
    
    Returns:
        Path to extracted regions atlas file
    
    Raises:
        PreprocessingError: If CanICA fails
    """
    # Skip if output already exists
    if output_regions_path.exists():
        logger.info(f"CanICA atlas exists, skipping: {output_regions_path.name}")
        return output_regions_path
    
    logger.info(f"Running CanICA on {len(func_files)} image(s)")
    logger.info(f"  n_components = {n_components}")
    
    try:
        # Convert paths to strings for nilearn
        func_files_str = [str(f) for f in func_files]
        
        # Run CanICA
        logger.info("  Fitting CanICA model...")
        canica = CanICA(
            n_components=n_components,
            mask=mask_img,
            smoothing_fwhm=6.0,
            standardize=True,
            random_state=0,  # For reproducibility
            n_jobs=1,
            verbose=0
        )
        
        canica.fit(func_files_str)
        
        # Get components
        components_img = canica.components_img_
        
        # Save components
        output_components_path.parent.mkdir(parents=True, exist_ok=True)
        components_img.to_filename(str(output_components_path))
        logger.info(f"  Saved ICA components to: {output_components_path.name}")
        
        # Extract regions from components
        logger.info(f"  Extracting regions (threshold={threshold}, min_size={min_region_size})...")
        extractor = RegionExtractor(
            components_img,
            threshold=threshold,
            min_region_size=min_region_size,
            standardize=True
        )
        
        extractor.fit()
        regions_img = extractor.regions_img_
        
        # Save extracted regions
        output_regions_path.parent.mkdir(parents=True, exist_ok=True)
        regions_img.to_filename(str(output_regions_path))
        
        n_regions = len(extractor.index_)
        logger.info(f"  Extracted {n_regions} region(s)")
        logger.info(f"  Saved atlas to: {output_regions_path.name}")
        
        return output_regions_path
    
    except Exception as e:
        raise PreprocessingError(f"CanICA failed: {e}")
