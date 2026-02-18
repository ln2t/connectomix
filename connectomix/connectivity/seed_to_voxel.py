"""Seed-to-voxel connectivity analysis using GLM."""

from pathlib import Path
from typing import List, Optional, Dict
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image
import matplotlib.pyplot as plt
from scipy import ndimage

from connectomix.connectivity.extraction import extract_seeds_timeseries
from connectomix.io.writers import save_nifti_with_sidecar
from connectomix.utils.exceptions import ConnectivityError
from connectomix.utils.validation import sanitize_filename



def compute_glm_contrast_map(
    func_img: nib.Nifti1Image,
    timeseries: np.ndarray,
    region_name: str,
    output_path: Path,
    brain_mask: Optional[nib.Nifti1Image] = None,
    regressor_name: str = 'regressor',
    logger: Optional[logging.Logger] = None,
    t_r: Optional[float] = None,
    metadata: Optional[Dict] = None,
) -> nib.Nifti1Image:
    """Compute GLM-based connectivity map from a timeseries.
    
    This is the shared GLM computation used by both seed-to-voxel and ROI-to-voxel.
    Performs only GLM computation and saves NIfTI + metadata. Visualization is handled
    separately by the calling functions.
    
    Args:
        func_img: Functional image (4D)
        timeseries: Extracted region timeseries, shape (n_timepoints,)
        region_name: Name of region (for logging and metadata)
        output_path: Path for output effect size map
        brain_mask: Brain mask image restricting analysis to brain voxels
        regressor_name: Name of the regressor column (for logging)
        logger: Optional logger instance
        t_r: Repetition time in seconds
        metadata: Optional metadata dictionary (will be merged with defaults)
    
    Returns:
        Effect size map as NIfTI image
    
    Raises:
        ConnectivityError: If analysis fails
    """
    try:
        # Create design matrix with region time series as regressor
        n_scans = len(timeseries)
        design_matrix = pd.DataFrame(
            np.column_stack([timeseries, np.ones(n_scans)]),
            columns=[regressor_name, 'intercept']
        )
        
        # Fit GLM
        if logger:
            logger.debug("  Fitting GLM...")
        
        glm_model = FirstLevelModel(
            t_r=t_r,
            mask_img=brain_mask,
            high_pass=None,  # Already filtered
            smoothing_fwhm=None,  # No additional smoothing
            standardize=False,  # Already standardized
            minimize_memory=False
        )
        
        glm_model.fit(func_img, design_matrices=[design_matrix])
        
        # Compute contrast for regressor (first column)
        if logger:
            logger.debug("  Computing effect size contrast...")
        
        contrast = np.array([1, 0])  # Effect of regressor, not intercept
        
        effect_size_map = glm_model.compute_contrast(
            contrast,
            output_type='effect_size'
        )
        
        # Validate effect size map
        effect_data = effect_size_map.get_fdata()
        if logger:
            logger.debug(f"  Effect size map stats: mean={effect_data.mean():.6f}, "
                        f"std={effect_data.std():.6f}, range=[{effect_data.min():.6f}, {effect_data.max():.6f}]")
        
        if np.allclose(effect_data, 0):
            raise ConnectivityError(
                f"Effect size map for {region_name} is all zeros. "
                f"The GLM may have failed to compute coefficients. "
                f"Check if the functional image has sufficient variability."
            )
        
        # Prepare metadata
        default_metadata = {
            'RegionName': region_name,
            'ContrastType': 'effect_size',
            'Description': f'Connectivity map for {region_name}'
        }
        if metadata:
            default_metadata.update(metadata)
        
        # Save effect size map
        save_nifti_with_sidecar(effect_size_map, output_path, default_metadata)
        
        if logger:
            logger.info(f"  Saved effect size map: {output_path.name}")
        
        return effect_size_map
    
    except ConnectivityError:
        raise
    except Exception as e:
        raise ConnectivityError(f"GLM-based connectivity analysis failed for {region_name}: {e}")


def find_masks_directory(denoised_func_path: Path) -> Path:
    """Find the masks directory from a denoised functional image path.
    
    Given a path like: /derivatives/fmridenoiser/sub-01/func/sub-01_bold.nii.gz
    Returns: /derivatives/fmridenoiser/sub-01/masks
    
    Args:
        denoised_func_path: Path to denoised functional image
    
    Returns:
        Path to masks directory
    
    Raises:
        ConnectivityError: If masks directory cannot be found
    """
    denoised_path = Path(denoised_func_path)
    
    # Go up the directory tree to find the subject directory
    # The structure is typically: derivatives/fmridenoiser/sub-XX/anat|func
    # We need to find: derivatives/fmridenoiser/sub-XX/masks
    
    current_path = denoised_path.parent
    
    # Go up to func/anat directory level
    while current_path.name not in ('func', 'anat'):
        current_path = current_path.parent
        if current_path == current_path.parent:  # Reached filesystem root
            raise ConnectivityError(
                f"Cannot determine subject directory from path: {denoised_func_path}"
            )
    
    # Go up one level to get to subject directory (sub-XX)
    subject_dir = current_path.parent
    
    # The masks directory should be at subject_dir/masks
    masks_dir = subject_dir / "masks"
    
    if not masks_dir.exists():
        raise ConnectivityError(
            f"Masks directory not found at: {masks_dir}\n"
            f"Derived from denoised functional path: {denoised_func_path}"
        )
    
    return masks_dir


def load_brain_mask(
    masks_dir: Path,
    file_entities: Dict[str, str],
    logger: Optional[logging.Logger] = None
) -> nib.Nifti1Image:
    """Load brain mask from fmridenoiser masks folder with priority.
    
    Priority:
    1. Task-matching mask (e.g., sub-01_ses-01_task-rest_space-MNI_desc-brain_mask.nii.gz)
    2. Generic mask without task (e.g., sub-01_ses-01_space-MNI_desc-brain_mask.nii.gz)
    3. Task-matching mask without session (e.g., sub-01_task-rest_space-MNI_desc-brain_mask.nii.gz)
    4. Generic mask without session (e.g., sub-01_space-MNI_desc-brain_mask.nii.gz)
    
    Args:
        masks_dir: Path to masks directory from fmridenoiser derivatives
        file_entities: Dictionary with BIDS entities (sub, task, space, session, etc.)
        logger: Optional logger instance
    
    Returns:
        Loaded brain mask as NIfTI image
    
    Raises:
        ConnectivityError: If no appropriate mask found
    """
    masks_dir = Path(masks_dir)
    
    if not masks_dir.exists():
        raise ConnectivityError(f"Masks directory does not exist: {masks_dir}")
    
    sub = file_entities.get('sub')
    task = file_entities.get('task')
    space = file_entities.get('space', 'MNI')
    session = file_entities.get('session') or file_entities.get('ses')
    
    if not sub:
        raise ConnectivityError("Subject ID ('sub') required in file_entities")
    
    # Build base filename components
    base_parts = [f"sub-{sub}"]
    
    # Add session if present
    if session:
        base_parts.append(f"ses-{session}")
    
    # Try task-specific mask first if task is available
    if task:
        task_parts = base_parts + [f"task-{task}", f"space-{space}", "desc-brain_mask.nii.gz"]
        task_mask_pattern = "_".join(task_parts)
        task_mask_path = masks_dir / task_mask_pattern
        
        if task_mask_path.exists():
            if logger:
                logger.debug(f"Loading task-specific brain mask: {task_mask_path.name}")
            return nib.load(task_mask_path)
    
    # Fall back to generic mask
    generic_parts = base_parts + [f"space-{space}", "desc-brain_mask.nii.gz"]
    generic_mask_pattern = "_".join(generic_parts)
    generic_mask_path = masks_dir / generic_mask_pattern
    
    if generic_mask_path.exists():
        if logger:
            logger.debug(f"Loading generic brain mask: {generic_mask_path.name}")
        return nib.load(generic_mask_path)
    
    # Fallback: Try without session if session was specified
    fallback_task_mask_pattern = None
    fallback_generic_mask_pattern = None
    
    if session:
        base_parts_no_ses = [f"sub-{sub}"]
        
        if task:
            fallback_task_parts = base_parts_no_ses + [f"task-{task}", f"space-{space}", "desc-brain_mask.nii.gz"]
            fallback_task_mask_pattern = "_".join(fallback_task_parts)
            fallback_task_mask_path = masks_dir / fallback_task_mask_pattern
            
            if fallback_task_mask_path.exists():
                if logger:
                    logger.debug(f"Loading task-specific brain mask (without session): {fallback_task_mask_path.name}")
                return nib.load(fallback_task_mask_path)
        
        fallback_generic_parts = base_parts_no_ses + [f"space-{space}", "desc-brain_mask.nii.gz"]
        fallback_generic_mask_pattern = "_".join(fallback_generic_parts)
        fallback_generic_mask_path = masks_dir / fallback_generic_mask_pattern
        
        if fallback_generic_mask_path.exists():
            if logger:
                logger.debug(f"Loading generic brain mask (without session): {fallback_generic_mask_path.name}")
            return nib.load(fallback_generic_mask_path)
    
    # No mask found - list available files for debugging
    available_masks = list(masks_dir.glob("*brain_mask.nii.gz"))
    mask_names = [m.name for m in available_masks]
    
    error_msg = (
        f"No brain mask found for sub={sub}, ses={session}, task={task}, space={space}\n"
        f"Looked for:\n"
    )
    search_count = 1
    if task:
        error_msg += f"  {search_count}. {task_mask_pattern}\n"
        search_count += 1
    error_msg += f"  {search_count}. {generic_mask_pattern}\n"
    
    if session and fallback_task_mask_pattern:
        search_count += 1
        error_msg += f"  {search_count}. {fallback_task_mask_pattern} (fallback without session)\n"
    
    if session and fallback_generic_mask_pattern:
        search_count += 1
        error_msg += f"  {search_count}. {fallback_generic_mask_pattern} (fallback without session)\n"
    
    if mask_names:
        error_msg += f"Available masks in {masks_dir}:\n  " + "\n  ".join(mask_names)
    else:
        error_msg += f"No masks found in {masks_dir}"
    
    raise ConnectivityError(error_msg)


def compute_seed_to_voxel(
    func_img: nib.Nifti1Image,
    seed_coords: np.ndarray,
    seed_name: str,
    output_path: Path,
    denoised_func_path: Optional[Path] = None,
    file_entities: Optional[Dict[str, str]] = None,
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
        denoised_func_path: Path to denoised functional image (to find masks directory)
        file_entities: Dictionary with BIDS entities (sub, task, space, etc.)
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
        # Load brain mask if denoised path and entities are provided
        brain_mask_img = None
        if denoised_func_path and file_entities:
            try:
                masks_dir = find_masks_directory(denoised_func_path)
                brain_mask_img = load_brain_mask(masks_dir, file_entities, logger)
            except ConnectivityError as e:
                if logger:
                    logger.warning(f"Could not load brain mask: {e}")
                # Continue without brain mask - GLM will analyze all voxels
        
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
            logger.debug(f"  Seed time series stats: mean={seed_timeseries.mean():.6f}, "
                        f"std={seed_timeseries.std():.6f}, range=[{seed_timeseries.min():.6f}, {seed_timeseries.max():.6f}]")
        
        # Validate seed time series quality
        if np.allclose(seed_timeseries, 0):
            raise ConnectivityError(
                f"Seed time series for {seed_name} is all zeros. "
                f"Check seed coordinates {seed_coords.flatten()} and radius {radius}mm."
            )
        
        if np.std(seed_timeseries) < 1e-10:
            raise ConnectivityError(
                f"Seed time series for {seed_name} has no variance "
                f"(std={np.std(seed_timeseries):.2e}). "
                f"Check if seed is outside the functional image."
            )
        
        # Use shared GLM computation function
        metadata = {
            'SeedName': seed_name,
            'SeedCoordinates_mm': seed_coords.flatten().tolist(),
            'SeedRadius_mm': radius,
            'SeedCenterCoords_mm': seed_coords.flatten().tolist(),
            'AnalysisMethod': 'seedToVoxel',
        }
        
        effect_size_map = compute_glm_contrast_map(
            func_img=func_img,
            timeseries=seed_timeseries,
            region_name=seed_name,
            output_path=output_path,
            brain_mask=brain_mask_img,
            regressor_name='seed',
            logger=logger,
            t_r=t_r,
            metadata=metadata,
        )
        
        # Create visualization with seed sphere overlay
        try:
            from nilearn import plotting as nplot
            from connectomix.utils.visualization import _create_seed_sphere
            import json
            
            # Use seed coordinates for cut_coords
            cut_coords = tuple(seed_coords.flatten())
            
            # Create orthogonal plot
            fig = plt.figure(figsize=(16, 5))
            display = nplot.plot_stat_map(
                effect_size_map,
                threshold=0,
                display_mode='ortho',
                cut_coords=cut_coords,
                colorbar=True,
                cmap='cold_hot',
                title=f"Connectivity Map - {seed_name}",
                figure=fig,
            )
            
            # Overlay seed sphere
            try:
                seed_sphere_img = _create_seed_sphere(effect_size_map, seed_coords.flatten(), radius)
                
                # Validate sphere has non-zero values
                sphere_data = seed_sphere_img.get_fdata()
                n_nonzero = np.sum(sphere_data > 0)
                if logger:
                    logger.debug(f"  Seed sphere: {n_nonzero} voxels, shape={sphere_data.shape}")
                
                if n_nonzero > 0:
                    display.add_contours(
                        seed_sphere_img,
                        levels=[0.5],
                        colors='lime',
                        linewidths=2.0,
                    )
                    if logger:
                        logger.debug(f"  Added seed sphere contours: coords={seed_coords.flatten()}, radius={radius}mm")
                else:
                    if logger:
                        logger.warning(f"  Seed sphere is empty (no voxels), coords={seed_coords.flatten()}, radius={radius}mm")
            except Exception as sphere_error:
                if logger:
                    logger.warning(f"  Could not overlay seed sphere: {sphere_error}")
                    logger.debug(f"  Sphere error details:", exc_info=True)
            
            # Save plot to figures directory
            # Remove .nii/.nii.gz extension and add .png
            png_name = output_path.name.replace('.nii.gz', '').replace('.nii', '') + '.png'
            plot_output = output_path.parent.parent / 'figures' / png_name
            plot_output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_output, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            if logger:
                logger.info(f"  Saved plot: {plot_output.name}")
        
        except Exception as plot_error:
            if logger:
                logger.warning(f"Could not create visualization: {plot_error}")
        
        return output_path
    
    except ConnectivityError:
        raise
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
        # Build output path with sanitized seed_name to handle spaces and special characters
        safe_seed_name = sanitize_filename(seed_name)
        output_filename = output_pattern.format(seed_name=safe_seed_name)
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
