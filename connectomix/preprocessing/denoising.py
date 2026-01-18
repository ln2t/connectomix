"""Functional image denoising."""

from pathlib import Path
from typing import List, Optional
import logging
import json
import nibabel as nib
from nilearn import image
import numpy as np

from connectomix.io.readers import load_confounds, get_repetition_time
from connectomix.utils.exceptions import PreprocessingError


def denoise_image(
    img_path: Path,
    confounds_path: Path,
    confound_names: List[str],
    high_pass: float,
    low_pass: float,
    output_path: Path,
    logger: logging.Logger,
    t_r: Optional[float] = None,
    overwrite: bool = True
) -> Path:
    """Denoise functional image using confound regression and filtering.
    
    Args:
        img_path: Path to functional image
        confounds_path: Path to confounds TSV file
        confound_names: List of confound column names to regress out
        high_pass: High-pass filter cutoff in Hz
        low_pass: Low-pass filter cutoff in Hz
        output_path: Path for denoised output
        logger: Logger instance
        t_r: Repetition time in seconds (will be read from JSON if None)
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to denoised image
    
    Raises:
        PreprocessingError: If denoising fails
    """
    # Skip if exists and not overwriting
    if output_path.exists() and not overwrite:
        # Validate that existing file matches current parameters
        _validate_existing_denoised_file(
            output_path=output_path,
            confound_names=confound_names,
            high_pass=high_pass,
            low_pass=low_pass,
            logger=logger,
        )
        logger.info(f"Denoised file exists with matching parameters, skipping: {output_path.name}")
        return output_path
    
    logger.info(f"Denoising {img_path.name}")
    
    try:
        # Load confounds (supports wildcards like 'c_comp_cor_*')
        confounds, expanded_names = load_confounds(confounds_path, confound_names)
        logger.debug(f"  Loaded {len(expanded_names)} confound(s): {expanded_names[:5]}{'...' if len(expanded_names) > 5 else ''}")
        
        # Get TR if not provided
        if t_r is None:
            json_path = img_path.with_suffix('').with_suffix('.json')
            if json_path.exists():
                t_r = get_repetition_time(json_path)
                logger.debug(f"  TR = {t_r}s")
            else:
                # Try to get TR from NIfTI header
                img = nib.load(img_path)
                if len(img.header.get_zooms()) > 3:
                    t_r = float(img.header.get_zooms()[3])
                    logger.debug(f"  TR from header = {t_r}s")
                else:
                    logger.warning(
                        f"  No JSON sidecar and TR not in NIfTI header - "
                        f"filtering will not work correctly"
                    )
        
        # Clean image - load it first to ensure header is available
        img = nib.load(img_path)
        cleaned = image.clean_img(
            img,
            confounds=confounds,
            high_pass=high_pass,
            low_pass=low_pass,
            standardize=True,  # Standardize (z-score) signal
            detrend=True,      # Remove linear trend
            t_r=t_r
        )
        
        logger.debug(
            f"  Applied: high_pass={high_pass}Hz, low_pass={low_pass}Hz, "
            f"standardize=True, detrend=True"
        )
        
        # Save denoised image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_filename(str(output_path))
        
        # Create JSON sidecar with denoising info
        sidecar_data = {
            'Confounds': confound_names,
            'HighPass_Hz': high_pass,
            'LowPass_Hz': low_pass,
            'Standardized': True,
            'Detrended': True,
            'RepetitionTime': t_r
        }
        
        sidecar_path = output_path.with_suffix('').with_suffix('.json')
        with sidecar_path.open('w') as f:
            json.dump(sidecar_data, f, indent=2)
        
        logger.debug(f"  Saved to: {output_path}")
        
        return output_path
    
    except Exception as e:
        raise PreprocessingError(f"Failed to denoise {img_path.name}: {e}")


def compute_denoising_quality_metrics(
    original_img: nib.Nifti1Image,
    denoised_img: nib.Nifti1Image,
    mask_img: Optional[nib.Nifti1Image] = None
) -> dict:
    """Compute quality metrics for denoising.
    
    Args:
        original_img: Original functional image
        denoised_img: Denoised functional image
        mask_img: Optional brain mask
    
    Returns:
        Dictionary with quality metrics
    """
    # Get data
    original_data = original_img.get_fdata()
    denoised_data = denoised_img.get_fdata()
    
    # Apply mask if provided
    if mask_img is not None:
        mask_data = mask_img.get_fdata().astype(bool)
        original_masked = original_data[mask_data]
        denoised_masked = denoised_data[mask_data]
    else:
        original_masked = original_data.reshape(-1, original_data.shape[-1])
        denoised_masked = denoised_data.reshape(-1, denoised_data.shape[-1])
    
    # Compute metrics
    metrics = {}
    
    # Temporal standard deviation (measure of noise)
    metrics['temporal_std_original'] = float(np.nanmean(np.nanstd(original_masked, axis=-1)))
    metrics['temporal_std_denoised'] = float(np.nanmean(np.nanstd(denoised_masked, axis=-1)))
    metrics['std_reduction_percent'] = float(
        100 * (1 - metrics['temporal_std_denoised'] / metrics['temporal_std_original'])
    )
    
    # Temporal SNR (mean / std over time)
    tsnr_original = np.nanmean(original_masked, axis=-1) / (np.nanstd(original_masked, axis=-1) + 1e-10)
    tsnr_denoised = np.nanmean(denoised_masked, axis=-1) / (np.nanstd(denoised_masked, axis=-1) + 1e-10)
    
    metrics['tsnr_original'] = float(np.nanmean(tsnr_original))
    metrics['tsnr_denoised'] = float(np.nanmean(tsnr_denoised))
    metrics['tsnr_improvement_percent'] = float(
        100 * (metrics['tsnr_denoised'] - metrics['tsnr_original']) / (metrics['tsnr_original'] + 1e-10)
    )
    
    return metrics


def compute_denoising_histogram_data(
    original_img: nib.Nifti1Image,
    denoised_img: nib.Nifti1Image,
    mask_img: Optional[nib.Nifti1Image] = None,
    n_bins: int = 100,
    subsample: int = 10
) -> dict:
    """Compute histogram data for before/after denoising comparison.
    
    Args:
        original_img: Original functional image (before denoising)
        denoised_img: Denoised functional image
        mask_img: Optional brain mask to restrict analysis to brain voxels
        n_bins: Number of histogram bins
        subsample: Subsample factor to reduce memory usage (e.g., 10 = use every 10th value)
    
    Returns:
        Dictionary containing:
            - 'original_data': Flattened original voxel values (subsampled)
            - 'denoised_data': Flattened denoised voxel values (subsampled)
            - 'original_stats': Dict with mean, std, min, max of original
            - 'denoised_stats': Dict with mean, std, min, max of denoised
    """
    # Get data
    original_data = original_img.get_fdata()
    denoised_data = denoised_img.get_fdata()
    
    # Apply mask if provided, otherwise create a simple non-zero mask from original
    if mask_img is not None:
        mask_data = mask_img.get_fdata().astype(bool)
    else:
        # Use voxels that have non-zero variance over time (brain voxels)
        temporal_std = np.std(original_data, axis=-1)
        mask_data = temporal_std > 0
    
    # Flatten: for each voxel in mask, get all timepoints
    # Shape: (n_voxels_in_mask, n_timepoints) -> flatten to 1D
    original_masked = original_data[mask_data].flatten()
    denoised_masked = denoised_data[mask_data].flatten()
    
    # Remove NaN values
    original_masked = original_masked[~np.isnan(original_masked)]
    denoised_masked = denoised_masked[~np.isnan(denoised_masked)]
    
    # Subsample to reduce memory usage (can be millions of values)
    if subsample > 1 and len(original_masked) > 100000:
        original_masked = original_masked[::subsample]
        denoised_masked = denoised_masked[::subsample]
    
    # Compute statistics
    original_stats = {
        'mean': float(np.mean(original_masked)),
        'std': float(np.std(original_masked)),
        'min': float(np.min(original_masked)),
        'max': float(np.max(original_masked)),
        'n_values': len(original_masked)
    }
    
    denoised_stats = {
        'mean': float(np.mean(denoised_masked)),
        'std': float(np.std(denoised_masked)),
        'min': float(np.min(denoised_masked)),
        'max': float(np.max(denoised_masked)),
        'n_values': len(denoised_masked)
    }
    
    return {
        'original_data': original_masked,
        'denoised_data': denoised_masked,
        'original_stats': original_stats,
        'denoised_stats': denoised_stats
    }


def _validate_existing_denoised_file(
    output_path: Path,
    confound_names: List[str],
    high_pass: float,
    low_pass: float,
    logger: logging.Logger,
) -> None:
    """Validate that existing denoised file matches current parameters.
    
    When a denoised file already exists and overwrite is False, this function
    checks that the existing file was created with the same denoising parameters
    as the current configuration. This prevents accidentally reusing a denoised
    file that was created with different parameters.
    
    Args:
        output_path: Path to existing denoised file
        confound_names: List of confound column names to regress out
        high_pass: High-pass filter cutoff in Hz
        low_pass: Low-pass filter cutoff in Hz
        logger: Logger instance
    
    Raises:
        PreprocessingError: If parameters don't match or JSON sidecar is missing
    """
    sidecar_path = output_path.with_suffix('').with_suffix('.json')
    
    if not sidecar_path.exists():
        raise PreprocessingError(
            f"Cannot validate existing denoised file '{output_path.name}': "
            f"JSON sidecar not found at '{sidecar_path.name}'. "
            f"The existing denoised file may have been created with unknown parameters. "
            f"Either delete the existing file or set --overwrite-denoised to reprocess."
        )
    
    try:
        with sidecar_path.open('r') as f:
            existing_params = json.load(f)
    except json.JSONDecodeError as e:
        raise PreprocessingError(
            f"Cannot validate existing denoised file '{output_path.name}': "
            f"JSON sidecar is corrupted ({e}). "
            f"Delete the existing file or set --overwrite-denoised to reprocess."
        )
    
    # Build expected parameters
    expected_params = {
        'Confounds': sorted(confound_names),
        'HighPass_Hz': high_pass,
        'LowPass_Hz': low_pass,
    }
    
    # Normalize existing confounds for comparison
    existing_confounds = sorted(existing_params.get('Confounds', []))
    existing_high_pass = existing_params.get('HighPass_Hz')
    existing_low_pass = existing_params.get('LowPass_Hz')
    
    # Check for mismatches
    mismatches = []
    
    if existing_confounds != expected_params['Confounds']:
        mismatches.append(
            f"  - Confounds: existing={existing_confounds}, requested={expected_params['Confounds']}"
        )
    
    if existing_high_pass != expected_params['HighPass_Hz']:
        mismatches.append(
            f"  - HighPass_Hz: existing={existing_high_pass}, requested={expected_params['HighPass_Hz']}"
        )
    
    if existing_low_pass != expected_params['LowPass_Hz']:
        mismatches.append(
            f"  - LowPass_Hz: existing={existing_low_pass}, requested={expected_params['LowPass_Hz']}"
        )
    
    if mismatches:
        mismatch_details = "\n".join(mismatches)
        raise PreprocessingError(
            f"Existing denoised file '{output_path.name}' was created with different parameters!\n"
            f"\n"
            f"Parameter mismatches:\n"
            f"{mismatch_details}\n"
            f"\n"
            f"This can lead to incorrect results if the pipeline uses data that was "
            f"denoised differently than intended.\n"
            f"\n"
            f"To resolve this, either:\n"
            f"  1. Delete the existing denoised file and its JSON sidecar to reprocess with new parameters\n"
            f"  2. Set --overwrite-denoised to force reprocessing\n"
            f"  3. Adjust your configuration to match the existing denoised file's parameters"
        )
    
    logger.debug(f"  Validated existing denoised file parameters match current configuration")
