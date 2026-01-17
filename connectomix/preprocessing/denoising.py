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
        logger.info(f"Denoised file exists, skipping: {output_path.name}")
        return output_path
    
    logger.info(f"Denoising {img_path.name}")
    
    try:
        # Load confounds
        confounds = load_confounds(confounds_path, confound_names)
        logger.debug(f"  Loaded {len(confound_names)} confound(s)")
        
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
