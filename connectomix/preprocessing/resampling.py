"""Image resampling and geometric consistency checking."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import nibabel as nib
from nilearn import image
import logging
import json

from connectomix.utils.exceptions import PreprocessingError


def check_geometric_consistency(
    func_files: List[Path],
    logger: logging.Logger,
    reference_file: Optional[Path] = None
) -> Tuple[bool, Dict[str, Dict[str, np.ndarray]]]:
    """Check if all functional images have consistent geometry.
    
    This is CRITICAL: checks ALL functional images in the dataset,
    not just the selected subset, to ensure group-level compatibility.
    
    Args:
        func_files: List of paths to ALL functional images in dataset
        logger: Logger instance
        reference_file: Path to the reference image to use for comparison.
                       If None, uses the first image in func_files.
    
    Returns:
        Tuple of (is_consistent, geometry_dict)
        - is_consistent: True if all images have same geometry
        - geometry_dict: Maps file path to {'shape': shape, 'affine': affine}
    
    Raises:
        PreprocessingError: If no functional images found
    """
    if not func_files:
        raise PreprocessingError("No functional images found for geometry check")
    
    logger.info(f"Checking geometric consistency across {len(func_files)} image(s)")
    
    # Determine and load reference geometry first
    if reference_file is not None:
        # Use specified reference file
        if reference_file not in func_files:
            logger.warning(f"Reference file {reference_file.name} not in dataset - will be loaded separately")
        ref_img = nib.load(reference_file)
        reference_shape = ref_img.shape[:3]
        reference_affine = np.round(ref_img.affine, decimals=6)
        logger.debug(f"Reference geometry from {reference_file.name}: shape={reference_shape}, affine diagonal={np.diag(reference_affine)[:3]}")
    else:
        # Use first file as reference
        ref_img = nib.load(func_files[0])
        reference_shape = ref_img.shape[:3]
        reference_affine = np.round(ref_img.affine, decimals=6)
        logger.debug(f"Reference geometry from {func_files[0].name}: shape={reference_shape}, affine diagonal={np.diag(reference_affine)[:3]}")
    
    # Collect geometry information for all files
    geometries = {}
    
    for func_file in func_files:
        img = nib.load(func_file)
        
        # Get spatial dimensions only (exclude time)
        shape = img.shape[:3]
        
        # Round affine to avoid numerical precision issues
        affine = np.round(img.affine, decimals=6)
        
        geometries[str(func_file)] = {
            'shape': shape,
            'affine': affine
        }
    
    # Check consistency
    is_consistent = True
    mismatches = []
    
    for file_path, geom in geometries.items():
        file_name = Path(file_path).name
        
        if not np.array_equal(geom['shape'], reference_shape):
            logger.warning(
                f"Shape mismatch: {file_name} has {geom['shape']} "
                f"vs reference {reference_shape}"
            )
            mismatches.append(file_name)
            is_consistent = False
        
        if not np.allclose(geom['affine'], reference_affine, rtol=1e-5):
            logger.warning(
                f"Affine mismatch: {file_name} differs from reference"
            )
            if file_name not in mismatches:
                mismatches.append(file_name)
            is_consistent = False
    
    if is_consistent:
        logger.info("✓ All images have consistent geometry - no resampling needed")
    else:
        logger.warning(
            f"⚠ Geometric inconsistencies detected in {len(mismatches)} image(s) - "
            f"resampling required"
        )
    
    return is_consistent, geometries


def resample_to_reference(
    img: Union[Path, nib.Nifti1Image],
    reference: Union[Path, nib.Nifti1Image],
    output_path: Path,
    logger: logging.Logger,
    interpolation: str = 'continuous'
) -> nib.Nifti1Image:
    """Resample image to reference space.
    
    Args:
        img: Image to resample (path or NIfTI image)
        reference: Reference image (path or NIfTI image)
        output_path: Path for resampled output
        logger: Logger instance
        interpolation: Interpolation method ('continuous', 'nearest')
    
    Returns:
        Resampled NIfTI image
    
    Raises:
        PreprocessingError: If resampling fails
    """
    # Skip if already exists
    if output_path.exists():
        logger.info(f"Resampled file exists, skipping: {output_path.name}")
        return nib.load(output_path)
    
    # Get name for logging
    if isinstance(img, (str, Path)):
        img_name = Path(img).name
        img = nib.load(img)
    else:
        img_name = "image"
    
    if isinstance(reference, (str, Path)):
        reference = nib.load(reference)
    
    logger.info(f"Resampling {img_name}")
    
    try:
        # Resample with explicit parameters to avoid FutureWarnings
        resampled = image.resample_to_img(
            img,
            reference,
            interpolation=interpolation,
            force_resample=True,
            copy_header=True
        )
        
        # Round affine to avoid numerical precision issues
        resampled_data = resampled.get_fdata()
        resampled_affine = np.round(resampled.affine, decimals=6)
        
        resampled_img = nib.Nifti1Image(
            resampled_data,
            resampled_affine,
            resampled.header
        )
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(resampled_img, output_path)
        
        logger.debug(f"  Saved to: {output_path}")
        
        return resampled_img
    
    except Exception as e:
        raise PreprocessingError(f"Failed to resample {img_name}: {e}")


def save_geometry_info(
    img: nib.Nifti1Image,
    output_path: Path,
    reference_path: Optional[Path] = None,
    reference_img: Optional[nib.Nifti1Image] = None,
    original_path: Optional[Path] = None,
    original_img: Optional[nib.Nifti1Image] = None,
    source_json: Optional[Path] = None,
) -> None:
    """Save geometric information and metadata to JSON file.
    
    Args:
        img: NIfTI image (the resampled/output image)
        output_path: Path for JSON file
        reference_path: Path to reference image if resampling was used
        reference_img: Reference NIfTI image (for saving its affine)
        original_path: Path to the original image before resampling
        original_img: Original NIfTI image (for saving its geometry)
        source_json: Path to source JSON sidecar to copy metadata from
    """
    geometry = {
        'Shape': list(img.shape),
        'Affine': np.round(img.affine, decimals=6).tolist(),
        'VoxelSize_mm': [
            float(img.header.get_zooms()[i]) 
            for i in range(min(3, len(img.header.get_zooms())))
        ],
    }
    
    # Add resampling information if applicable
    if reference_path is not None:
        geometry['Resampled'] = True
        geometry['ReferenceUsed'] = str(reference_path)
        
        # Add reference affine if reference image provided
        if reference_img is not None:
            geometry['ReferenceAffine'] = np.round(reference_img.affine, decimals=6).tolist()
            geometry['ReferenceShape'] = list(reference_img.shape[:3])
            geometry['ReferenceVoxelSize_mm'] = [
                float(reference_img.header.get_zooms()[i])
                for i in range(min(3, len(reference_img.header.get_zooms())))
            ]
        
        # Add original geometry if original image provided
        if original_path is not None:
            geometry['OriginalFile'] = str(original_path)
        if original_img is not None:
            geometry['OriginalShape'] = list(original_img.shape[:3])
            geometry['OriginalAffine'] = np.round(original_img.affine, decimals=6).tolist()
            geometry['OriginalVoxelSize_mm'] = [
                float(original_img.header.get_zooms()[i])
                for i in range(min(3, len(original_img.header.get_zooms())))
            ]
    else:
        geometry['Resampled'] = False
    
    # Try to get TR from image header
    zooms = img.header.get_zooms()
    if len(zooms) > 3:
        geometry['RepetitionTime'] = float(zooms[3])
    
    # If source JSON provided, copy relevant metadata
    if source_json and source_json.exists():
        try:
            with source_json.open() as f:
                source_meta = json.load(f)
            # Copy important fields
            for key in ['RepetitionTime', 'TaskName', 'SliceTiming', 'EchoTime']:
                if key in source_meta and key not in geometry:
                    geometry[key] = source_meta[key]
        except Exception:
            pass  # Silently ignore if can't read source JSON
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        json.dump(geometry, f, indent=2)


def validate_group_geometry(
    geometry_files: List[Path],
    logger: logging.Logger
) -> None:
    """Validate all subjects have consistent geometry for group analysis.
    
    Args:
        geometry_files: List of paths to geometry JSON files
        logger: Logger instance
    
    Raises:
        PreprocessingError: If geometric inconsistency detected
    """
    logger.info("Validating geometric consistency for group analysis")
    
    if not geometry_files:
        raise PreprocessingError(
            "No geometry files found. "
            "Re-run participant-level analysis."
        )
    
    geometries = {}
    reference_shape = None
    reference_affine = None
    
    for geom_file in geometry_files:
        with open(geom_file) as f:
            geom = json.load(f)
        
        shape = geom['Shape'][:3]  # Spatial dimensions only
        affine = np.array(geom['Affine'])
        
        subject = geom_file.parts[-2]  # Extract subject from path
        geometries[subject] = {'shape': shape, 'affine': affine}
        
        if reference_shape is None:
            reference_shape = shape
            reference_affine = affine
    
    # Check consistency
    mismatched = []
    for subject, geom in geometries.items():
        if geom['shape'] != reference_shape:
            mismatched.append(
                f"{subject}: shape {geom['shape']} != {reference_shape}"
            )
        
        if not np.allclose(geom['affine'], reference_affine, rtol=1e-5):
            mismatched.append(
                f"{subject}: affine differs from reference"
            )
    
    if mismatched:
        error_msg = (
            "Geometric inconsistency detected for group analysis:\n" +
            "\n".join(f"  - {m}" for m in mismatched) +
            "\n\nAll participants must have identical image geometry. "
            "Re-run participant-level analysis with consistent resampling."
        )
        raise PreprocessingError(error_msg)
    
    logger.info(f"✓ Geometry validation passed for {len(geometries)} subject(s)")
