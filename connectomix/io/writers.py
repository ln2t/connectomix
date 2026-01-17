"""File writers for outputs."""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


def save_nifti_with_sidecar(
    img: nib.Nifti1Image,
    output_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """Save NIfTI image with JSON sidecar.
    
    Args:
        img: NIfTI image to save
        output_path: Path for output NIfTI file
        metadata: Dictionary of metadata to save in JSON sidecar
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NIfTI image
    nib.save(img, output_path)
    
    # Add timestamp to metadata
    metadata_with_timestamp = metadata.copy()
    metadata_with_timestamp['CreationTime'] = datetime.now().isoformat()
    
    # Save JSON sidecar
    sidecar_path = output_path.with_suffix('').with_suffix('.json')
    with sidecar_path.open('w') as f:
        json.dump(metadata_with_timestamp, f, indent=2)


def save_matrix_with_sidecar(
    matrix: np.ndarray,
    output_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """Save NumPy matrix with JSON sidecar.
    
    Args:
        matrix: NumPy array to save
        output_path: Path for output .npy file
        metadata: Dictionary of metadata to save in JSON sidecar
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save matrix
    np.save(output_path, matrix)
    
    # Add matrix shape and timestamp to metadata
    metadata_with_info = metadata.copy()
    metadata_with_info['Shape'] = list(matrix.shape)
    metadata_with_info['Dtype'] = str(matrix.dtype)
    metadata_with_info['CreationTime'] = datetime.now().isoformat()
    
    # Save JSON sidecar
    sidecar_path = output_path.with_suffix('.json')
    with sidecar_path.open('w') as f:
        json.dump(metadata_with_info, f, indent=2)


def save_tsv(
    dataframe,
    output_path: Path,
    metadata: Dict[str, Any] = None
) -> None:
    """Save pandas DataFrame as TSV with optional JSON sidecar.
    
    Args:
        dataframe: pandas DataFrame to save
        output_path: Path for output TSV file
        metadata: Optional dictionary of metadata to save in JSON sidecar
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save TSV
    dataframe.to_csv(output_path, sep='\t', index=False)
    
    # Save JSON sidecar if metadata provided
    if metadata:
        metadata_with_info = metadata.copy()
        metadata_with_info['Columns'] = list(dataframe.columns)
        metadata_with_info['NumRows'] = len(dataframe)
        metadata_with_info['CreationTime'] = datetime.now().isoformat()
        
        sidecar_path = output_path.with_suffix('.json')
        with sidecar_path.open('w') as f:
            json.dump(metadata_with_info, f, indent=2)


def save_json(
    data: Dict[str, Any],
    output_path: Path
) -> None:
    """Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        output_path: Path for output JSON file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any Path objects to strings
    data_serializable = _make_serializable(data)
    
    # Save JSON
    with output_path.open('w') as f:
        json.dump(data_serializable, f, indent=2)


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format.
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON-serializable version of object
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    else:
        return obj
