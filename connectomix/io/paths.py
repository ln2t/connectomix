"""BIDS dataset path validation and initialization."""

from pathlib import Path
import json
from typing import Optional
import logging

from connectomix.utils.exceptions import BIDSError


def validate_bids_dir(path: Path) -> None:
    """Validate BIDS directory structure.
    
    Args:
        path: Path to BIDS dataset root
    
    Raises:
        BIDSError: If directory is not a valid BIDS dataset
    """
    if not path.exists():
        raise BIDSError(
            f"BIDS directory not found: {path}\n"
            f"Please check the path and try again."
        )
    
    if not path.is_dir():
        raise BIDSError(
            f"BIDS path is not a directory: {path}"
        )
    
    # Check for dataset_description.json
    dataset_desc = path / "dataset_description.json"
    if not dataset_desc.exists():
        raise BIDSError(
            f"Not a valid BIDS dataset: {path}\n"
            f"Missing dataset_description.json\n"
            f"See https://bids.neuroimaging.io for BIDS specification."
        )
    
    # Validate dataset_description.json content
    try:
        with dataset_desc.open() as f:
            desc = json.load(f)
        
        if "Name" not in desc:
            raise BIDSError(
                f"Invalid dataset_description.json: missing 'Name' field"
            )
        
        if "BIDSVersion" not in desc:
            raise BIDSError(
                f"Invalid dataset_description.json: missing 'BIDSVersion' field"
            )
    
    except json.JSONDecodeError as e:
        raise BIDSError(
            f"Invalid dataset_description.json: {e}"
        )


def create_dataset_description(
    output_dir: Path,
    version: str,
    source_datasets: Optional[list] = None
) -> None:
    """Create dataset_description.json for derivatives.
    
    Args:
        output_dir: Path to output directory
        version: Connectomix version string
        source_datasets: Optional list of source dataset descriptions
    """
    dataset_desc = {
        "Name": "Connectomix derivatives",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{
            "Name": "Connectomix",
            "Version": version,
            "CodeURL": "https://github.com/ln2t/connectomix"
        }]
    }
    
    if source_datasets:
        dataset_desc["SourceDatasets"] = source_datasets
    
    output_path = output_dir / "dataset_description.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        json.dump(dataset_desc, f, indent=2)


def create_output_directories(
    output_dir: Path,
    analysis_level: str = "participant"
) -> None:
    """Create standard output directory structure.
    
    Args:
        output_dir: Path to output directory
        analysis_level: Either "participant" or "group"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config backup directory
    config_dir = output_dir / "config" / "backups"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create group directory if needed
    if analysis_level == "group":
        group_dir = output_dir / "group"
        group_dir.mkdir(parents=True, exist_ok=True)


def validate_derivatives_dir(
    derivatives_dir: Path,
    derivative_name: str = "fmriprep"
) -> None:
    """Validate derivatives directory.
    
    Args:
        derivatives_dir: Path to derivatives directory
        derivative_name: Name of the derivative (e.g., "fmriprep")
    
    Raises:
        BIDSError: If directory is not valid
    """
    if not derivatives_dir.exists():
        raise BIDSError(
            f"{derivative_name} derivatives directory not found: {derivatives_dir}\n"
            f"Please specify the correct path using:\n"
            f"  --derivatives {derivative_name}=/path/to/{derivative_name}"
        )
    
    if not derivatives_dir.is_dir():
        raise BIDSError(
            f"{derivative_name} path is not a directory: {derivatives_dir}"
        )
    
    # Check for dataset_description.json
    dataset_desc = derivatives_dir / "dataset_description.json"
    if not dataset_desc.exists():
        raise BIDSError(
            f"Not a valid {derivative_name} derivatives directory: {derivatives_dir}\n"
            f"Missing dataset_description.json"
        )
