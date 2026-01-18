"""File readers for various formats."""

import fnmatch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json


def expand_confound_wildcards(
    confound_patterns: List[str],
    available_columns: List[str],
) -> List[str]:
    """Expand wildcard patterns to match actual confound column names.
    
    Supports shell-style wildcards:
    - '*' matches any number of characters
    - '?' matches single character
    - '[seq]' matches any character in seq
    
    Args:
        confound_patterns: List of confound names, may contain wildcards.
        available_columns: List of available column names in confounds file.
    
    Returns:
        List of expanded confound column names (no duplicates, order preserved).
    
    Example:
        >>> expand_confound_wildcards(['trans_*', 'rot_*'], columns)
        ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        
        >>> expand_confound_wildcards(['a_comp_cor_0*'], columns)
        ['a_comp_cor_00', 'a_comp_cor_01', ..., 'a_comp_cor_09']
    """
    expanded = []
    seen = set()
    
    for pattern in confound_patterns:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            # Pattern contains wildcards - expand it
            matches = fnmatch.filter(available_columns, pattern)
            # Sort matches for consistent ordering
            matches = sorted(matches)
            for match in matches:
                if match not in seen:
                    expanded.append(match)
                    seen.add(match)
        else:
            # No wildcards - use as-is
            if pattern not in seen:
                expanded.append(pattern)
                seen.add(pattern)
    
    return expanded


def load_confounds(
    confounds_path: Path,
    confound_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Load and extract confounds from TSV file.
    
    Supports wildcard patterns in confound names:
    - '*' matches any number of characters
    - '?' matches single character
    
    Args:
        confounds_path: Path to fMRIPrep confounds TSV file
        confound_names: List of confound column names or wildcard patterns
    
    Returns:
        Tuple of (confounds_array, expanded_names):
        - confounds_array: NumPy array of shape (n_timepoints, n_confounds)
        - expanded_names: List of actual confound column names used
    
    Raises:
        ValueError: If confound columns don't exist or no matches found
        FileNotFoundError: If confounds file doesn't exist
    
    Example:
        >>> confounds, names = load_confounds(path, ['trans_*', 'rot_*', 'csf'])
        >>> print(names)
        ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf']
    """
    if not confounds_path.exists():
        raise FileNotFoundError(f"Confounds file not found: {confounds_path}")
    
    # Load TSV file
    df = pd.read_csv(confounds_path, sep='\t')
    
    # Expand wildcard patterns
    expanded_names = expand_confound_wildcards(confound_names, df.columns.tolist())
    
    # Check if any patterns had no matches
    for pattern in confound_names:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            matches = fnmatch.filter(df.columns.tolist(), pattern)
            if not matches:
                available = sorted(df.columns.tolist())
                raise ValueError(
                    f"No confounds matching pattern '{pattern}' in {confounds_path.name}.\n"
                    f"  Available columns: {available[:15]}{'...' if len(available) > 15 else ''}"
                )
        else:
            # Literal name - check it exists
            if pattern not in df.columns:
                available = sorted(df.columns.tolist())
                # Suggest similar columns
                similar = [c for c in available if pattern.lower() in c.lower()]
                suggestion = f"\n  Similar columns: {similar[:5]}" if similar else ""
                raise ValueError(
                    f"Confound '{pattern}' not found in {confounds_path.name}.{suggestion}\n"
                    f"  Available columns: {available[:15]}{'...' if len(available) > 15 else ''}"
                )
    
    if not expanded_names:
        raise ValueError(
            f"No confounds selected after expanding patterns: {confound_names}"
        )
    
    # Extract confounds
    confounds = df[expanded_names].values
    
    # Handle NaN values (replace with 0 or column mean)
    if np.any(np.isnan(confounds)):
        # Replace NaN with column mean
        col_means = np.nanmean(confounds, axis=0)
        nan_mask = np.isnan(confounds)
        confounds[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    
    return confounds, expanded_names


def load_seeds_file(seeds_path: Path) -> Tuple[List[str], np.ndarray]:
    """Load seeds from TSV file.
    
    Expected format:
        name    x    y    z
        PCC     0   -52   18
        mPFC    0    52    0
    
    Args:
        seeds_path: Path to seeds TSV file
    
    Returns:
        Tuple of (seed_names, coordinates_array)
        - seed_names: List of seed region names
        - coordinates_array: NumPy array of shape (n_seeds, 3) with MNI coordinates
    
    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If seeds file doesn't exist
    """
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seeds file not found: {seeds_path}")
    
    # Load TSV file
    df = pd.read_csv(seeds_path, sep='\t')
    
    # Check for required columns
    required_cols = ['name', 'x', 'y', 'z']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Seeds file missing required columns: {sorted(missing)}\n"
            f"Required columns: {required_cols}\n"
            f"Found columns: {df.columns.tolist()}"
        )
    
    # Extract data
    names = df['name'].tolist()
    coords = df[['x', 'y', 'z']].values.astype(float)
    
    return names, coords


def load_participants_tsv(bids_dir: Path) -> pd.DataFrame:
    """Load participants.tsv file from BIDS dataset.
    
    Args:
        bids_dir: Path to BIDS dataset root
    
    Returns:
        DataFrame with participant information
    
    Raises:
        FileNotFoundError: If participants.tsv doesn't exist
    """
    participants_path = bids_dir / "participants.tsv"
    
    if not participants_path.exists():
        raise FileNotFoundError(
            f"participants.tsv not found in {bids_dir}\n"
            f"This file is required for group-level analysis."
        )
    
    df = pd.read_csv(participants_path, sep='\t')
    
    # Ensure participant_id column exists
    if 'participant_id' not in df.columns:
        raise ValueError(
            f"participants.tsv missing 'participant_id' column"
        )
    
    return df


def load_json_sidecar(json_path: Path) -> Dict[str, Any]:
    """Load JSON sidecar file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Dictionary with JSON contents
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with json_path.open() as f:
        data = json.load(f)
    
    return data


def get_repetition_time(json_path: Path) -> float:
    """Get repetition time (TR) from JSON sidecar.
    
    Args:
        json_path: Path to functional image JSON sidecar
    
    Returns:
        TR in seconds
    
    Raises:
        ValueError: If TR not found in JSON
    """
    data = load_json_sidecar(json_path)
    
    if 'RepetitionTime' not in data:
        raise ValueError(
            f"RepetitionTime not found in {json_path.name}"
        )
    
    return float(data['RepetitionTime'])
