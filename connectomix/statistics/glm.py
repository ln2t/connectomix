"""Second-level GLM for group-level analysis.

This module provides functions for fitting second-level general linear models
to connectivity maps and computing statistical contrasts for group comparisons
and covariate analyses.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from nilearn.image import smooth_img

from connectomix.utils.exceptions import StatisticalError

logger = logging.getLogger(__name__)


def build_design_matrix(
    subjects: List[str],
    participants_df: Optional[pd.DataFrame] = None,
    covariates: Optional[List[str]] = None,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Build a design matrix for second-level GLM analysis.
    
    Creates a design matrix from participants metadata. Supports continuous
    covariates, categorical variables (automatically dummy-coded), and
    optional intercept.
    
    Args:
        subjects: List of subject IDs (e.g., ["sub-01", "sub-02"]).
            Must match the order of connectivity maps.
        participants_df: DataFrame from participants.tsv with subject metadata.
            Index or 'participant_id' column should contain subject IDs.
            If None, only intercept can be added.
        covariates: List of column names from participants_df to include
            as covariates. Categorical columns are automatically dummy-coded.
        add_intercept: Whether to add an intercept column to the design matrix.
    
    Returns:
        Design matrix as DataFrame with subjects as rows and regressors as columns.
    
    Raises:
        StatisticalError: If subjects are missing from participants_df or
            requested covariates are not found.
    
    Example:
        >>> subjects = ["sub-01", "sub-02", "sub-03", "sub-04"]
        >>> participants = pd.DataFrame({
        ...     "participant_id": subjects,
        ...     "age": [25, 30, 28, 35],
        ...     "group": ["patient", "patient", "control", "control"]
        ... })
        >>> dm = build_design_matrix(
        ...     subjects, participants, covariates=["age", "group"]
        ... )
    """
    # Normalize subject IDs (ensure 'sub-' prefix)
    normalized_subjects = []
    for sub in subjects:
        if not sub.startswith("sub-"):
            normalized_subjects.append(f"sub-{sub}")
        else:
            normalized_subjects.append(sub)
    
    # Initialize design matrix with subjects as index
    design_matrix = pd.DataFrame(index=normalized_subjects)
    
    if covariates and len(covariates) > 0:
        if participants_df is None:
            raise StatisticalError(
                "participants_df is required when covariates are specified.\n"
                "Provide participants.tsv data or remove covariates from config."
            )
        
        # Normalize participants_df index
        if "participant_id" in participants_df.columns:
            participants_df = participants_df.set_index("participant_id")
        
        # Normalize index to have 'sub-' prefix
        participants_df.index = [
            idx if idx.startswith("sub-") else f"sub-{idx}"
            for idx in participants_df.index
        ]
        
        # Check all subjects are in participants_df
        missing = set(normalized_subjects) - set(participants_df.index)
        if missing:
            raise StatisticalError(
                f"Subjects not found in participants.tsv: {sorted(missing)}\n"
                f"Available subjects: {sorted(participants_df.index.tolist())}"
            )
        
        # Check all covariates exist
        missing_covariates = set(covariates) - set(participants_df.columns)
        if missing_covariates:
            raise StatisticalError(
                f"Covariates not found in participants.tsv: {sorted(missing_covariates)}\n"
                f"Available columns: {sorted(participants_df.columns.tolist())}"
            )
        
        # Extract covariates for our subjects
        covariate_data = participants_df.loc[normalized_subjects, covariates].copy()
        
        # Process each covariate
        for covariate in covariates:
            col_data = covariate_data[covariate]
            
            # Check for categorical (object/category dtype)
            if col_data.dtype == "object" or col_data.dtype.name == "category":
                # Dummy-code categorical variables
                dummies = pd.get_dummies(col_data, prefix=covariate, drop_first=True)
                for dummy_col in dummies.columns:
                    design_matrix[dummy_col] = dummies[dummy_col].values
                logger.debug(
                    f"Dummy-coded categorical covariate '{covariate}' "
                    f"into {len(dummies.columns)} columns"
                )
            else:
                # Continuous covariate - demean for interpretability
                centered = col_data - col_data.mean()
                design_matrix[covariate] = centered.values
                logger.debug(
                    f"Added continuous covariate '{covariate}' "
                    f"(mean={col_data.mean():.2f}, std={col_data.std():.2f})"
                )
    
    # Add intercept if requested
    if add_intercept:
        # Insert intercept as first column
        design_matrix.insert(0, "intercept", 1.0)
        logger.debug("Added intercept column")
    
    logger.info(
        f"Built design matrix: {len(design_matrix)} subjects × "
        f"{len(design_matrix.columns)} regressors"
    )
    logger.debug(f"Design matrix columns: {list(design_matrix.columns)}")
    
    return design_matrix


def fit_second_level_model(
    stat_maps: List[Union[str, Path, nib.Nifti1Image]],
    design_matrix: pd.DataFrame,
    smoothing_fwhm: Optional[float] = None,
    mask_img: Optional[Union[str, Path, nib.Nifti1Image]] = None,
) -> SecondLevelModel:
    """Fit a second-level GLM to a set of statistical maps.
    
    Uses nilearn's SecondLevelModel to perform voxel-wise regression
    of connectivity maps on the design matrix.
    
    Args:
        stat_maps: List of paths to NIfTI files or Nifti1Image objects.
            One map per subject in the same order as design matrix rows.
        design_matrix: Design matrix from build_design_matrix().
            Rows must match stat_maps order.
        smoothing_fwhm: FWHM in mm for spatial smoothing. Applied before
            model fitting. None for no smoothing.
        mask_img: Brain mask. If None, computed automatically from data.
    
    Returns:
        Fitted SecondLevelModel object.
    
    Raises:
        StatisticalError: If number of maps doesn't match design matrix rows.
    
    Example:
        >>> maps = [f"sub-{i:02d}_connectivity.nii.gz" for i in range(1, 5)]
        >>> model = fit_second_level_model(maps, design_matrix, smoothing_fwhm=8.0)
    """
    # Validate inputs
    if len(stat_maps) != len(design_matrix):
        raise StatisticalError(
            f"Number of stat maps ({len(stat_maps)}) doesn't match "
            f"design matrix rows ({len(design_matrix)}).\n"
            f"Ensure one connectivity map per subject."
        )
    
    # Convert paths to strings for nilearn
    stat_maps_str = [
        str(m) if isinstance(m, Path) else m for m in stat_maps
    ]
    
    # Apply smoothing if requested
    if smoothing_fwhm is not None and smoothing_fwhm > 0:
        logger.info(f"Smoothing maps with FWHM = {smoothing_fwhm} mm")
        stat_maps_str = [
            smooth_img(m, fwhm=smoothing_fwhm) for m in stat_maps_str
        ]
    
    # Create and fit model
    logger.info(f"Fitting second-level model with {len(stat_maps)} subjects")
    
    model = SecondLevelModel(mask_img=mask_img)
    model.fit(stat_maps_str, design_matrix=design_matrix)
    
    logger.info("Second-level model fitted successfully")
    
    return model


def compute_contrast(
    model: SecondLevelModel,
    contrast_def: Union[str, List[float], np.ndarray],
    design_matrix: pd.DataFrame,
    stat_type: str = "t",
    output_type: str = "stat",
) -> nib.Nifti1Image:
    """Compute a statistical contrast from a fitted second-level model.
    
    Args:
        model: Fitted SecondLevelModel from fit_second_level_model().
        contrast_def: Contrast definition. Can be:
            - String: Column name (e.g., "age", "intercept", "group_patient")
            - List/array: Contrast vector (e.g., [1, -1, 0] for A vs B)
        design_matrix: Design matrix used to fit the model.
            Required for string contrast definitions.
        stat_type: Type of statistic. "t" for t-statistic, "F" for F-statistic.
        output_type: Type of output. "stat" for statistic map, "z_score" for
            z-transformed map, "p_value" for p-value map, "effect_size" for
            effect size map.
    
    Returns:
        NIfTI image containing the statistical map.
    
    Raises:
        StatisticalError: If contrast definition is invalid or column not found.
    
    Example:
        >>> # Test if intercept (mean connectivity) is significant
        >>> t_map = compute_contrast(model, "intercept", design_matrix)
        >>> 
        >>> # Test group difference (patient vs control)
        >>> t_map = compute_contrast(model, "group_patient", design_matrix)
        >>> 
        >>> # Custom contrast vector
        >>> t_map = compute_contrast(model, [1, -1, 0], design_matrix)
    """
    # Build contrast vector from string definition
    if isinstance(contrast_def, str):
        contrast_vector = _string_to_contrast_vector(contrast_def, design_matrix)
        contrast_name = contrast_def
    else:
        contrast_vector = np.array(contrast_def)
        contrast_name = "custom"
        
        # Validate length
        if len(contrast_vector) != len(design_matrix.columns):
            raise StatisticalError(
                f"Contrast vector length ({len(contrast_vector)}) doesn't match "
                f"number of design matrix columns ({len(design_matrix.columns)}).\n"
                f"Design matrix columns: {list(design_matrix.columns)}"
            )
    
    logger.info(f"Computing contrast '{contrast_name}' ({stat_type}-statistic)")
    logger.debug(f"Contrast vector: {contrast_vector}")
    
    # Compute contrast
    stat_map = model.compute_contrast(
        contrast_vector,
        stat_type=stat_type,
        output_type=output_type,
    )
    
    return stat_map


def _string_to_contrast_vector(
    contrast_str: str,
    design_matrix: pd.DataFrame,
) -> np.ndarray:
    """Convert string contrast definition to numeric vector.
    
    Args:
        contrast_str: String contrast definition.
            - Simple: column name (e.g., "age", "intercept")
            - Difference: "A-B" for contrast between two columns
        design_matrix: Design matrix with column names.
    
    Returns:
        Contrast vector as numpy array.
    
    Raises:
        StatisticalError: If column names not found in design matrix.
    """
    columns = list(design_matrix.columns)
    n_cols = len(columns)
    
    # Check for difference contrast (A-B)
    if "-" in contrast_str and not contrast_str.startswith("-"):
        parts = contrast_str.split("-")
        if len(parts) == 2:
            col_a, col_b = parts[0].strip(), parts[1].strip()
            
            if col_a not in columns:
                raise StatisticalError(
                    f"Contrast column '{col_a}' not found in design matrix.\n"
                    f"Available columns: {columns}"
                )
            if col_b not in columns:
                raise StatisticalError(
                    f"Contrast column '{col_b}' not found in design matrix.\n"
                    f"Available columns: {columns}"
                )
            
            vector = np.zeros(n_cols)
            vector[columns.index(col_a)] = 1.0
            vector[columns.index(col_b)] = -1.0
            return vector
    
    # Simple column contrast
    if contrast_str not in columns:
        raise StatisticalError(
            f"Contrast '{contrast_str}' not found in design matrix columns.\n"
            f"Available columns: {columns}\n"
            f"For difference contrasts, use format 'columnA-columnB'."
        )
    
    vector = np.zeros(n_cols)
    vector[columns.index(contrast_str)] = 1.0
    return vector


def save_design_matrix(
    design_matrix: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save design matrix to TSV file with JSON sidecar.
    
    Args:
        design_matrix: Design matrix DataFrame.
        output_path: Output path for TSV file.
    
    Returns:
        Path to saved TSV file.
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save TSV
    design_matrix.to_csv(output_path, sep="\t")
    
    # Save sidecar
    sidecar = {
        "Description": "Design matrix for second-level GLM analysis",
        "Columns": list(design_matrix.columns),
        "NumberOfSubjects": len(design_matrix),
        "NumberOfRegressors": len(design_matrix.columns),
    }
    
    sidecar_path = output_path.with_suffix(".json")
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2)
    
    logger.info(f"Saved design matrix to {output_path}")
    
    return output_path
