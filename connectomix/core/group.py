"""Group-level pipeline orchestration.

This module orchestrates the complete group-level analysis workflow,
coordinating statistical modeling, inference, and result reporting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from connectomix.config.defaults import GroupConfig
from connectomix.config.loader import save_config
from connectomix.io.bids import create_bids_layout, build_bids_path
from connectomix.io.paths import create_dataset_description
from connectomix.io.readers import load_participants_tsv
from connectomix.io.writers import save_nifti_with_sidecar
from connectomix.preprocessing.resampling import validate_group_geometry
from connectomix.statistics.glm import (
    build_design_matrix,
    fit_second_level_model,
    compute_contrast,
    save_design_matrix,
)
from connectomix.statistics.permutation import run_permutation_test
from connectomix.statistics.thresholding import (
    apply_uncorrected_threshold,
    apply_fdr_threshold,
    apply_fwe_threshold,
)
from connectomix.statistics.clustering import label_clusters, save_cluster_table
from connectomix.utils.exceptions import BIDSError, ConnectomixError, StatisticalError
from connectomix.utils.logging import timer, log_section


def run_group_pipeline(
    bids_dir: Path,
    output_dir: Path,
    config: GroupConfig,
    derivatives: Optional[Dict[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Path]]:
    """Run the complete group-level statistical analysis pipeline.
    
    This function orchestrates:
    1. Discovery of participant-level connectivity outputs
    2. Validation of geometric consistency across subjects
    3. Loading participant metadata and building design matrix
    4. Fitting second-level GLM
    5. Computing statistical contrasts
    6. Running permutation testing for FWE correction (if requested)
    7. Applying thresholding strategies
    8. Extracting and labeling clusters
    9. Saving all outputs with BIDS-compliant naming
    
    Args:
        bids_dir: Path to BIDS dataset root.
        output_dir: Path for output derivatives (should be same as participant-level).
        config: GroupConfig instance with analysis parameters.
        derivatives: Optional dict mapping derivative names to paths.
        logger: Logger instance. If None, creates one.
    
    Returns:
        Dictionary with keys mapping to lists of output file paths:
            - 'stat_maps': Unthresholded statistical maps
            - 'thresholded_maps': Thresholded maps for each strategy
            - 'cluster_tables': Cluster tables
            - 'design_matrix': Design matrix TSV
    
    Raises:
        BIDSError: If BIDS dataset or participant outputs are invalid.
        StatisticalError: If statistical analysis fails.
    
    Example:
        >>> from connectomix.config.defaults import GroupConfig
        >>> config = GroupConfig(
        ...     subject=["01", "02", "03", "04"],
        ...     task="rest",
        ...     method="roiToRoi",
        ...     covariates=["group", "age"],
        ...     contrast="group_patient",
        ...     thresholding_strategies=["fdr", "fwe"]
        ... )
        >>> outputs = run_group_pipeline(
        ...     Path("/data/bids"),
        ...     Path("/data/output"),
        ...     config
        ... )
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Validate configuration
    config.validate()
    
    outputs = {
        'stat_maps': [],
        'thresholded_maps': [],
        'cluster_tables': [],
        'design_matrix': [],
    }
    
    with timer(logger, "Group-level analysis"):
        # === Step 1: Setup ===
        log_section(logger, "Setup")
        
        # Create output directories
        group_dir = output_dir / "group" / config.method / config.analysis_name
        group_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup configuration
        config_backup_dir = output_dir / "config" / "backups"
        save_config(config, config_backup_dir)
        
        # === Step 2: Discover participant-level outputs ===
        log_section(logger, "File Discovery")
        
        # Find connectivity maps from participant-level analysis
        connectivity_maps, subjects = _find_participant_outputs(
            output_dir, config, logger
        )
        
        n_subjects = len(subjects)
        logger.info(f"Found {n_subjects} participant-level output(s)")
        
        if n_subjects < 3:
            raise StatisticalError(
                f"Group analysis requires at least 3 subjects, found {n_subjects}.\n"
                "Run participant-level analysis first."
            )
        
        # === Step 3: Validate geometry ===
        log_section(logger, "Geometry Validation")
        
        if config.method in ["seedToVoxel", "roiToVoxel"]:
            # For voxel-level methods, check geometry consistency
            validate_group_geometry([Path(m) for m in connectivity_maps], logger)
        
        # === Step 4: Load participant data and build design matrix ===
        log_section(logger, "Design Matrix")
        
        # Load participants.tsv
        participants_df = _load_participants_data(bids_dir, subjects, logger)
        
        # Build design matrix
        design_matrix = build_design_matrix(
            subjects=subjects,
            participants_df=participants_df,
            covariates=config.covariates,
            add_intercept=config.add_intercept,
        )
        
        logger.info(f"Design matrix: {len(design_matrix)} subjects × {len(design_matrix.columns)} regressors")
        logger.debug(f"Regressors: {list(design_matrix.columns)}")
        
        # Save design matrix
        dm_path = _build_group_output_path(group_dir, config, "designMatrix", ".tsv")
        save_design_matrix(design_matrix, dm_path)
        outputs['design_matrix'].append(dm_path)
        
        # === Step 5: Fit GLM and compute contrast ===
        log_section(logger, "Statistical Modeling")
        
        if config.method in ["seedToVoxel", "roiToVoxel"]:
            # Voxel-level analysis
            stat_map_path, stat_map = _run_voxel_level_glm(
                connectivity_maps, design_matrix, config, group_dir, logger
            )
            outputs['stat_maps'].append(stat_map_path)
            
            # === Step 6: Permutation testing (if requested) ===
            null_distribution = None
            
            if "fwe" in config.thresholding_strategies:
                log_section(logger, "Permutation Testing")
                
                perm_results = run_permutation_test(
                    stat_maps=connectivity_maps,
                    design_matrix=design_matrix,
                    contrast=config.contrast,
                    n_permutations=config.n_permutations,
                    n_jobs=config.n_jobs,
                    two_sided=config.two_sided_test,
                    smoothing_fwhm=config.smoothing,
                )
                
                null_distribution = perm_results['null_distribution']
                
                logger.info(
                    f"FWE thresholds: 95%={perm_results['threshold_95']:.3f}, "
                    f"99%={perm_results['threshold_99']:.3f}"
                )
            
            # === Step 7: Apply thresholding strategies ===
            log_section(logger, "Thresholding")
            
            thresholded_outputs = _apply_thresholding_strategies(
                stat_map=stat_map,
                null_distribution=null_distribution,
                config=config,
                group_dir=group_dir,
                logger=logger,
            )
            outputs['thresholded_maps'].extend(thresholded_outputs['maps'])
            
            # === Step 8: Cluster analysis ===
            log_section(logger, "Cluster Analysis")
            
            for threshold_info in thresholded_outputs['threshold_info']:
                cluster_table = label_clusters(
                    stat_map=threshold_info['map'],
                    threshold=threshold_info['threshold'],
                    atlas_name="aal",
                    min_cluster_size=10,
                    two_sided=config.two_sided_test,
                )
                
                if not cluster_table.empty:
                    cluster_path = _build_group_output_path(
                        group_dir, config,
                        f"threshold-{threshold_info['strategy']}_clusterTable",
                        ".tsv"
                    )
                    save_cluster_table(
                        cluster_table, cluster_path,
                        metadata={
                            "ThresholdStrategy": threshold_info['strategy'],
                            "ThresholdValue": threshold_info['threshold'],
                            "Alpha": threshold_info['alpha'],
                        }
                    )
                    outputs['cluster_tables'].append(cluster_path)
                    
                    logger.info(
                        f"  {threshold_info['strategy']}: "
                        f"{len(cluster_table)} cluster(s)"
                    )
        
        else:
            # Matrix-level analysis (seed-to-seed, roi-to-roi)
            # Group analysis for matrices is different - compute group mean/stats
            logger.warning(
                f"Group-level analysis for method '{config.method}' "
                f"is not yet fully implemented. "
                f"Matrix outputs will be averaged."
            )
            
            mean_matrix_path = _compute_group_matrix_stats(
                connectivity_maps, config, group_dir, logger
            )
            outputs['stat_maps'].append(mean_matrix_path)
        
        # === Summary ===
        log_section(logger, "Summary")
        
        logger.info(f"Analyzed {n_subjects} subjects")
        logger.info(f"Contrast: {config.contrast}")
        logger.info(f"Thresholding strategies: {config.thresholding_strategies}")
        logger.info(f"Outputs saved to: {group_dir}")
    
    return outputs


def _find_participant_outputs(
    output_dir: Path,
    config: GroupConfig,
    logger: logging.Logger,
) -> Tuple[List[str], List[str]]:
    """Find participant-level connectivity outputs."""
    connectivity_maps = []
    subjects = []
    
    # Build pattern based on method and entities
    pattern_parts = []
    
    if config.task:
        pattern_parts.append(f"task-{config.task}")
    if config.space:
        pattern_parts.append(f"space-{config.space}")
    
    # Look for participant directories
    for sub_dir in sorted(output_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        
        subject_id = sub_dir.name.replace("sub-", "")
        
        # Apply subject filter if specified
        if config.subject and subject_id not in config.subject:
            continue
        
        # Find connectivity outputs
        if config.method in ["seedToVoxel", "roiToVoxel"]:
            # Look for NIfTI effect size maps
            search_pattern = f"*method-{config.method}*effectSize*.nii.gz"
            matches = list(sub_dir.rglob(search_pattern))
            
            # Filter by task/space if specified
            for match in matches:
                name = match.name
                if config.task and f"task-{config.task}" not in name:
                    continue
                if config.space and f"space-{config.space}" not in name:
                    continue
                
                connectivity_maps.append(str(match))
                subjects.append(subject_id)
                break  # One per subject for now
        
        else:
            # Look for numpy matrices
            search_pattern = f"*method-{config.method}*correlation*.npy"
            matches = list(sub_dir.rglob(search_pattern))
            
            for match in matches:
                name = match.name
                if config.task and f"task-{config.task}" not in name:
                    continue
                
                connectivity_maps.append(str(match))
                subjects.append(subject_id)
                break
    
    if not connectivity_maps:
        raise BIDSError(
            f"No participant-level outputs found for method '{config.method}'.\n"
            f"Please run participant-level analysis first.\n"
            f"Searched in: {output_dir}"
        )
    
    logger.info(f"Found connectivity outputs for {len(subjects)} subjects")
    
    return connectivity_maps, subjects


def _load_participants_data(
    bids_dir: Path,
    subjects: List[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load and filter participants.tsv."""
    participants_path = bids_dir / "participants.tsv"
    
    if not participants_path.exists():
        logger.warning(
            f"participants.tsv not found at {participants_path}. "
            f"Covariates will not be available."
        )
        return None
    
    df = load_participants_tsv(participants_path)
    
    # Normalize participant_id
    if 'participant_id' in df.columns:
        df['participant_id'] = df['participant_id'].str.replace('sub-', '')
    
    logger.debug(f"Loaded participants.tsv with {len(df)} rows")
    
    return df


def _run_voxel_level_glm(
    connectivity_maps: List[str],
    design_matrix: pd.DataFrame,
    config: GroupConfig,
    group_dir: Path,
    logger: logging.Logger,
) -> Tuple[Path, nib.Nifti1Image]:
    """Run GLM for voxel-level methods."""
    logger.info(f"Fitting second-level GLM with {len(connectivity_maps)} maps")
    
    # Fit model
    model = fit_second_level_model(
        stat_maps=connectivity_maps,
        design_matrix=design_matrix,
        smoothing_fwhm=config.smoothing if config.smoothing > 0 else None,
    )
    
    # Compute contrast
    stat_map = compute_contrast(
        model=model,
        contrast_def=config.contrast,
        design_matrix=design_matrix,
        stat_type="t",
        output_type="stat",
    )
    
    # Save stat map
    stat_map_path = _build_group_output_path(group_dir, config, "stat-t", ".nii.gz")
    
    metadata = {
        "Description": f"T-statistic map for contrast: {config.contrast}",
        "Contrast": config.contrast,
        "NumberOfSubjects": len(connectivity_maps),
        "SmoothingFWHM": config.smoothing,
    }
    
    save_nifti_with_sidecar(stat_map, stat_map_path, metadata)
    logger.info(f"Saved t-statistic map: {stat_map_path.name}")
    
    return stat_map_path, stat_map


def _apply_thresholding_strategies(
    stat_map: nib.Nifti1Image,
    null_distribution: Optional[np.ndarray],
    config: GroupConfig,
    group_dir: Path,
    logger: logging.Logger,
) -> Dict:
    """Apply all requested thresholding strategies."""
    results = {
        'maps': [],
        'threshold_info': [],
    }
    
    for strategy in config.thresholding_strategies:
        logger.info(f"Applying {strategy} threshold")
        
        if strategy == "uncorrected":
            thresholded, threshold = apply_uncorrected_threshold(
                stat_map,
                alpha=config.uncorrected_alpha,
                two_sided=config.two_sided_test,
            )
            alpha = config.uncorrected_alpha
            
        elif strategy == "fdr":
            thresholded, threshold = apply_fdr_threshold(
                stat_map,
                alpha=config.fdr_alpha,
                two_sided=config.two_sided_test,
            )
            alpha = config.fdr_alpha
            
        elif strategy == "fwe":
            if null_distribution is None:
                logger.warning("FWE requested but no null distribution available")
                continue
            
            thresholded, threshold = apply_fwe_threshold(
                stat_map,
                null_distribution=null_distribution,
                alpha=config.fwe_alpha,
                two_sided=config.two_sided_test,
            )
            alpha = config.fwe_alpha
            
        else:
            logger.warning(f"Unknown thresholding strategy: {strategy}")
            continue
        
        # Save thresholded map
        output_path = _build_group_output_path(
            group_dir, config, f"threshold-{strategy}_stat-t", ".nii.gz"
        )
        
        metadata = {
            "Description": f"T-statistic map thresholded with {strategy} correction",
            "ThresholdStrategy": strategy,
            "ThresholdValue": float(threshold),
            "Alpha": alpha,
            "TwoSided": config.two_sided_test,
        }
        
        save_nifti_with_sidecar(thresholded, output_path, metadata)
        results['maps'].append(output_path)
        
        results['threshold_info'].append({
            'strategy': strategy,
            'threshold': threshold,
            'alpha': alpha,
            'map': thresholded,
        })
        
        logger.info(f"  {strategy}: threshold={threshold:.3f}")
    
    return results


def _compute_group_matrix_stats(
    connectivity_maps: List[str],
    config: GroupConfig,
    group_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Compute group statistics for matrix-level methods."""
    # Load all matrices
    matrices = [np.load(m) for m in connectivity_maps]
    
    # Stack and compute mean
    stacked = np.stack(matrices, axis=0)
    mean_matrix = np.mean(stacked, axis=0)
    
    # Save mean matrix
    output_path = _build_group_output_path(
        group_dir, config, "stat-mean_correlation", ".npy"
    )
    
    np.save(output_path, mean_matrix)
    
    # Save sidecar
    import json
    sidecar = {
        "Description": "Group mean connectivity matrix",
        "NumberOfSubjects": len(matrices),
        "MatrixShape": list(mean_matrix.shape),
    }
    
    sidecar_path = output_path.with_suffix('.json')
    with sidecar_path.open('w') as f:
        json.dump(sidecar, f, indent=2)
    
    logger.info(f"Saved group mean matrix: {output_path.name}")
    
    return output_path


def _build_group_output_path(
    group_dir: Path,
    config: GroupConfig,
    suffix: str,
    extension: str,
) -> Path:
    """Build output path for group-level files."""
    parts = []
    
    if config.task:
        parts.append(f"task-{config.task}")
    if config.space:
        parts.append(f"space-{config.space}")
    
    parts.append(f"method-{config.method}")
    parts.append(f"analysis-{config.analysis_name}")
    parts.append(suffix)
    
    filename = "_".join(parts) + extension
    
    return group_dir / filename
