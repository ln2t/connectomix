"""Group-level pipeline orchestration.

This module orchestrates the complete group-level analysis workflow using
tangent space connectivity analysis.

The tangent space approach:
1. Loads participant-level ROI time series
2. Computes a group geometric mean covariance matrix
3. Projects individual connectivity into tangent space
4. Outputs group mean and individual deviation matrices
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from connectomix.config.defaults import GroupConfig
from connectomix.connectivity.group_connectivity import (
    discover_participant_timeseries,
    load_timeseries,
    compute_tangent_connectivity,
)
from connectomix.io.writers import save_matrix_with_sidecar
from connectomix.utils.exceptions import ConnectomixError


def run_group_pipeline(
    bids_dir: Path,
    output_dir: Path,
    config: GroupConfig,
    derivatives: Optional[Dict[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """Run the group-level tangent space connectivity pipeline.
    
    This pipeline:
    1. Discovers participant-level time series files
    2. Loads and validates time series data
    3. Computes tangent space connectivity (group mean + individual deviations)
    4. Saves outputs with BIDS-like naming
    5. Generates a group-level HTML report
    
    Args:
        bids_dir: Path to BIDS dataset root.
        output_dir: Path for output derivatives.
        config: GroupConfig instance with analysis parameters.
        derivatives: Optional dict mapping derivative names to paths.
            Should contain 'connectomix' key pointing to participant outputs.
        logger: Logger instance. If None, creates one.
    
    Returns:
        List of paths to output files.
    
    Raises:
        ConnectomixError: If no participant data found or computation fails.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("CONNECTOMIX - Group-Level Analysis")
    logger.info("=" * 60)
    logger.info("")
    
    # Validate configuration
    config.validate()
    
    # Determine participant derivatives path
    if config.participant_derivatives:
        participant_dir = Path(config.participant_derivatives)
    elif derivatives and 'connectomix' in derivatives:
        participant_dir = derivatives['connectomix']
    else:
        # Default: look in output_dir (assumes participant-level already run)
        participant_dir = output_dir
    
    if not participant_dir.exists():
        raise ConnectomixError(
            f"Participant derivatives directory not found: {participant_dir}"
        )
    
    logger.info(f"Participant derivatives: {participant_dir}")
    logger.info(f"Atlas: {config.atlas}")
    logger.info(f"Method: {config.method}")
    
    # Get task filter
    task = config.tasks[0] if config.tasks else None
    session = config.sessions[0] if config.sessions else None
    
    if task:
        logger.info(f"Task filter: {task}")
    if session:
        logger.info(f"Session filter: {session}")
    
    logger.info("")
    
    # Step 1: Discover participant time series files
    logger.info("Step 1: Discovering participant time series files...")
    timeseries_files = discover_participant_timeseries(
        derivatives_dir=participant_dir,
        atlas=config.atlas,
        task=task,
        session=session,
        subjects=config.subjects,
    )
    
    # Step 2: Load time series data
    logger.info("Step 2: Loading time series data...")
    subject_ids, timeseries_list = load_timeseries(timeseries_files)
    
    logger.info(f"  Loaded {len(subject_ids)} subjects")
    logger.info(f"  Subjects: {', '.join(subject_ids)}")
    logger.info("")
    
    # Step 3: Compute tangent space connectivity
    logger.info("Step 3: Computing tangent space connectivity...")
    results = compute_tangent_connectivity(
        timeseries_list=timeseries_list,
        subject_ids=subject_ids,
        vectorize=config.vectorize,
    )
    
    logger.info(f"  Group mean matrix: {results['n_regions']} x {results['n_regions']}")
    logger.info(f"  Individual tangent matrices: {results['n_subjects']}")
    logger.info("")
    
    # Step 4: Save outputs
    logger.info("Step 4: Saving outputs...")
    output_paths = _save_group_outputs(
        results=results,
        output_dir=output_dir,
        config=config,
        task=task,
        session=session,
        logger=logger,
    )
    
    # Step 5: Generate report
    logger.info("Step 5: Generating HTML report...")
    report_path = _generate_group_report(
        results=results,
        config=config,
        output_dir=output_dir,
        task=task,
        session=session,
        logger=logger,
    )
    output_paths.append(report_path)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Group analysis complete!")
    logger.info(f"  Subjects processed: {results['n_subjects']}")
    logger.info(f"  ROI regions: {results['n_regions']}")
    logger.info(f"  Output files: {len(output_paths)}")
    logger.info(f"  Report: {report_path.name}")
    logger.info("=" * 60)
    
    return output_paths


def _generate_group_report(
    results: Dict,
    config: GroupConfig,
    output_dir: Path,
    task: Optional[str],
    session: Optional[str],
    logger: logging.Logger,
) -> Path:
    """Generate HTML report for group analysis."""
    from connectomix.utils.reports import GroupReportGenerator
    
    group_dir = output_dir / "group"
    
    report = GroupReportGenerator(
        results=results,
        config=config,
        output_dir=group_dir,
        task=task,
        session=session,
    )
    
    return report.generate()


def _save_group_outputs(
    results: Dict,
    output_dir: Path,
    config: GroupConfig,
    task: Optional[str],
    session: Optional[str],
    logger: logging.Logger,
) -> List[Path]:
    """Save group analysis outputs with BIDS-like naming.
    
    Output structure:
        group/
            task-X_atlas-Y_desc-groupMean_connectivity.npy
            task-X_atlas-Y_desc-whitening_matrix.npy
            sub-01/
                task-X_atlas-Y_desc-tangent_connectivity.npy
    """
    output_paths = []
    
    # Create group output directory
    group_dir = output_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    
    # Build base filename parts
    base_parts = []
    if task:
        base_parts.append(f"task-{task}")
    if session:
        base_parts.append(f"ses-{session}")
    base_parts.append(f"atlas-{config.atlas}")
    if config.label:
        base_parts.append(f"label-{config.label}")
    
    base_name = "_".join(base_parts)
    
    # Save group mean connectivity
    mean_filename = f"{base_name}_desc-groupMean_connectivity.npy"
    mean_path = group_dir / mean_filename
    
    mean_metadata = {
        'Description': 'Group mean connectivity matrix (geometric mean of covariances)',
        'AnalysisMethod': 'tangent',
        'Atlas': config.atlas,
        'NumberOfSubjects': results['n_subjects'],
        'NumberOfRegions': results['n_regions'],
        'SubjectIDs': results['subject_ids'],
        'Task': task,
        'Session': session,
        'GeneratedBy': 'connectomix',
        'GeneratedAt': datetime.now().isoformat(),
    }
    
    save_matrix_with_sidecar(results['group_mean'], mean_path, mean_metadata)
    output_paths.append(mean_path)
    logger.info(f"  Saved group mean: {mean_path.name}")
    
    # Save whitening matrix
    whitening_filename = f"{base_name}_desc-whitening_matrix.npy"
    whitening_path = group_dir / whitening_filename
    
    whitening_metadata = {
        'Description': 'Whitening matrix for tangent space projection',
        'AnalysisMethod': 'tangent',
        'Shape': list(results['whitening'].shape),
    }
    
    save_matrix_with_sidecar(results['whitening'], whitening_path, whitening_metadata)
    output_paths.append(whitening_path)
    logger.info(f"  Saved whitening matrix: {whitening_path.name}")
    
    # Save individual tangent matrices
    for sub_id, tangent_matrix in results['tangent_matrices'].items():
        sub_dir = group_dir / f"sub-{sub_id}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        tangent_filename = f"sub-{sub_id}_{base_name}_desc-tangent_connectivity.npy"
        tangent_path = sub_dir / tangent_filename
        
        tangent_metadata = {
            'Description': 'Individual tangent space deviation from group mean',
            'AnalysisMethod': 'tangent',
            'SubjectID': sub_id,
            'Atlas': config.atlas,
            'NumberOfRegions': results['n_regions'],
        }
        
        save_matrix_with_sidecar(tangent_matrix, tangent_path, tangent_metadata)
        output_paths.append(tangent_path)
    
    logger.info(f"  Saved {len(results['tangent_matrices'])} individual tangent matrices")
    
    # Save analysis summary JSON
    summary_path = group_dir / f"{base_name}_analysis-summary.json"
    summary = {
        'analysis_type': 'tangent_connectivity',
        'atlas': config.atlas,
        'task': task,
        'session': session,
        'n_subjects': results['n_subjects'],
        'n_regions': results['n_regions'],
        'subject_ids': results['subject_ids'],
        'output_files': [str(p.relative_to(output_dir)) for p in output_paths],
        'generated_at': datetime.now().isoformat(),
        'connectomix_version': '3.0.0',
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    output_paths.append(summary_path)
    logger.info(f"  Saved analysis summary: {summary_path.name}")
    
    return output_paths
