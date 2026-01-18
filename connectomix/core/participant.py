"""Participant-level pipeline orchestration.

This module orchestrates the complete participant-level analysis workflow,
coordinating file discovery, preprocessing, and connectivity computation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from connectomix.config.defaults import ParticipantConfig
from connectomix.config.loader import save_config
from connectomix.io.bids import create_bids_layout, build_bids_path, query_participant_files
from connectomix.io.paths import create_dataset_description
from connectomix.io.readers import load_seeds_file, get_repetition_time
from connectomix.preprocessing.resampling import (
    check_geometric_consistency,
    resample_to_reference,
    save_geometry_info,
)
from connectomix.preprocessing.denoising import denoise_image
from connectomix.preprocessing.canica import run_canica_atlas
from connectomix.preprocessing.censoring import (
    TemporalCensor,
    load_events_file,
    find_events_file,
)
from connectomix.connectivity.seed_to_voxel import compute_seed_to_voxel
from connectomix.connectivity.roi_to_voxel import compute_roi_to_voxel
from connectomix.connectivity.seed_to_seed import compute_seed_to_seed
from connectomix.connectivity.roi_to_roi import compute_roi_to_roi, compute_roi_to_roi_all_measures
from connectomix.utils.exceptions import BIDSError, ConnectomixError, PreprocessingError
from connectomix.utils.logging import timer, log_section
from connectomix.utils.reports import ParticipantReportGenerator
from connectomix.utils.matrix import CONNECTIVITY_KINDS
from connectomix.core.version import __version__


def run_participant_pipeline(
    bids_dir: Path,
    output_dir: Path,
    config: ParticipantConfig,
    derivatives: Optional[Dict[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Path]]:
    """Run the complete participant-level analysis pipeline.
    
    This function orchestrates:
    1. BIDS layout creation and file discovery
    2. Geometric consistency checking (across ALL subjects)
    3. Resampling if needed
    4. Denoising (confound regression + temporal filtering)
    5. CanICA atlas generation (if method=roiToRoi and atlas=canica)
    6. Connectivity computation (one of four methods)
    7. Output saving with BIDS-compliant names and JSON sidecars
    
    Args:
        bids_dir: Path to BIDS dataset root.
        output_dir: Path for output derivatives.
        config: ParticipantConfig instance with analysis parameters.
        derivatives: Optional dict mapping derivative names to paths.
            If None, looks for fmriprep in standard location.
        logger: Logger instance. If None, creates one.
    
    Returns:
        Dictionary with keys mapping to lists of output file paths:
            - 'connectivity': List of connectivity output files
            - 'denoised': List of denoised functional images
            - 'resampled': List of resampled images (if resampling needed)
    
    Raises:
        BIDSError: If BIDS dataset or derivatives are invalid.
        ConnectomixError: If pipeline fails.
    
    Example:
        >>> from connectomix.config.defaults import ParticipantConfig
        >>> config = ParticipantConfig(
        ...     subject=["01", "02"],
        ...     tasks=["rest"],
        ...     method="roiToRoi",
        ...     atlas="schaefer2018n100"
        ... )
        >>> outputs = run_participant_pipeline(
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
        'connectivity': [],
        'denoised': [],
        'resampled': [],
    }
    
    with timer(logger, "Participant-level analysis"):
        # === Step 1: Setup ===
        log_section(logger, "Setup")
        
        # Create output directory and dataset description
        output_dir.mkdir(parents=True, exist_ok=True)
        create_dataset_description(output_dir, __version__)
        
        # Backup configuration
        config_backup_dir = output_dir / "config" / "backups"
        save_config(config, config_backup_dir)
        
        # Create BIDS layout
        layout = create_bids_layout(bids_dir, derivatives, logger)
        
        # Validate requested participant labels exist in dataset
        available_subjects = set(layout.get_subjects())
        if config.subject:
            requested_subjects = config.subject if isinstance(config.subject, list) else [config.subject]
            missing_subjects = [s for s in requested_subjects if s not in available_subjects]
            if missing_subjects:
                # Provide helpful suggestions for similar subject IDs
                suggestions = []
                for missing in missing_subjects:
                    similar = [s for s in available_subjects if missing.lower() in s.lower() or s.lower() in missing.lower()]
                    if similar:
                        suggestions.append(f"  '{missing}' - did you mean: {', '.join(sorted(similar)[:5])}?")
                    else:
                        suggestions.append(f"  '{missing}' - no similar IDs found")
                
                raise BIDSError(
                    f"Requested participant(s) not found in dataset:\n"
                    + "\n".join(suggestions) + "\n\n"
                    f"Available subjects ({len(available_subjects)} total): "
                    f"{', '.join(sorted(available_subjects)[:10])}"
                    + (f"... and {len(available_subjects) - 10} more" if len(available_subjects) > 10 else "")
                    + "\n\nNote: Specify participant labels WITHOUT the 'sub-' prefix."
                )
        
        # === Step 2: Query files ===
        log_section(logger, "File Discovery")
        
        # Build entity filter from config
        entities = _build_entity_filter(config)
        
        # Query fMRIPrep files
        files = query_participant_files(layout, entities, logger)
        
        n_files = len(files['func'])
        logger.info(f"Found {n_files} functional file(s) to process")
        
        if n_files == 0:
            raise BIDSError(
                "No functional files found matching the specified criteria.\n"
                "Please check your BIDS entities and fMRIPrep outputs."
            )
        
        # === Step 3: Geometric consistency check ===
        log_section(logger, "Geometric Consistency")
        
        # Get ALL functional files for geometry check (not just selected)
        all_func_files = _get_all_functional_files(layout, entities, logger)
        
        is_consistent, geometries = check_geometric_consistency(
            all_func_files, logger
        )
        
        # Determine reference image
        if config.reference_functional_file == "first_functional_file":
            reference_img = nib.load(files['func'][0])
            logger.info(f"Using first functional file as reference")
        else:
            reference_img = nib.load(config.reference_functional_file)
            logger.info(f"Using custom reference: {config.reference_functional_file}")
        
        # === Step 4: Generate CanICA atlas if needed ===
        atlas_img = None
        atlas_labels = None
        
        if config.method == "roiToRoi" and config.atlas == "canica":
            log_section(logger, "CanICA Atlas Generation")
            
            # Use denoised images for ICA
            canica_output = output_dir / "atlas" / "canica_atlas.nii.gz"
            
            atlas_img, atlas_labels = run_canica_atlas(
                func_imgs=[nib.load(f) for f in files['func']],
                output_path=canica_output,
                n_components=config.n_components,
                threshold=config.canica_threshold,
                min_region_size=config.canica_min_region_size,
                logger=logger,
            )
            
            logger.info(f"Generated CanICA atlas with {len(atlas_labels)} regions")
        
        elif config.method == "roiToRoi":
            # Load standard atlas
            atlas_img, atlas_labels = _load_standard_atlas(config.atlas, logger)
        
        # === Step 5: Load seeds if needed ===
        seeds_coords = None
        seeds_names = None
        
        if config.method in ["seedToVoxel", "seedToSeed"]:
            if config.seeds_file is None:
                raise ConnectomixError(
                    f"seeds_file is required for method '{config.method}'"
                )
            
            seeds_names, seeds_coords = load_seeds_file(config.seeds_file)
            logger.info(f"Loaded {len(seeds_names)} seed(s) from {config.seeds_file}")
        
        # === Step 6: Process each functional file ===
        log_section(logger, "Processing")
        
        for i, (func_path, confounds_path) in enumerate(
            zip(files['func'], files['confounds'])
        ):
            func_path = Path(func_path)
            confounds_path = Path(confounds_path)
            
            logger.info(f"Processing file {i+1}/{n_files}: {func_path.name}")
            
            # Extract entities from filename
            file_entities = _extract_entities_from_path(func_path)
            
            # Track all connectivity outputs for this file
            connectivity_paths = []
            
            with timer(logger, f"  Subject {file_entities.get('sub', 'unknown')}"):
                # --- Resample if needed ---
                if not is_consistent:
                    resampled_path = _get_output_path(
                        output_dir, file_entities, "resampled", "bold", ".nii.gz",
                        label=config.label, subfolder="func"
                    )
                    
                    func_img = resample_to_reference(
                        func_path,
                        reference_img,
                        resampled_path,
                        logger,
                    )
                    outputs['resampled'].append(resampled_path)
                    
                    # Save geometry info with source JSON for TR metadata
                    geometry_path = resampled_path.with_suffix('').with_suffix('.json')
                    if not geometry_path.exists():
                        source_json = func_path.with_suffix('').with_suffix('.json')
                        save_geometry_info(func_img, geometry_path, source_json=source_json)
                    
                    # Use resampled path for denoising
                    input_for_denoise = resampled_path
                else:
                    func_img = nib.load(func_path)
                    input_for_denoise = func_path
                
                # Load the input image for denoising (for histogram comparison)
                input_img_for_histogram = nib.load(input_for_denoise)
                
                # --- Denoise ---
                denoised_path = _get_output_path(
                    output_dir, file_entities, "denoised", "bold", ".nii.gz",
                    label=config.label, subfolder="func"
                )
                
                denoise_image(
                    input_for_denoise,
                    confounds_path,
                    config.confounds,
                    config.high_pass,
                    config.low_pass,
                    denoised_path,
                    logger,
                    overwrite=config.overwrite_denoised_files,
                )
                outputs['denoised'].append(denoised_path)
                
                # Load denoised image for connectivity
                denoised_img = nib.load(denoised_path)
                
                # Compute denoising histogram data for QA report
                from connectomix.preprocessing.denoising import compute_denoising_histogram_data
                denoising_histogram_data = compute_denoising_histogram_data(
                    original_img=input_img_for_histogram,
                    denoised_img=denoised_img,
                )
                
                # --- Apply temporal censoring if enabled ---
                censor = None
                censoring_summary = None
                
                if config.temporal_censoring.enabled:
                    censor, censoring_summary = _apply_temporal_censoring(
                        denoised_img=denoised_img,
                        func_path=func_path,
                        confounds_path=confounds_path,
                        layout=layout,
                        config=config,
                        logger=logger,
                    )
                    
                    # Add censoring entity to filenames if motion censoring is used
                    censoring_entity = censor.get_censoring_entity()
                    if censoring_entity:
                        file_entities['censoring'] = censoring_entity
                
                # --- Compute connectivity ---
                # If condition selection is enabled, compute connectivity for each condition
                connectivity_matrices_for_report = {}  # Store all matrices for report
                roi_names_for_report = None
                
                if censor and censor.condition_masks:
                    # Compute connectivity for each condition separately
                    for condition_name, condition_mask in censor.condition_masks.items():
                        n_vols = np.sum(condition_mask)
                        
                        # Warn if very few volumes, but still compute connectivity
                        # (validation warnings are already issued in censoring.py)
                        if n_vols < config.temporal_censoring.min_volumes_retained:
                            logger.warning(
                                f"⚠️ Condition '{condition_name}' has only {n_vols} volumes "
                                f"(recommended minimum: {config.temporal_censoring.min_volumes_retained}). "
                                f"Connectivity will be computed but results may be unreliable."
                            )
                        
                        # Skip only if no volumes at all
                        if n_vols == 0:
                            logger.warning(f"Skipping condition '{condition_name}': no volumes to analyze")
                            continue
                        
                        logger.info(f"Computing connectivity for condition '{condition_name}' ({n_vols} volumes)")
                        
                        # Apply censoring to get condition-specific image
                        condition_img = censor.apply_to_image(denoised_img, condition=condition_name)
                        
                        # Add condition to entities for output naming
                        condition_entities = dict(file_entities)
                        condition_entities['condition'] = condition_name
                        
                        connectivity_paths_cond, conn_matrices, roi_names = _compute_connectivity(
                            denoised_img=condition_img,
                            method=config.method,
                            output_dir=output_dir,
                            file_entities=condition_entities,
                            config=config,
                            seeds_coords=seeds_coords,
                            seeds_names=seeds_names,
                            atlas_img=atlas_img,
                            atlas_labels=atlas_labels,
                            logger=logger,
                        )
                        outputs['connectivity'].extend(connectivity_paths_cond)
                        connectivity_paths.extend(connectivity_paths_cond)
                        
                        # Store matrices for report (per condition)
                        if conn_matrices:
                            for kind, matrix in conn_matrices.items():
                                connectivity_matrices_for_report[f"{condition_name}_{kind}"] = matrix
                            roi_names_for_report = roi_names
                else:
                    # No condition selection - compute connectivity on full (possibly censored) data
                    if censor:
                        # Apply global censoring (motion, initial drop)
                        denoised_img = censor.apply_to_image(denoised_img)
                    
                    connectivity_paths_single, conn_matrices, roi_names = _compute_connectivity(
                        denoised_img=denoised_img,
                        method=config.method,
                        output_dir=output_dir,
                        file_entities=file_entities,
                        config=config,
                        seeds_coords=seeds_coords,
                        seeds_names=seeds_names,
                        atlas_img=atlas_img,
                        atlas_labels=atlas_labels,
                        logger=logger,
                    )
                    outputs['connectivity'].extend(connectivity_paths_single)
                    connectivity_paths = connectivity_paths_single
                    
                    # Store matrices for report
                    if conn_matrices:
                        connectivity_matrices_for_report = conn_matrices
                        roi_names_for_report = roi_names
                
                # --- Generate HTML Report ---
                # Extract condition name(s) from censoring summary for report filename
                condition_for_report = None
                if censoring_summary and censoring_summary.get('conditions'):
                    condition_names = list(censoring_summary['conditions'].keys())
                    if len(condition_names) == 1:
                        condition_for_report = condition_names[0]
                    elif len(condition_names) > 1:
                        # Multiple conditions - join with "+"
                        condition_for_report = '+'.join(sorted(condition_names))
                
                # Get censoring entity from file_entities (set earlier if FD threshold used)
                censoring_for_report = file_entities.get('censoring')
                
                _generate_participant_report(
                    file_entities=file_entities,
                    config=config,
                    output_dir=output_dir,
                    confounds_path=confounds_path,
                    connectivity_paths=connectivity_paths,
                    atlas_labels=atlas_labels,
                    logger=logger,
                    censoring_summary=censoring_summary,
                    condition=condition_for_report,
                    censoring=censoring_for_report,
                    connectivity_matrices=connectivity_matrices_for_report,
                    roi_names=roi_names_for_report,
                    denoising_histogram_data=denoising_histogram_data,
                )
        
        # === Summary ===
        log_section(logger, "Summary")
        
        logger.info(f"Processed {n_files} functional file(s)")
        logger.info(f"Generated {len(outputs['connectivity'])} connectivity output(s)")
        logger.info(f"Outputs saved to: {output_dir}")
    
    return outputs


def _build_cli_command(
    config: ParticipantConfig,
    file_entities: Dict[str, str],
) -> str:
    """Build a generic CLI command from the configuration for reproducibility.
    
    Paths are replaced with placeholders to avoid exposing user-specific paths.
    
    Parameters
    ----------
    config : ParticipantConfig
        Configuration object.
    file_entities : dict
        BIDS entities for the current file.
    
    Returns
    -------
    str
        CLI command with generic path placeholders.
    """
    parts = ["connectomix <rawdata> <derivatives> participant"]
    
    # Add participant label (check both 'subject' and 'sub' keys)
    subject = file_entities.get('subject') or file_entities.get('sub')
    if subject:
        parts.append(f"-p {subject}")
    
    # Add task
    task = file_entities.get('task')
    if task:
        parts.append(f"-t {task}")
    
    # Add session if present
    session = file_entities.get('session') or file_entities.get('ses')
    if session:
        parts.append(f"-s {session}")
    
    # Add run if present
    run = file_entities.get('run')
    if run:
        parts.append(f"-r {run}")
    
    # Add method
    parts.append(f"--method {config.method}")
    
    # Add atlas for ROI methods
    if config.method in ["roiToRoi", "seedToSeed"] and config.atlas:
        parts.append(f"--atlas {config.atlas}")
    
    # Add denoising options - show ALL confounds for full reproducibility
    if config.confounds:
        confounds_str = " ".join(config.confounds)
        parts.append(f"--confounds {confounds_str}")
    
    if config.high_pass:
        parts.append(f"--high-pass {config.high_pass}")
    
    if config.low_pass:
        parts.append(f"--low-pass {config.low_pass}")
    
    # Add temporal censoring options if enabled
    if config.temporal_censoring.enabled:
        tc = config.temporal_censoring
        
        if tc.drop_initial_volumes > 0:
            parts.append(f"--drop-initial {tc.drop_initial_volumes}")
        
        if tc.motion_censoring.enabled and tc.motion_censoring.fd_threshold:
            parts.append(f"--fd-threshold {tc.motion_censoring.fd_threshold}")
            extend = tc.motion_censoring.extend_before or tc.motion_censoring.extend_after
            if extend > 0:
                parts.append(f"--fd-extend {extend}")
        
        if tc.condition_selection.enabled and tc.condition_selection.conditions:
            conditions_str = " ".join(tc.condition_selection.conditions)
            parts.append(f"--conditions {conditions_str}")
    
    # Add label if set
    if config.label:
        parts.append(f"--label {config.label}")
    
    # Join with spaces - each option on same line for clean display
    return " ".join(parts)


def _generate_participant_report(
    file_entities: Dict[str, str],
    config: ParticipantConfig,
    output_dir: Path,
    confounds_path: Path,
    connectivity_paths: List[Path],
    atlas_labels: Optional[List[str]],
    logger: logging.Logger,
    censoring_summary: Optional[Dict] = None,
    condition: Optional[str] = None,
    censoring: Optional[str] = None,
    connectivity_matrices: Optional[Dict[str, np.ndarray]] = None,
    roi_names: Optional[List[str]] = None,
    denoising_histogram_data: Optional[Dict] = None,
) -> Optional[Path]:
    """Generate HTML report for a participant analysis.
    
    Parameters
    ----------
    file_entities : dict
        BIDS entities for the file.
    config : ParticipantConfig
        Configuration object.
    output_dir : Path
        Output directory.
    confounds_path : Path
        Path to confounds file.
    connectivity_paths : list
        List of connectivity output paths.
    atlas_labels : list or None
        Atlas labels for ROI-based methods.
    logger : Logger
        Logger instance.
    censoring_summary : dict or None
        Summary of temporal censoring applied.
    condition : str or None
        Condition name for output filename (when --conditions is used).
    censoring : str or None
        Censoring method entity for output filename (e.g., 'fd05').
    connectivity_matrices : dict or None
        Dictionary mapping connectivity kind to matrix (for roiToRoi method).
    roi_names : list or None
        List of ROI names for connectivity matrices.
    
    Returns
    -------
    Path or None
        Path to generated report, or None if generation failed.
    """
    try:
        log_section(logger, "Generating HTML Report")
        
        # Load confounds for denoising plots
        confounds_df = pd.read_csv(confounds_path, sep='\t')
        
        # Use the same wildcard expansion as denoising to get exact confound names
        from connectomix.io.readers import expand_confound_wildcards
        selected_confounds = expand_confound_wildcards(
            config.confounds, 
            confounds_df.columns.tolist()
        )
        
        if not selected_confounds:
            # Fall back to some common confounds
            common = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            selected_confounds = [c for c in common if c in confounds_df.columns]
        
        # Use provided connectivity matrices, or load from file for backward compatibility
        if connectivity_matrices is None:
            connectivity_matrix = None
            if config.method in ["seedToSeed", "roiToRoi"]:
                # Find matrix files
                matrix_files = [p for p in connectivity_paths if p.suffix == '.npy']
                if matrix_files:
                    connectivity_matrix = np.load(matrix_files[0])
                    
                    # Set ROI names from atlas labels
                    if config.method == "roiToRoi" and atlas_labels:
                        roi_names = atlas_labels
                    elif config.method == "seedToSeed":
                        # Load seeds file to get names
                        if config.seeds_file:
                            _, roi_names = load_seeds_file(Path(config.seeds_file))
            
            # Convert single matrix to dict format
            if connectivity_matrix is not None:
                connectivity_matrices = {'correlation': connectivity_matrix}
        
        # Build subject identifier
        subject_id = file_entities.get('sub', 'unknown')
        session = file_entities.get('ses', '')
        task = file_entities.get('task', '')
        run = file_entities.get('run', '')
        
        subject_label = f"sub-{subject_id}"
        if session:
            subject_label += f"_ses-{session}"
        if task:
            subject_label += f"_task-{task}"
        if run:
            subject_label += f"_run-{run}"
        
        # Build desc entity based on method and atlas/seeds
        if config.method in ["roiToRoi", "roiToVoxel"]:
            desc = config.atlas if config.atlas else config.method
        elif config.method in ["seedToSeed", "seedToVoxel"]:
            # Use seeds file name without extension
            if config.seeds_file:
                desc = Path(config.seeds_file).stem
            else:
                desc = config.method
        else:
            desc = config.method
        
        # Get the primary connectivity matrix for the old interface (use correlation)
        primary_matrix = None
        if connectivity_matrices:
            primary_matrix = connectivity_matrices.get('correlation')
        
        # Create report generator
        report = ParticipantReportGenerator(
            subject_id=subject_label,
            config=config,
            output_dir=output_dir,
            confounds_df=confounds_df,
            selected_confounds=selected_confounds,
            connectivity_matrix=primary_matrix,
            roi_names=roi_names,
            connectivity_paths=connectivity_paths,
            logger=logger,
            desc=desc,
            label=config.label,
            censoring_summary=censoring_summary,
            condition=condition,
            censoring=censoring,
        )
        
        # Add denoising histogram data if available
        if denoising_histogram_data is not None:
            report.add_denoising_histogram_data(denoising_histogram_data)
        
        # Build and set the command line for reproducibility
        cli_command = _build_cli_command(config, file_entities)
        report.set_command_line(cli_command)
        
        # Add all connectivity matrices to the report
        if connectivity_matrices and roi_names:
            for kind, matrix in connectivity_matrices.items():
                # Skip if this is already the primary matrix (correlation)
                if kind == 'correlation':
                    continue
                report.add_connectivity_matrix(matrix, roi_names, f"{config.atlas}_{kind}")
        
        # Generate report
        report_path = report.generate()
        
        logger.info(f"HTML report generated: {report_path}")
        
        return report_path
        
    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")
        logger.debug("Report generation error details:", exc_info=True)
        return None


def _apply_temporal_censoring(
    denoised_img: nib.Nifti1Image,
    func_path: Path,
    confounds_path: Path,
    layout: "BIDSLayout",
    config: ParticipantConfig,
    logger: logging.Logger,
) -> Tuple[TemporalCensor, Dict]:
    """Apply temporal censoring to functional data.
    
    Parameters
    ----------
    denoised_img : Nifti1Image
        Denoised functional image.
    func_path : Path
        Original functional file path (for finding events file).
    confounds_path : Path
        Path to confounds file.
    layout : BIDSLayout
        BIDS layout for finding events file.
    config : ParticipantConfig
        Configuration object.
    logger : Logger
        Logger instance.
    
    Returns
    -------
    censor : TemporalCensor
        Configured temporal censor object.
    summary : dict
        Censoring summary for reporting.
    """
    log_section(logger, "Temporal Censoring")
    
    # Get data dimensions
    n_volumes = denoised_img.shape[-1]
    
    # Get TR
    json_path = func_path.with_suffix('').with_suffix('.json')
    if json_path.exists():
        tr = get_repetition_time(json_path)
    else:
        # Try to get TR from NIfTI header
        if len(denoised_img.header.get_zooms()) > 3:
            tr = float(denoised_img.header.get_zooms()[3])
        else:
            tr = 2.0  # Default assumption
            logger.warning(f"Could not determine TR, assuming {tr}s")
    
    logger.info(f"Functional data: {n_volumes} volumes, TR={tr}s")
    
    # Create temporal censor
    censor = TemporalCensor(
        config=config.temporal_censoring,
        n_volumes=n_volumes,
        tr=tr,
        logger=logger,
    )
    
    # Apply initial drop
    if config.temporal_censoring.drop_initial_volumes > 0:
        censor.apply_initial_drop()
    
    # Apply motion censoring
    if config.temporal_censoring.motion_censoring.enabled:
        confounds_df = pd.read_csv(confounds_path, sep='\t')
        censor.apply_motion_censoring(confounds_df)
    
    # Apply condition selection
    if config.temporal_censoring.condition_selection.enabled:
        cs = config.temporal_censoring.condition_selection
        
        # Find events file using BIDSLayout
        if cs.events_file == "auto":
            events_path = find_events_file(func_path, layout, logger)
            if events_path is None:
                raise PreprocessingError(
                    f"Could not find events.tsv file for {func_path.name}. "
                    f"Please specify --events-file or place events.tsv in BIDS structure."
                )
        else:
            events_path = Path(cs.events_file)
        
        # Load events
        events_df = load_events_file(events_path, logger)
        logger.info(f"Loaded events file: {events_path.name}")
        
        # Apply condition selection
        censor.apply_condition_selection(events_df)
    
    # Apply custom mask if provided
    if config.temporal_censoring.custom_mask_file:
        censor.apply_custom_mask(config.temporal_censoring.custom_mask_file)
    
    # Validate enough volumes remain
    censor.validate()
    
    # Get summary for reporting
    summary = censor.get_summary()
    
    logger.info(
        f"Censoring result: {summary['n_retained']}/{summary['n_original']} volumes retained "
        f"({summary['fraction_retained']:.1%})"
    )
    
    return censor, summary


def _build_entity_filter(config: ParticipantConfig) -> Dict[str, any]:
    """Build BIDS entity filter from config."""
    entities = {}
    
    if config.subject:
        entities['subject'] = config.subject
    if config.tasks:
        entities['task'] = config.tasks[0] if len(config.tasks) == 1 else config.tasks
    if config.sessions:
        entities['session'] = config.sessions[0] if len(config.sessions) == 1 else config.sessions
    if config.runs:
        entities['run'] = config.runs[0] if len(config.runs) == 1 else config.runs
    if config.spaces:
        entities['space'] = config.spaces[0] if len(config.spaces) == 1 else config.spaces
    
    return entities


def _get_all_functional_files(
    layout,
    entities: Dict,
    logger: logging.Logger,
) -> List[Path]:
    """Get ALL functional files for geometry check, not just selected ones."""
    # Query all fMRIPrep outputs (ignoring subject filter)
    query = {
        'extension': 'nii.gz',
        'suffix': 'bold',
        'desc': 'preproc',
        'scope': 'derivatives',
    }
    
    # Keep space filter for consistency
    if entities.get('space'):
        query['space'] = entities['space']
    
    all_files = layout.get(**query)
    
    logger.debug(f"Found {len(all_files)} total functional files for geometry check")
    
    return [Path(f.path) for f in all_files]


def _extract_entities_from_path(path: Path) -> Dict[str, str]:
    """Extract BIDS entities from filename."""
    entities = {}
    
    # Get filename without extension
    name = path.name
    for ext in ['.nii.gz', '.nii', '.json', '.tsv']:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    
    # Parse entity-value pairs
    parts = name.split('_')
    for part in parts[:-1]:  # Last part is suffix
        if '-' in part:
            key, value = part.split('-', 1)
            entities[key] = value
    
    return entities


def _get_output_path(
    output_dir: Path,
    entities: Dict[str, str],
    desc: str,
    suffix: str,
    extension: str,
    label: Optional[str] = None,
    subfolder: Optional[str] = None,
) -> Path:
    """Build output path with BIDS naming.
    
    Args:
        output_dir: Base output directory
        entities: BIDS entities dict
        desc: Description for desc- entity
        suffix: BIDS suffix (e.g., bold, correlation)
        extension: File extension (e.g., .nii.gz, .npy)
        label: Optional custom label
        subfolder: Optional subfolder within subject dir (e.g., 'func', 'connectivity_data')
    """
    # Start with subject directory
    sub_dir = output_dir / f"sub-{entities.get('sub', 'unknown')}"
    
    if entities.get('ses'):
        sub_dir = sub_dir / f"ses-{entities['ses']}"
    
    # Add subfolder if specified
    if subfolder:
        sub_dir = sub_dir / subfolder
    
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    parts = []
    
    entity_order = ['sub', 'ses', 'task', 'run', 'space', 'censoring', 'condition', 'method', 'atlas', 'seed', 'roi']
    for key in entity_order:
        if key in entities and entities[key]:
            parts.append(f"{key}-{entities[key]}")
    
    # Add custom label if provided
    if label:
        parts.append(f"label-{label}")
    
    parts.append(f"desc-{desc}")
    parts.append(suffix)
    
    filename = "_".join(parts) + extension
    
    return sub_dir / filename


def _load_standard_atlas(
    atlas_name: str,
    logger: logging.Logger,
) -> Tuple[nib.Nifti1Image, List[str]]:
    """Load a standard atlas."""
    logger.info(f"Loading atlas: {atlas_name}")
    
    if atlas_name.startswith("schaefer2018"):
        from nilearn.datasets import fetch_atlas_schaefer_2018
        
        # Extract number of parcels (format: schaefer2018n100 or schaefer2018n200)
        if "n" in atlas_name:
            n_rois = int(atlas_name.split("n")[1])
        else:
            n_rois = 100
        
        atlas = fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
        atlas_img = nib.load(atlas['maps'])
        labels = atlas['labels']
        
        # Decode labels if bytes
        if isinstance(labels[0], bytes):
            labels = [l.decode('utf-8') for l in labels]
        
        # Remove 'Background' label (index 0) as it's not extracted as an ROI
        labels = [l for l in labels if l != 'Background']
        
    elif atlas_name == "aal":
        from nilearn.datasets import fetch_atlas_aal
        
        atlas = fetch_atlas_aal()
        atlas_img = nib.load(atlas['maps'])
        labels = atlas['labels']
        # Remove 'Background' label (index 0) as it's not extracted as an ROI
        labels = [l for l in labels if l != 'Background']
        
    elif atlas_name == "harvardoxford":
        from nilearn.datasets import fetch_atlas_harvard_oxford
        
        atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = atlas['maps']
        if isinstance(atlas_img, str):
            atlas_img = nib.load(atlas_img)
        labels = atlas['labels']
        # Remove 'Background' label (index 0) as it's not extracted as an ROI
        labels = [l for l in labels if l != 'Background']
        
    else:
        raise ConnectomixError(
            f"Unknown atlas: {atlas_name}\n"
            f"Available atlases: schaefer2018n100, schaefer2018n200, aal, harvardoxford"
        )
    
    # Final sanity check: remove any 'Background' entries
    labels = [l for l in labels if l.lower() != 'background']
    
    logger.info(f"Loaded {atlas_name} atlas with {len(labels)} regions")
    
    return atlas_img, list(labels)


def _compute_connectivity(
    denoised_img: nib.Nifti1Image,
    method: str,
    output_dir: Path,
    file_entities: Dict[str, str],
    config: ParticipantConfig,
    seeds_coords: Optional[np.ndarray],
    seeds_names: Optional[List[str]],
    atlas_img: Optional[nib.Nifti1Image],
    atlas_labels: Optional[List[str]],
    logger: logging.Logger,
) -> Tuple[List[Path], Optional[Dict[str, np.ndarray]], Optional[List[str]]]:
    """Compute connectivity using the specified method.
    
    Returns:
        Tuple of:
            - output_paths: List of output file paths
            - connectivity_matrices: Dict of connectivity matrices (for roiToRoi) or None
            - roi_names: List of ROI names (for roiToRoi) or None
    """
    output_paths = []
    connectivity_matrices = None
    roi_names = None
    
    # Add method to entities
    file_entities = dict(file_entities)
    file_entities['method'] = method
    
    if method == "seedToVoxel":
        # Compute for each seed
        for i, (coord, name) in enumerate(zip(seeds_coords, seeds_names)):
            file_entities['seed'] = name
            
            output_path = _get_output_path(
                output_dir, file_entities, name, "effectSize", ".nii.gz",
                label=config.label, subfolder="connectivity_data"
            )
            
            compute_seed_to_voxel(
                func_img=denoised_img,
                seed_coords=coord,
                seed_name=name,
                output_path=output_path,
                logger=logger,
                radius=config.radius,
            )
            output_paths.append(output_path)
    
    elif method == "roiToVoxel":
        # Load ROI masks
        for roi_path in config.roi_masks:
            roi_img = nib.load(roi_path)
            roi_name = Path(roi_path).stem
            
            file_entities['roi'] = roi_name
            
            output_path = _get_output_path(
                output_dir, file_entities, roi_name, "effectSize", ".nii.gz",
                label=config.label, subfolder="connectivity_data"
            )
            
            compute_roi_to_voxel(
                func_img=denoised_img,
                roi_img=roi_img,
                roi_name=roi_name,
                output_path=output_path,
                logger=logger,
            )
            output_paths.append(output_path)
    
    elif method == "seedToSeed":
        output_path = _get_output_path(
            output_dir, file_entities, "seeds", "correlation", ".npy",
            label=config.label, subfolder="connectivity_data"
        )
        
        compute_seed_to_seed(
            func_img=denoised_img,
            seed_coords=seeds_coords,
            seed_names=seeds_names,
            output_path=output_path,
            logger=logger,
            radius=config.radius,
        )
        output_paths.append(output_path)
    
    elif method == "roiToRoi":
        file_entities['atlas'] = config.atlas
        
        # Build base filename for all outputs
        base_parts = []
        entity_order = ['sub', 'ses', 'task', 'run', 'space', 'censoring', 'condition', 'method', 'atlas']
        for key in entity_order:
            if key in file_entities and file_entities[key]:
                base_parts.append(f"{key}-{file_entities[key]}")
        if config.label:
            base_parts.append(f"label-{config.label}")
        base_filename = "_".join(base_parts)
        
        # Determine output subdirectory
        sub_dir = output_dir / f"sub-{file_entities.get('sub', 'unknown')}"
        if file_entities.get('ses'):
            sub_dir = sub_dir / f"ses-{file_entities['ses']}"
        connectivity_data_dir = sub_dir / "connectivity_data"
        
        # Compute all connectivity measures and save timeseries
        time_series, matrices, matrix_paths, ts_path, roi_names = compute_roi_to_roi_all_measures(
            func_img=denoised_img,
            atlas_img=atlas_img,
            atlas_name=config.atlas,
            output_dir=connectivity_data_dir,
            base_filename=base_filename,
            logger=logger,
            roi_names=atlas_labels,
            save_timeseries=True,
        )
        
        output_paths.extend(matrix_paths.values())
        if ts_path:
            output_paths.append(ts_path)
        
        connectivity_matrices = matrices
    
    else:
        raise ConnectomixError(f"Unknown method: {method}")
    
    return output_paths, connectivity_matrices, roi_names
