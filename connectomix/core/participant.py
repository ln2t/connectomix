"""Participant-level pipeline orchestration.

This module orchestrates the complete participant-level analysis workflow,
coordinating file discovery, preprocessing, and connectivity computation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from connectomix.config.defaults import ParticipantConfig
from connectomix.config.loader import save_config
from connectomix.io.bids import create_bids_layout, build_bids_path, query_participant_files
from connectomix.io.paths import create_dataset_description
from connectomix.io.readers import load_seeds_file
from connectomix.preprocessing.resampling import (
    check_geometric_consistency,
    resample_to_reference,
    save_geometry_info,
)
from connectomix.preprocessing.denoising import denoise_image
from connectomix.preprocessing.canica import run_canica_atlas
from connectomix.connectivity.seed_to_voxel import compute_seed_to_voxel
from connectomix.connectivity.roi_to_voxel import compute_roi_to_voxel
from connectomix.connectivity.seed_to_seed import compute_seed_to_seed
from connectomix.connectivity.roi_to_roi import compute_roi_to_roi
from connectomix.utils.exceptions import BIDSError, ConnectomixError
from connectomix.utils.logging import timer, log_section
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
        ...     atlas="schaefer2018_100"
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
            
            with timer(logger, f"  Subject {file_entities.get('sub', 'unknown')}"):
                # --- Resample if needed ---
                if not is_consistent:
                    resampled_path = _get_output_path(
                        output_dir, file_entities, "resampled", "bold", ".nii.gz"
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
                
                # --- Denoise ---
                denoised_path = _get_output_path(
                    output_dir, file_entities, "denoised", "bold", ".nii.gz"
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
                
                # --- Compute connectivity ---
                connectivity_paths = _compute_connectivity(
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
                outputs['connectivity'].extend(connectivity_paths)
        
        # === Summary ===
        log_section(logger, "Summary")
        
        logger.info(f"Processed {n_files} functional file(s)")
        logger.info(f"Generated {len(outputs['connectivity'])} connectivity output(s)")
        logger.info(f"Outputs saved to: {output_dir}")
    
    return outputs


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
) -> Path:
    """Build output path with BIDS naming."""
    # Start with subject directory
    sub_dir = output_dir / f"sub-{entities.get('sub', 'unknown')}"
    
    if entities.get('ses'):
        sub_dir = sub_dir / f"ses-{entities['ses']}"
    
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    parts = []
    
    entity_order = ['sub', 'ses', 'task', 'run', 'space']
    for key in entity_order:
        if key in entities:
            parts.append(f"{key}-{entities[key]}")
    
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
        
        # Extract number of parcels
        if "_" in atlas_name:
            n_rois = int(atlas_name.split("_")[1])
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
            f"Available atlases: schaefer2018_100, schaefer2018_200, aal, harvardoxford"
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
) -> List[Path]:
    """Compute connectivity using the specified method."""
    output_paths = []
    
    # Add method to entities
    file_entities = dict(file_entities)
    file_entities['method'] = method
    
    if method == "seedToVoxel":
        # Compute for each seed
        for i, (coord, name) in enumerate(zip(seeds_coords, seeds_names)):
            file_entities['seed'] = name
            
            output_path = _get_output_path(
                output_dir, file_entities, name, "effectSize", ".nii.gz"
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
                output_dir, file_entities, roi_name, "effectSize", ".nii.gz"
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
            output_dir, file_entities, "seeds", "correlation", ".npy"
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
        
        output_path = _get_output_path(
            output_dir, file_entities, config.atlas, "correlation", ".npy"
        )
        
        compute_roi_to_roi(
            func_img=denoised_img,
            atlas_img=atlas_img,
            atlas_name=config.atlas,
            output_path=output_path,
            logger=logger,
            kind=config.connectivity_kind,
            roi_names=atlas_labels,
        )
        output_paths.append(output_path)
    
    else:
        raise ConnectomixError(f"Unknown method: {method}")
    
    return output_paths
