"""BIDS layout creation and path building."""

from bids import BIDSLayout
from pathlib import Path
from typing import Dict, Optional, Any
import logging

from connectomix.io.paths import validate_bids_dir, validate_derivatives_dir


def create_bids_layout(
    bids_dir: Path,
    derivatives: Optional[Dict[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> BIDSLayout:
    """Create BIDS layout with derivatives.
    
    Args:
        bids_dir: Path to BIDS dataset root
        derivatives: Dictionary mapping derivative names to paths
        logger: Optional logger instance
    
    Returns:
        BIDSLayout instance
    
    Raises:
        BIDSError: If BIDS directory or derivatives are invalid
    """
    # Validate BIDS directory
    validate_bids_dir(bids_dir)
    
    if logger:
        logger.info(f"Creating BIDS layout for {bids_dir}")
    
    # Set up derivatives paths as a list (pybids expects list, not dict)
    derivatives_list = []
    
    if derivatives:
        for name, path in derivatives.items():
            validate_derivatives_dir(path, name)
            derivatives_list.append(str(path))
            if logger:
                logger.debug(f"  Adding {name} derivatives: {path}")
    else:
        # Try to find fmriprep in standard location
        default_fmriprep = bids_dir / "derivatives" / "fmriprep"
        if default_fmriprep.exists():
            derivatives_list.append(str(default_fmriprep))
            if logger:
                logger.debug(f"  Found fmriprep at default location: {default_fmriprep}")
    
    # Create layout - derivatives should be a list of paths or False
    layout = BIDSLayout(
        str(bids_dir),
        derivatives=derivatives_list if derivatives_list else False,
        validate=False  # Skip validation for speed
    )
    
    if logger:
        n_subjects = len(layout.get_subjects())
        logger.info(f"Found {n_subjects} subject(s) in dataset")
    
    return layout


def build_bids_path(
    output_dir: Path,
    entities: Dict[str, Any],
    suffix: str,
    extension: str,
    level: str = "participant"
) -> Path:
    """Build BIDS-compliant output path.
    
    Args:
        output_dir: Output directory root
        entities: Dictionary of BIDS entities
        suffix: File suffix (e.g., "bold", "effectSize")
        extension: File extension (e.g., ".nii.gz", ".json")
        level: Analysis level ("participant" or "group")
    
    Returns:
        Complete BIDS-compliant path
    
    Example:
        >>> entities = {
        ...     'subject': '01',
        ...     'session': '1',
        ...     'task': 'rest',
        ...     'space': 'MNI152NLin2009cAsym',
        ...     'method': 'seedToVoxel',
        ...     'seed': 'PCC'
        ... }
        >>> path = build_bids_path(
        ...     Path('/output'),
        ...     entities,
        ...     'effectSize',
        ...     '.nii.gz'
        ... )
        >>> # Returns: /output/sub-01/ses-1/sub-01_ses-1_task-rest_space-MNI_method-seedToVoxel_seed-PCC_effectSize.nii.gz
    """
    # Start with output directory
    if level == "participant":
        path = output_dir / f"sub-{entities['subject']}"
        if 'session' in entities and entities['session']:
            path = path / f"ses-{entities['session']}"
    else:  # group
        path = output_dir / "group"
        
        if 'method' in entities and entities['method']:
            path = path / entities['method']
        
        if 'analysis' in entities and entities['analysis']:
            path = path / entities['analysis']
        
        if 'session' in entities and entities['session']:
            path = path / f"ses-{entities['session']}"
    
    # Build filename from entities
    parts = []
    
    # Define entity order (following BIDS specification)
    entity_order = [
        'subject', 'session', 'task', 'acquisition', 'ceagent',
        'reconstruction', 'direction', 'run', 'echo', 'space',
        'denoise', 'condition', 'method', 'seed', 'roi', 'data', 'atlas', 'analysis',
        'desc', 'threshold', 'stat'
    ]
    
    for entity_name in entity_order:
        if entity_name in entities and entities[entity_name] is not None:
            value = entities[entity_name]
            # Handle lists (convert to hyphen-separated string)
            if isinstance(value, list):
                value = '-'.join(str(v) for v in value)
            parts.append(f"{entity_name}-{value}")
    
    # Add suffix
    parts.append(suffix)
    
    # Create filename
    filename = "_".join(parts) + extension
    
    # Ensure directory exists
    path.mkdir(parents=True, exist_ok=True)
    
    return path / filename


def query_participant_files(
    layout: BIDSLayout,
    entities: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, list]:
    """Query fMRIPrep files for participant-level analysis.
    
    Args:
        layout: BIDSLayout instance
        entities: Dictionary of BIDS entities for filtering
        logger: Optional logger instance
    
    Returns:
        Dictionary with keys 'func', 'json', 'confounds' containing file lists
    
    Raises:
        ValueError: If file counts don't match
    """
    # Build query parameters
    query_params = {
        'extension': 'nii.gz',
        'suffix': 'bold',
        'desc': 'preproc',
        'scope': 'derivatives',
        'invalid_filters': 'allow',  # Allow derivative-specific entities like 'desc'
    }
    
    # Add optional filters
    if entities.get('subject'):
        query_params['subject'] = entities['subject']
    if entities.get('session'):
        query_params['session'] = entities['session']
    if entities.get('task'):
        query_params['task'] = entities['task']
    if entities.get('run'):
        query_params['run'] = entities['run']
    if entities.get('space'):
        query_params['space'] = entities['space']
    
    # Query functional files
    func_files = layout.get(**query_params)
    
    if logger:
        logger.info(f"Found {len(func_files)} functional image(s)")
    
    if len(func_files) == 0:
        # Attempt a relaxed search (without 'desc') to provide helpful
        # debugging information to the user about available candidate files.
        relaxed_q = {k: v for k, v in query_params.items() if k != "desc"}
        try:
            candidate_files = layout.get(**relaxed_q)
        except Exception:
            candidate_files = []

        candidate_paths = [f.path for f in candidate_files][:10]

        raise ValueError(
            f"No functional images found matching criteria:\n"
            f"  {query_params}\n"
            f"Example candidate files (relaxed search without 'desc'): {candidate_paths}\n"
            f"Please check your BIDS entities and fMRIPrep outputs. If fMRIPrep "
            f"is in a non-standard location, specify it with --derivatives "
            f"(e.g., --derivatives fmriprep=/path/to/fmriprep)."
        )
    
    # Get corresponding JSON and confounds files
    json_files = []
    confounds_files = []
    
    for func_file in func_files:
        # Get JSON sidecar
        json_file = layout.get_metadata(func_file.path)
        json_files.append(func_file.path.replace('.nii.gz', '.json'))
        
        # Get confounds file
        confounds_query = {
            'subject': func_file.entities['subject'],
            'suffix': 'timeseries',
            'desc': 'confounds',
            'extension': 'tsv',
            'scope': 'derivatives',
            'invalid_filters': 'allow',  # Allow derivative-specific entities
        }
        
        # Add optional entities
        for entity in ['session', 'task', 'run']:
            if entity in func_file.entities:
                confounds_query[entity] = func_file.entities[entity]
        
        confounds = layout.get(**confounds_query)
        
        if len(confounds) == 0:
            raise ValueError(
                f"No confounds file found for {func_file.filename}"
            )
        
        confounds_files.append(confounds[0].path)
    
    # Validate counts match
    if len(func_files) != len(json_files) or len(func_files) != len(confounds_files):
        raise ValueError(
            f"File count mismatch:\n"
            f"  Functional: {len(func_files)}\n"
            f"  JSON: {len(json_files)}\n"
            f"  Confounds: {len(confounds_files)}"
        )
    
    return {
        'func': [f.path for f in func_files],
        'json': json_files,
        'confounds': confounds_files
    }
