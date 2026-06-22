"""Main entry point for Connectomix."""

import sys
import logging
from pathlib import Path
from typing import Optional

from connectomix.cli import create_parser, parse_derivatives_arg
from connectomix.utils.logging import setup_logging
from connectomix.config.defaults import (
    ParticipantConfig,
    GroupConfig,
)
from connectomix.config.loader import load_config_file, config_from_dict
from connectomix.core.participant import run_participant_pipeline
from connectomix.core.group import run_group_pipeline
from connectomix.core.version import __version__


def main():
    """Main entry point for Connectomix.
    
    Parses command-line arguments and runs the appropriate pipeline
    (participant-level or group-level analysis).
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Print header
    logger.info("=" * 60)
    logger.info(f"Connectomix v{__version__}")
    logger.info(f"Analysis level: {args.analysis_level}")
    logger.info("=" * 60)
    
    try:
        # Parse derivatives argument
        derivatives_dict = parse_derivatives_arg(args.derivatives)
        
        # Run appropriate pipeline
        if args.analysis_level == "participant":
            # Load or create participant config
            if args.config:
                logger.info(f"Loading configuration from: {args.config}")
                config_dict = load_config_file(args.config)
                config = config_from_dict(config_dict, ParticipantConfig)
            else:
                logger.info("Using default configuration")
                config = ParticipantConfig()
            
            # Get participant labels to process (convert to list if needed)
            participant_labels = args.participant_label if args.participant_label else [None]
            
            # Get conditions to process (convert to list if needed)
            # If multiple conditions provided, each runs as independent analysis
            has_conditions = hasattr(args, 'conditions') and args.conditions
            conditions_to_loop = args.conditions if has_conditions else [None]
            
            # Loop over each participant label
            for participant_label in participant_labels:
                # Loop over each condition (if provided)
                for condition in conditions_to_loop:
                    # Create fresh config for each participant/condition combination
                    if args.config:
                        config_dict = load_config_file(args.config)
                        config = config_from_dict(config_dict, ParticipantConfig)
                    else:
                        config = ParticipantConfig()
                    
                    # Override config with CLI arguments
                    if participant_label:
                        config.subject = [participant_label]
                    # Note: config uses plural field names for tasks/sessions/runs/spaces
                    if args.task:
                        config.tasks = [args.task]
                    if args.session:
                        config.sessions = [args.session]
                    if args.run:
                        config.runs = [args.run]
                    if args.space:
                        config.spaces = [args.space]
                    if args.label:
                        config.label = args.label
                    if args.atlas:
                        config.atlas = args.atlas
                    if args.method:
                        config.method = args.method
                    
                    # Handle ROI-to-voxel specific CLI arguments
                    if hasattr(args, 'roi_masks') and args.roi_masks:
                        config.roi_masks = [Path(m) for m in args.roi_masks]
                    
                    if hasattr(args, 'roi_atlas') and args.roi_atlas:
                        config.roi_atlas = args.roi_atlas
                    
                    if hasattr(args, 'roi_label') and args.roi_label:
                        config.roi_label = args.roi_label
                    
                    # Handle condition-based masking CLI options
                    # For each loop iteration, pass only the current condition
                    _configure_condition_masking(args, config, logger, condition)
                    
                    run_participant_pipeline(
                        bids_dir=args.bids_dir,
                        output_dir=args.output_dir,
                        config=config,
                        derivatives=derivatives_dict,
                        logger=logger,
                    )
        else:  # group
            # Load or create group config
            if args.config:
                logger.info(f"Loading configuration from: {args.config}")
                config_dict = load_config_file(args.config)
                config = config_from_dict(config_dict, GroupConfig)
            else:
                logger.info("Using default configuration")
                config = GroupConfig()
            
            # Override config with CLI arguments
            if hasattr(args, 'participant_derivatives') and args.participant_derivatives:
                config.participant_derivatives = args.participant_derivatives
            if args.participant_label:
                # For group-level, participant_label is a list of labels
                config.subjects = args.participant_label
            if args.task:
                config.tasks = [args.task]
            if args.session:
                config.sessions = [args.session]
            if hasattr(args, 'atlas') and args.atlas:
                config.atlas = args.atlas
            if hasattr(args, 'method') and args.method:
                config.method = args.method
            if args.label:
                config.label = args.label
            
            run_group_pipeline(
                bids_dir=args.bids_dir,
                output_dir=args.output_dir,
                config=config,
                derivatives=derivatives_dict,
                logger=logger,
            )
        
        logger.info("=" * 60)
        logger.info("Analysis completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def _configure_condition_masking(args, config: ParticipantConfig, logger: logging.Logger, condition: Optional[str] = None):
    """Configure condition-based masking from CLI arguments.
    
    When multiple conditions are provided via --conditions, this function is called
    once per condition to run independent analyses. Each call receives a single
    condition value (or None if conditions weren't specified).
    
    Args:
        args: Parsed CLI arguments.
        config: ParticipantConfig to update.
        logger: Logger instance.
        condition: Single condition value (from loop iteration). If provided, only
                  this condition is used. If None, uses all conditions from args.
    """
    # Check if condition-based masking is enabled
    has_conditions = condition is not None or (hasattr(args, 'conditions') and args.conditions)
    
    if not has_conditions:
        return  # No condition-based masking specified
    
    # Determine which conditions to use:
    # - If iterating with a single condition, use just that one
    # - Otherwise, use all conditions from args (should be single value for non-looped case)
    conditions_to_set = [condition] if condition else args.conditions
    
    # Enable condition masking in both config paths
    # (condition_masking is the new primary config)
    config.condition_masking.enabled = True
    config.condition_masking.conditions = conditions_to_set
    logger.info(f"Condition-based masking enabled: {conditions_to_set}")
    
    # Also enable temporal_censoring (required for _apply_temporal_censoring to run)
    config.temporal_censoring.enabled = True
    # Store conditions in legacy config path for backward compatibility
    config.temporal_censoring.condition_selection = {
        'enabled': True,
        'conditions': conditions_to_set,
    }
    
    if hasattr(args, 'events_file') and args.events_file:
        config.condition_masking.events_file = args.events_file
        # Also store in legacy config
        config.temporal_censoring.condition_selection['events_file'] = args.events_file
    else:
        config.temporal_censoring.condition_selection['events_file'] = 'auto'
    
    if hasattr(args, 'transition_buffer') and args.transition_buffer > 0:
        config.condition_masking.transition_buffer = args.transition_buffer
        logger.info(f"  Transition buffer: {args.transition_buffer}s")


if __name__ == "__main__":
    main()
