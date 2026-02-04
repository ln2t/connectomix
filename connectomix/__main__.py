"""Main entry point for Connectomix."""

import sys
import logging
from pathlib import Path

from connectomix.cli import create_parser, parse_derivatives_arg
from connectomix.utils.logging import setup_logging
from connectomix.config.defaults import (
    ParticipantConfig,
    GroupConfig,
    TemporalCensoringConfig,
    ConditionSelectionConfig,
    MotionCensoringConfig,
)
from connectomix.config.loader import load_config_file
from connectomix.config.strategies import get_denoising_strategy, DenoisingStrategySpec
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
                config = ParticipantConfig(**config_dict)
            else:
                logger.info("Using default configuration")
                config = ParticipantConfig()
            
            # Get participant labels to process (convert to list if needed)
            participant_labels = args.participant_label if args.participant_label else [None]
            
            # Loop over each participant label
            for participant_label in participant_labels:
                # Create fresh config for each participant
                if args.config:
                    config_dict = load_config_file(args.config)
                    config = ParticipantConfig(**config_dict)
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
                if args.denoising:
                    # Get strategy specification
                    strategy_spec = get_denoising_strategy(args.denoising)
                    
                    # Check for rigid strategy conflicts
                    if strategy_spec.is_rigid:
                        has_manual_fd = hasattr(args, 'fd_threshold') and args.fd_threshold is not None
                        has_manual_scrub = hasattr(args, 'scrub') and args.scrub and args.scrub > 0
                        if has_manual_fd or has_manual_scrub:
                            raise ValueError(
                                f"The '{args.denoising}' denoising strategy includes its own "
                                f"censoring parameters (FD threshold={strategy_spec.fd_threshold} cm, "
                                f"scrub={strategy_spec.min_segment_length}) and cannot be combined "
                                f"with --fd-threshold or --scrub. Use a different strategy or "
                                f"remove the manual censoring options."
                            )
                    
                    # Apply strategy
                    config.denoising_strategy = args.denoising
                    config.confounds = strategy_spec.confounds
                    
                    # Apply strategy censoring parameters if defined
                    if strategy_spec.fd_threshold is not None:
                        config.temporal_censoring.enabled = True
                        config.temporal_censoring.motion_censoring.enabled = True
                        config.temporal_censoring.motion_censoring.fd_threshold = strategy_spec.fd_threshold
                        config.temporal_censoring.motion_censoring.min_segment_length = strategy_spec.min_segment_length
                        logger.info(f"Denoising strategy '{args.denoising}' includes censoring:")
                        mm_equiv = strategy_spec.fd_threshold * 10.0
                        logger.info(f"  FD threshold: {strategy_spec.fd_threshold} cm ({mm_equiv:.1f} mm)")
                        if strategy_spec.min_segment_length > 0:
                            logger.info(f"  Min segment length: {strategy_spec.min_segment_length} volumes")
                if args.label:
                    config.label = args.label
                if args.atlas:
                    config.atlas = args.atlas
                if args.method:
                    config.method = args.method
                
                # Handle temporal censoring CLI options
                _configure_temporal_censoring(args, config, logger)
                
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
                config = GroupConfig(**config_dict)
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


def _configure_temporal_censoring(args, config: ParticipantConfig, logger: logging.Logger):
    """Configure temporal censoring from CLI arguments.
    
    Args:
        args: Parsed CLI arguments.
        config: ParticipantConfig to update.
        logger: Logger instance.
    """
    # Check if any censoring options are provided
    has_conditions = hasattr(args, 'conditions') and args.conditions
    has_fd_threshold = hasattr(args, 'fd_threshold') and args.fd_threshold is not None
    has_drop_initial = hasattr(args, 'drop_initial') and args.drop_initial and args.drop_initial > 0
    has_scrub = hasattr(args, 'scrub') and args.scrub and args.scrub > 0
    
    # Validate --scrub requires --fd-threshold
    if has_scrub and not has_fd_threshold:
        # Check if strategy already set fd_threshold
        if not config.temporal_censoring.motion_censoring.enabled:
            raise ValueError(
                "--scrub requires --fd-threshold to be set. "
                "Segment filtering only makes sense after motion censoring."
            )
    
    if not (has_conditions or has_fd_threshold or has_drop_initial):
        return  # No censoring options specified
    
    # Enable temporal censoring
    config.temporal_censoring.enabled = True
    logger.info("Temporal censoring enabled")
    
    # Condition-based censoring (task fMRI)
    if has_conditions:
        config.temporal_censoring.condition_selection.enabled = True
        config.temporal_censoring.condition_selection.conditions = args.conditions
        logger.info(f"  Condition selection: {args.conditions}")
        
        if hasattr(args, 'events_file') and args.events_file:
            config.temporal_censoring.condition_selection.events_file = args.events_file
        
        if hasattr(args, 'include_baseline') and args.include_baseline:
            config.temporal_censoring.condition_selection.include_baseline = True
            logger.info("  Including baseline periods")
        
        if hasattr(args, 'transition_buffer') and args.transition_buffer > 0:
            config.temporal_censoring.condition_selection.transition_buffer = args.transition_buffer
            logger.info(f"  Transition buffer: {args.transition_buffer}s")
    
    # Motion censoring (FD threshold)
    if has_fd_threshold:
        config.temporal_censoring.motion_censoring.enabled = True
        config.temporal_censoring.motion_censoring.fd_threshold = args.fd_threshold
        # Show both cm (internal units) and mm (more common in the literature)
        try:
            mm_equiv = float(args.fd_threshold) * 10.0
            logger.info(f"  Motion censoring: FD > {args.fd_threshold} cm ({mm_equiv:.2f} mm)")
        except Exception:
            logger.info(f"  Motion censoring: FD > {args.fd_threshold} cm")
        
        if hasattr(args, 'fd_extend') and args.fd_extend > 0:
            config.temporal_censoring.motion_censoring.extend_before = args.fd_extend
            config.temporal_censoring.motion_censoring.extend_after = args.fd_extend
            logger.info(f"  Motion extend: ±{args.fd_extend} volumes")
    
    # Segment filtering (scrub)
    if has_scrub:
        config.temporal_censoring.motion_censoring.min_segment_length = args.scrub
        logger.info(f"  Segment filtering: min {args.scrub} contiguous volumes")
    
    # Drop initial volumes
    if has_drop_initial:
        config.temporal_censoring.drop_initial_volumes = args.drop_initial
        logger.info(f"  Dropping {args.drop_initial} initial volumes")


if __name__ == "__main__":
    main()
