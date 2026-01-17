"""Main entry point for Connectomix."""

import sys
import logging
from pathlib import Path

from connectomix.cli import create_parser, parse_derivatives_arg
from connectomix.utils.logging import setup_logging
from connectomix.config.defaults import ParticipantConfig, GroupConfig
from connectomix.config.loader import load_config_file
from connectomix.config.strategies import get_denoising_strategy
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
            
            # Override config with CLI arguments
            if args.participant_label:
                config.subject = [args.participant_label]
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
                # store requested strategy and apply it to confounds
                config.denoising_strategy = args.denoising
                config.confounds = get_denoising_strategy(args.denoising)
            
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


if __name__ == "__main__":
    main()
