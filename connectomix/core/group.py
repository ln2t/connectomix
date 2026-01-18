"""Group-level pipeline orchestration.

This module will orchestrate the complete group-level analysis workflow.
Currently under development.
"""

import logging
from pathlib import Path
from typing import Dict, Optional


def run_group_pipeline(
    bids_dir: Path,
    output_dir: Path,
    config: "GroupConfig",  # noqa: F821
    derivatives: Optional[Dict[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Run the group-level statistical analysis pipeline.
    
    Currently a placeholder - group analysis is under development.
    
    Args:
        bids_dir: Path to BIDS dataset root.
        output_dir: Path for output derivatives.
        config: GroupConfig instance with analysis parameters.
        derivatives: Optional dict mapping derivative names to paths.
        logger: Logger instance. If None, creates one.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Group analysis in progress")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Group-level analysis is currently under development.")
    logger.info("This feature will be available in a future release.")
    logger.info("")
    logger.info("For now, please use participant-level analysis:")
    logger.info("  connectomix BIDS_DIR OUTPUT_DIR participant [options]")
    logger.info("")
