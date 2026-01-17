"""Core pipeline orchestration for Connectomix."""

from connectomix.core.version import __version__
from connectomix.core.participant import run_participant_pipeline
from connectomix.core.group import run_group_pipeline

__all__ = [
    "__version__",
    "run_participant_pipeline",
    "run_group_pipeline",
]
