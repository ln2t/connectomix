"""Configuration loading and validation for Connectomix."""

from connectomix.config.defaults import ParticipantConfig, GroupConfig
from connectomix.config.loader import load_config_file, merge_configs, config_from_dict
from connectomix.config.validator import ConfigValidator
from connectomix.config.strategies import DENOISING_STRATEGIES

__all__ = [
    "ParticipantConfig",
    "GroupConfig",
    "load_config_file",
    "merge_configs",
    "config_from_dict",
    "ConfigValidator",
    "DENOISING_STRATEGIES",
]
