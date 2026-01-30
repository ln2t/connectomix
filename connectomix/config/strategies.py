"""Predefined denoising strategies for fMRI data."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DenoisingStrategySpec:
    """Specification for a denoising strategy.
    
    Attributes:
        confounds: List of confound column names to regress out.
        description: Human-readable description of the strategy.
        fd_threshold: If set, enables motion censoring with this FD threshold (in cm).
        min_segment_length: Minimum contiguous segment length to keep after censoring.
            If > 0, segments shorter than this are also censored (scrubbing).
        is_rigid: If True, manual --fd-threshold or --scrub options are not allowed
            when using this strategy (raises an error).
    """
    confounds: List[str]
    description: str = ""
    fd_threshold: Optional[float] = None
    min_segment_length: int = 0
    is_rigid: bool = False


# Common confound sets for reuse
_MOTION_6P = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z"
]

_MOTION_12P = _MOTION_6P + [
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1"
]

_MOTION_24P = _MOTION_12P + [
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2"
]

_CSF_DERIVATIVES = [
    "csf", "csf_derivative1", "csf_power2", "csf_derivative1_power2"
]

_WM_DERIVATIVES = [
    "white_matter", "white_matter_derivative1", "white_matter_power2", "white_matter_derivative1_power2"
]


# Registry of predefined denoising strategies
DENOISING_STRATEGIES: Dict[str, DenoisingStrategySpec] = {
    "minimal": DenoisingStrategySpec(
        confounds=_MOTION_6P.copy(),
        description="6 motion parameters only (basic motion correction)"
    ),
    
    "csfwm_6p": DenoisingStrategySpec(
        confounds=["csf", "white_matter"] + _MOTION_6P,
        description="CSF + WM signal + 6 motion parameters"
    ),
    
    "csfwm_12p": DenoisingStrategySpec(
        confounds=["csf", "white_matter"] + _MOTION_12P,
        description="CSF + WM signal + 12 motion parameters (6 + derivatives)"
    ),
    
    "gs_csfwm_6p": DenoisingStrategySpec(
        confounds=["global_signal", "csf", "white_matter"] + _MOTION_6P,
        description="Global signal + CSF + WM + 6 motion parameters"
    ),
    
    "gs_csfwm_12p": DenoisingStrategySpec(
        confounds=["global_signal", "csf", "white_matter"] + _MOTION_12P,
        description="Global signal + CSF + WM + 12 motion parameters (includes derivatives)"
    ),
    
    "csfwm_24p": DenoisingStrategySpec(
        confounds=["csf", "white_matter"] + _MOTION_24P,
        description="CSF + WM + 24 motion parameters (includes derivatives and squares)"
    ),
    
    "compcor_6p": DenoisingStrategySpec(
        confounds=[
            "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
            "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05"
        ] + _MOTION_6P,
        description="aCompCor (first 6 components) + 6 motion parameters"
    ),
    
    "simpleGSR": DenoisingStrategySpec(
        confounds=["global_signal", "csf", "white_matter"] + _MOTION_24P,
        description="Global signal regression: global_signal + CSF + WM + 24 motion parameters"
    ),
    
    "scrubbing5": DenoisingStrategySpec(
        confounds=_CSF_DERIVATIVES + _WM_DERIVATIVES + _MOTION_24P,
        description=(
            "Scrubbing strategy: CSF (with derivatives) + WM (with derivatives) + "
            "24 motion parameters + FD censoring (0.5 cm) + segment filtering (min 5 volumes)"
        ),
        fd_threshold=0.5,
        min_segment_length=5,
        is_rigid=True
    ),
}


def get_denoising_strategy(name: str) -> DenoisingStrategySpec:
    """Get the specification for a predefined denoising strategy.
    
    Args:
        name: Name of the denoising strategy
    
    Returns:
        DenoisingStrategySpec with confounds and optional censoring parameters
    
    Raises:
        ValueError: If strategy name is not recognized
    """
    if name not in DENOISING_STRATEGIES:
        available = ", ".join(DENOISING_STRATEGIES.keys())
        raise ValueError(
            f"Unknown denoising strategy: '{name}'. "
            f"Available strategies: {available}"
        )
    
    return DENOISING_STRATEGIES[name]


def get_denoising_confounds(name: str) -> List[str]:
    """Get confound list for a predefined denoising strategy.
    
    This is a convenience function that returns only the confounds list.
    For full strategy specification including censoring params, use get_denoising_strategy().
    
    Args:
        name: Name of the denoising strategy
    
    Returns:
        List of confound column names
    
    Raises:
        ValueError: If strategy name is not recognized
    """
    return get_denoising_strategy(name).confounds


def list_denoising_strategies() -> List[str]:
    """Get list of available denoising strategy names.
    
    Returns:
        List of strategy names
    """
    return list(DENOISING_STRATEGIES.keys())


def describe_denoising_strategy(name: str) -> str:
    """Get description of a denoising strategy.
    
    Args:
        name: Name of the denoising strategy
    
    Returns:
        Human-readable description
    
    Raises:
        ValueError: If strategy name is not recognized
    """
    return get_denoising_strategy(name).description
