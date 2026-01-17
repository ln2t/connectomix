"""Predefined denoising strategies for fMRI data."""

from typing import Dict, List


# Registry of predefined denoising strategies
DENOISING_STRATEGIES: Dict[str, List[str]] = {
    "minimal": [
        # 6 motion parameters only
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z"
    ],
    
    "csfwm_6p": [
        # CSF + WM signal + 6 motion parameters
        "csf", "white_matter",
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z"
    ],
    
    "csfwm_12p": [
        # CSF + WM signal + 12 motion parameters (6 + derivatives)
        "csf", "white_matter",
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z",
        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
        "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1"
    ],
    
    "gs_csfwm_6p": [
        # Global signal + CSF + WM + 6 motion parameters
        "global_signal", "csf", "white_matter",
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z"
    ],
    
    "gs_csfwm_12p": [
        # Global signal + CSF + WM + 12 motion parameters (6 + derivatives)
        "global_signal", "csf", "white_matter",
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z",
        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
        "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1"
    ],
    
    "csfwm_24p": [
        # CSF+WM + 24 motion parameters (6 + derivatives + squares)
        "csf", "white_matter",
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z",
        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
        "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
        "trans_x_power2", "trans_y_power2", "trans_z_power2",
        "rot_x_power2", "rot_y_power2", "rot_z_power2",
        "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
        "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2"
    ],
    
    "compcor_6p": [
        # CompCor components + 6 motion parameters
        "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
        "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05",
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z"
    ],
}


def get_denoising_strategy(name: str) -> List[str]:
    """Get confound list for a predefined denoising strategy.
    
    Args:
        name: Name of the denoising strategy
    
    Returns:
        List of confound column names
    
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
    descriptions = {
        "minimal": "6 motion parameters only (basic motion correction)",
        "csfwm_6p": "CSF+WM signal + 6 motion parameters",
        "gs_csfwm_12p": "Global signal + CSF + WM + 12 motion parameters (includes derivatives)",
        "csfwm_24p": "CSF+WM + 24 motion parameters (includes derivatives and squares)",
        "compcor_6p": "aCompCor (first 6 components) + 6 motion parameters",
    }
    
    if name not in descriptions:
        available = ", ".join(DENOISING_STRATEGIES.keys())
        raise ValueError(
            f"Unknown denoising strategy: '{name}'. "
            f"Available strategies: {available}"
        )
    
    return descriptions[name]
