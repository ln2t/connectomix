"""Default configuration dataclasses for Connectomix."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class MotionCensoringConfig:
    """Configuration for motion-based censoring.
    
    Attributes:
        enabled: Whether motion censoring is enabled.
        fd_threshold: Framewise displacement threshold in mm.
        fd_column: Column name for FD in confounds file.
        extend_before: Also censor N volumes before high-motion.
        extend_after: Also censor N volumes after high-motion.
    """
    enabled: bool = False
    fd_threshold: float = 0.5
    fd_column: str = "framewise_displacement"
    extend_before: int = 0
    extend_after: int = 0


@dataclass
class ConditionSelectionConfig:
    """Configuration for condition-based censoring (task fMRI).
    
    When enabled, separate connectivity matrices are computed for each
    condition, using only timepoints belonging to that condition.
    
    Attributes:
        enabled: Whether condition selection is enabled.
        events_file: Path to events TSV file, or "auto" to find from BIDS.
        conditions: List of condition names to include (empty = all).
        include_baseline: Include timepoints not in any condition.
        transition_buffer: Seconds to exclude around condition boundaries.
    """
    enabled: bool = False
    events_file: Optional[str] = "auto"
    conditions: List[str] = field(default_factory=list)
    include_baseline: bool = False
    transition_buffer: float = 0.0


@dataclass
class TemporalCensoringConfig:
    """Configuration for temporal censoring.
    
    Temporal censoring removes specific timepoints (volumes) from fMRI data
    before connectivity analysis. This is useful for:
    
    - **Dummy scan removal**: Discard initial volumes during scanner equilibration.
    - **Motion scrubbing**: Remove high-motion timepoints (based on FD).
    - **Condition selection**: For task fMRI, analyze only specific conditions.
    
    By default, temporal censoring is disabled. Enable specific features
    by setting the appropriate sub-configurations.
    
    Attributes:
        enabled: Master switch for temporal censoring.
        stage: When to apply censoring ("before_denoising" or "after_denoising").
        drop_initial_volumes: Number of initial volumes to drop.
        motion_censoring: Motion-based censoring configuration.
        condition_selection: Condition-based censoring configuration.
        custom_mask_file: Path to custom censoring mask TSV.
        min_volumes_retained: Minimum number of volumes required.
        min_fraction_retained: Minimum fraction of volumes required.
        warn_fraction_retained: Warn if retention falls below this.
    """
    enabled: bool = False
    stage: str = "before_denoising"
    drop_initial_volumes: int = 0
    motion_censoring: MotionCensoringConfig = field(default_factory=MotionCensoringConfig)
    condition_selection: ConditionSelectionConfig = field(default_factory=ConditionSelectionConfig)
    custom_mask_file: Optional[Path] = None
    min_volumes_retained: int = 50
    min_fraction_retained: float = 0.3
    warn_fraction_retained: float = 0.5


@dataclass
class ParticipantConfig:
    """Configuration for participant-level analysis.
    
    Attributes:
        subject: List of subject IDs to process
        tasks: List of task names to process
        sessions: List of session IDs to process
        runs: List of run IDs to process
        spaces: List of space names to process
        confounds: List of confound column names for regression
        high_pass: High-pass filter cutoff in Hz
        low_pass: Low-pass filter cutoff in Hz
        ica_aroma: Use ICA-AROMA denoised files
        reference_functional_file: Reference image path or "first_functional_file"
        overwrite_denoised_files: Whether to re-denoise if files exist
        method: Analysis method (seedToVoxel, roiToVoxel, seedToSeed, roiToRoi)
        seeds_file: Path to TSV file with seed coordinates (for seed methods)
        radius: Sphere radius in mm (for seed methods)
        roi_masks: List of paths to ROI mask files (for roiToVoxel)
        atlas: Atlas name or "canica" (for roiToRoi)
        connectivity_kind: Type of connectivity measure
        n_components: Number of ICA components (for CanICA)
        canica_threshold: Threshold for extracting regions from ICA components
        canica_min_region_size: Minimum region size in voxels (for CanICA)
    """
    
    # BIDS entity filters
    subject: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    sessions: Optional[List[str]] = None
    runs: Optional[List[str]] = None
    spaces: Optional[List[str]] = None
    
    # Custom label for output filenames
    label: Optional[str] = None
    
    # Preprocessing/denoising
    confounds: List[str] = field(default_factory=lambda: [
        "csf", "white_matter",
        "trans_x", "trans_y", "trans_z", 
        "rot_x", "rot_y", "rot_z"
    ])
    high_pass: float = 0.01
    low_pass: float = 0.08
    ica_aroma: bool = False
    reference_functional_file: str = "first_functional_file"
    overwrite_denoised_files: bool = False  # Skip denoising if file exists
    
    # Analysis method
    method: str = "roiToRoi"
    
    # Method-specific parameters - Seed-based
    seeds_file: Optional[Path] = None
    radius: float = 5.0
    
    # Method-specific parameters - ROI-based
    roi_masks: Optional[List[Path]] = None
    atlas: str = "schaefer2018_100"
    
    # Connectivity computation
    connectivity_kind: str = "correlation"
    
    # CanICA parameters
    n_components: int = 20
    canica_threshold: float = 1.0
    canica_min_region_size: int = 50
    
    # Temporal censoring configuration
    temporal_censoring: TemporalCensoringConfig = field(default_factory=TemporalCensoringConfig)
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        from connectomix.config.validator import ConfigValidator
        
        validator = ConfigValidator()
        
        # Validate method
        validator.validate_choice(
            self.method,
            ["seedToVoxel", "roiToVoxel", "seedToSeed", "roiToRoi"],
            "method"
        )
        
        # Validate alpha values
        validator.validate_alpha(self.high_pass, "high_pass")
        validator.validate_alpha(self.low_pass, "low_pass")
        
        # Validate positive values
        validator.validate_positive(self.radius, "radius")
        validator.validate_positive(self.n_components, "n_components")
        validator.validate_positive(self.canica_threshold, "canica_threshold")
        validator.validate_positive(self.canica_min_region_size, "canica_min_region_size")
        
        # Validate method-specific requirements
        if self.method in ["seedToVoxel", "seedToSeed"]:
            if self.seeds_file is None:
                validator.errors.append(
                    f"seeds_file is required for method '{self.method}'"
                )
        
        if self.method == "roiToVoxel":
            if self.roi_masks is None or len(self.roi_masks) == 0:
                validator.errors.append(
                    f"roi_masks is required for method '{self.method}'"
                )
        
        if self.method == "roiToRoi":
            if self.atlas is None:
                validator.errors.append(
                    f"atlas is required for method '{self.method}'"
                )
        
        # Validate ICA-AROMA incompatibility with motion parameters
        if self.ica_aroma:
            motion_params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
            motion_in_confounds = any(mp in self.confounds for mp in motion_params)
            if motion_in_confounds:
                validator.errors.append(
                    "ICA-AROMA is incompatible with motion parameters in confounds. "
                    "Remove motion parameters from confounds list."
                )
        
        # Raise if any errors
        validator.raise_if_errors()


@dataclass
class GroupConfig:
    """Configuration for group-level analysis.
    
    Placeholder configuration - group analysis is under development.
    """
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Currently a no-op as group analysis is under development.
        """
        pass
