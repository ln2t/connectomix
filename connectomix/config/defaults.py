"""Default configuration dataclasses for Connectomix."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


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
    
    Attributes:
        subject: List of subject IDs to include in analysis
        task: Task name (must be single value)
        session: Session ID (must be single value)
        run: Run ID (must be single value)
        space: Space name (must be single value)
        method: Analysis method (must match participant-level)
        smoothing: Spatial smoothing FWHM in mm
        analysis_name: Custom name for this analysis
        covariates: List of covariate column names from participants.tsv
        add_intercept: Add intercept to design matrix
        paired_tests: Perform paired tests (NOT YET IMPLEMENTED)
        contrast: String expression defining the contrast
        uncorrected_alpha: Significance level for uncorrected thresholding
        cluster_forming_alpha: Alpha for cluster-forming threshold
        fdr_alpha: False Discovery Rate alpha
        fwe_alpha: Family-Wise Error rate alpha
        two_sided_test: Perform two-sided vs one-sided test
        thresholding_strategies: List of thresholding strategies to apply
        n_permutations: Number of permutations for FWE correction
        n_jobs: Number of parallel jobs
    """
    
    # BIDS entity filters
    subject: Optional[List[str]] = None
    task: Optional[str] = None
    session: Optional[str] = None
    run: Optional[str] = None
    space: Optional[str] = None
    
    # Method
    method: str = "roiToRoi"
    smoothing: float = 8.0
    
    # Analysis naming
    analysis_name: str = "default"
    
    # Design matrix
    covariates: List[str] = field(default_factory=list)
    add_intercept: bool = True
    paired_tests: bool = False  # NOT YET IMPLEMENTED
    
    # Contrast
    contrast: str = "intercept"
    
    # Statistical thresholding
    uncorrected_alpha: float = 0.001
    cluster_forming_alpha: float = 0.01
    fdr_alpha: float = 0.05
    fwe_alpha: float = 0.05
    two_sided_test: bool = True
    thresholding_strategies: List[str] = field(default_factory=lambda: [
        "uncorrected", "fdr", "fwe"
    ])
    
    # Computational
    n_permutations: int = 10000
    n_jobs: int = 1
    
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
        validator.validate_alpha(self.uncorrected_alpha, "uncorrected_alpha")
        validator.validate_alpha(self.cluster_forming_alpha, "cluster_forming_alpha")
        validator.validate_alpha(self.fdr_alpha, "fdr_alpha")
        validator.validate_alpha(self.fwe_alpha, "fwe_alpha")
        
        # Validate positive values
        validator.validate_positive(self.smoothing, "smoothing")
        validator.validate_positive(self.n_permutations, "n_permutations")
        validator.validate_positive(self.n_jobs, "n_jobs")
        
        # Validate thresholding strategies
        for strategy in self.thresholding_strategies:
            validator.validate_choice(
                strategy,
                ["uncorrected", "fdr", "fwe"],
                "thresholding_strategy"
            )
        
        # Validate contrast is provided
        if not self.contrast:
            validator.errors.append("contrast must be specified for group-level analysis")
        
        # Warn about paired tests
        if self.paired_tests:
            validator.errors.append(
                "paired_tests=True is not yet implemented. Use paired_tests=False."
            )
        
        # Raise if any errors
        validator.raise_if_errors()
