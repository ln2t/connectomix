"""Temporal censoring for fMRI time series.

Temporal censoring removes specific timepoints (volumes) from fMRI data before
or after denoising. This is useful for:

1. **Dummy scan removal**: Discard initial volumes acquired during scanner 
   steady-state equilibration (typically 4-10 volumes).

2. **Motion scrubbing**: Remove timepoints with excessive head motion, 
   identified by high framewise displacement (FD) values.

3. **Condition-based selection**: For task fMRI, analyze only specific 
   experimental conditions (e.g., task vs rest periods).

4. **Custom censoring**: Apply user-defined censoring masks.

The module generates boolean temporal masks that can be applied to both
the fMRI data and confounds to maintain consistency.

Example:
    >>> from connectomix.preprocessing.censoring import TemporalCensor
    >>> censor = TemporalCensor(config, n_volumes=200, tr=2.0)
    >>> censor.apply_initial_drop()
    >>> censor.apply_motion_censoring(confounds_df)
    >>> censored_img = censor.apply_to_image(func_img)
    >>> censored_confounds = censor.apply_to_confounds(confounds_df)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import nibabel as nib
import numpy as np
import pandas as pd

from connectomix.utils.exceptions import PreprocessingError


logger = logging.getLogger(__name__)


@dataclass
class MotionCensoringConfig:
    """Configuration for motion-based censoring.
    
    Attributes:
        enabled: Whether motion censoring is enabled.
        fd_threshold: Framewise displacement threshold in cm (fMRIPrep reports FD in cm).
        fd_column: Column name for FD in confounds file.
        extend_before: Also censor N volumes before high-motion.
        extend_after: Also censor N volumes after high-motion.
        min_segment_length: Minimum contiguous segment length to keep (scrubbing).
            If > 0, continuous segments shorter than this are also censored.
    """
    enabled: bool = False
    fd_threshold: float = 0.5
    fd_column: str = "framewise_displacement"
    extend_before: int = 0
    extend_after: int = 0
    min_segment_length: int = 0


@dataclass
class ConditionSelectionConfig:
    """Configuration for condition-based censoring.
    
    Attributes:
        enabled: Whether condition selection is enabled.
        events_file: Path to events TSV file, or "auto" to find from BIDS.
        conditions: List of condition names to include.
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


class TemporalCensor:
    """Generate and apply temporal censoring masks.
    
    This class manages the creation and application of temporal masks
    for censoring fMRI volumes. Multiple censoring methods can be
    combined, and the resulting mask is applied consistently to both
    the image data and confounds.
    
    Attributes:
        config: Temporal censoring configuration.
        n_volumes: Number of volumes in the original data.
        tr: Repetition time in seconds.
        mask: Boolean mask (True = keep, False = censor).
        censoring_log: Dictionary tracking censoring reasons per volume.
    """
    
    def __init__(
        self,
        config: TemporalCensoringConfig,
        n_volumes: int,
        tr: float,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize temporal censor.
        
        Args:
            config: Temporal censoring configuration.
            n_volumes: Number of volumes in the data.
            tr: Repetition time in seconds.
            logger: Optional logger instance.
        """
        self.config = config
        self.n_volumes = n_volumes
        self.tr = tr
        self._logger = logger or logging.getLogger(__name__)
        
        # Initialize mask (all True = keep all volumes)
        self.mask = np.ones(n_volumes, dtype=bool)
        
        # Track reasons for censoring each volume
        self.censoring_log: Dict[int, List[str]] = {}
        
        # Store condition-specific masks for multi-condition analysis
        self.condition_masks: Dict[str, np.ndarray] = {}
        
        self._logger.debug(f"Initialized temporal censor for {n_volumes} volumes (TR={tr}s)")
    
    def apply_initial_drop(self) -> int:
        """Mark first N volumes for censoring.
        
        Returns:
            Number of volumes marked for censoring.
        """
        n_drop = self.config.drop_initial_volumes
        if n_drop <= 0:
            return 0
        
        if n_drop >= self.n_volumes:
            raise PreprocessingError(
                f"Cannot drop {n_drop} initial volumes from data with only "
                f"{self.n_volumes} volumes."
            )
        
        for i in range(n_drop):
            self.mask[i] = False
            self._add_censoring_reason(i, f"initial_drop")
        
        self._logger.info(f"Temporal censoring: marked {n_drop} initial volumes for removal")
        return n_drop
    
    def apply_motion_censoring(self, confounds_df: pd.DataFrame) -> int:
        """Mark high-motion volumes for censoring based on FD.
        
        Args:
            confounds_df: Confounds DataFrame with FD column.
            
        Returns:
            Number of volumes marked for censoring.
        """
        mc = self.config.motion_censoring
        if not mc.enabled:
            return 0
        
        # Get FD values
        if mc.fd_column not in confounds_df.columns:
            available = [c for c in confounds_df.columns if 'displacement' in c.lower() or 'fd' in c.lower()]
            raise PreprocessingError(
                f"FD column '{mc.fd_column}' not found in confounds. "
                f"Available motion-related columns: {available}"
            )
        
        fd_values = confounds_df[mc.fd_column].values
        
        # Handle NaN in first volume (common in fMRIPrep output)
        fd_values = np.nan_to_num(fd_values, nan=0.0)
        
        # Find volumes exceeding threshold
        high_motion = fd_values > mc.fd_threshold
        
        # Extend censoring before and after
        if mc.extend_before > 0 or mc.extend_after > 0:
            extended = np.zeros_like(high_motion)
            for i, is_high in enumerate(high_motion):
                if is_high:
                    start = max(0, i - mc.extend_before)
                    end = min(self.n_volumes, i + mc.extend_after + 1)
                    extended[start:end] = True
            high_motion = extended
        
        # Apply to mask
        n_censored = 0
        for i, censor in enumerate(high_motion):
            if censor:
                if self.mask[i]:  # Only count if not already censored
                    n_censored += 1
                self.mask[i] = False
                # Include unit (cm) in the censoring reason string for clarity
                self._add_censoring_reason(i, f"motion_fd>{mc.fd_threshold}cm")
        
        try:
            mm_equiv = float(mc.fd_threshold) * 10.0
            self._logger.info(
                f"Temporal censoring: marked {n_censored} volumes for motion "
                f"(FD > {mc.fd_threshold} cm ({mm_equiv:.2f} mm))"
            )
        except Exception:
            self._logger.info(
                f"Temporal censoring: marked {n_censored} volumes for motion "
                f"(FD > {mc.fd_threshold} cm)"
            )
        return n_censored
    
    def apply_segment_filtering(self, min_segment_length: int) -> int:
        """Remove continuous segments shorter than min_segment_length.
        
        After motion censoring, this method identifies contiguous runs of
        kept volumes (mask == True) and censors any segments that are
        shorter than the specified minimum length. This ensures only
        sufficiently long continuous data segments are used for connectivity.
        
        Args:
            min_segment_length: Minimum number of contiguous volumes required.
                Segments shorter than this will be censored.
        
        Returns:
            Number of additional volumes censored due to short segments.
        """
        if min_segment_length <= 0:
            return 0
        
        n_censored = 0
        
        # Find contiguous segments of kept volumes
        segment_start = None
        segments = []  # List of (start, end) tuples
        
        for i in range(self.n_volumes):
            if self.mask[i]:
                if segment_start is None:
                    segment_start = i
            else:
                if segment_start is not None:
                    segments.append((segment_start, i))
                    segment_start = None
        
        # Handle segment at the end
        if segment_start is not None:
            segments.append((segment_start, self.n_volumes))
        
        # Censor segments that are too short
        for start, end in segments:
            segment_length = end - start
            if segment_length < min_segment_length:
                for i in range(start, end):
                    self.mask[i] = False
                    self._add_censoring_reason(i, f"segment_too_short<{min_segment_length}")
                    n_censored += 1
        
        if n_censored > 0:
            self._logger.info(
                f"Temporal censoring: marked {n_censored} additional volumes "
                f"(segments shorter than {min_segment_length} volumes)"
            )
        
        return n_censored
    
    def apply_condition_selection(
        self,
        events_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Create masks for condition-based selection.
        
        For each condition in the events file, creates a boolean mask
        indicating which volumes belong to that condition.
        
        Args:
            events_df: Events DataFrame with 'onset', 'duration', 'trial_type'.
            
        Returns:
            Dictionary mapping condition names to boolean masks.
        """
        cs = self.config.condition_selection
        if not cs.enabled:
            return {}
        
        # Validate events DataFrame
        required_cols = ['onset', 'duration']
        for col in required_cols:
            if col not in events_df.columns:
                raise PreprocessingError(
                    f"Events file missing required column: '{col}'. "
                    f"Available columns: {list(events_df.columns)}"
                )
        
        # Determine condition column
        condition_col = None
        for possible in ['trial_type', 'condition', 'event_type']:
            if possible in events_df.columns:
                condition_col = possible
                break
        
        if condition_col is None:
            raise PreprocessingError(
                "Events file missing condition column. Expected one of: "
                "'trial_type', 'condition', 'event_type'. "
                f"Available columns: {list(events_df.columns)}"
            )
        
        # Get all unique conditions
        all_conditions = events_df[condition_col].unique().tolist()
        self._logger.info(f"Found conditions in events file: {all_conditions}")
        
        # Check if user requested "baseline" (special keyword for inter-trial intervals)
        baseline_requested = False
        if cs.conditions:
            # Check for special "baseline" keyword (case-insensitive)
            baseline_keywords = {'baseline', 'rest', 'iti', 'inter-trial'}
            requested_lower = {c.lower() for c in cs.conditions}
            baseline_requested = bool(requested_lower & baseline_keywords)
            
            # Filter out baseline keywords from conditions to process
            conditions_to_process = [c for c in cs.conditions if c.lower() not in baseline_keywords]
            
            # Validate remaining conditions exist in events file
            for cond in conditions_to_process:
                if cond not in all_conditions:
                    raise PreprocessingError(
                        f"Condition '{cond}' not found in events file. "
                        f"Available conditions: {all_conditions}\n"
                        f"Tip: Use 'baseline' to select inter-trial intervals."
                    )
        else:
            # Process all conditions
            conditions_to_process = all_conditions
        
        # Also check include_baseline flag
        baseline_requested = baseline_requested or cs.include_baseline
        
        # Create volume times (center of each volume)
        volume_times = np.arange(self.n_volumes) * self.tr + self.tr / 2
        
        # First, compute mask for ALL events (needed for baseline calculation)
        all_events_mask = np.zeros(self.n_volumes, dtype=bool)
        for _, event in events_df.iterrows():
            onset = event['onset']
            duration = event['duration']
            
            # Apply transition buffer for baseline calculation too
            buffered_onset = onset + cs.transition_buffer
            buffered_end = onset + duration - cs.transition_buffer
            
            if buffered_end <= buffered_onset:
                continue
            
            in_event = (volume_times >= buffered_onset) & (volume_times < buffered_end)
            all_events_mask |= in_event
        
        # Create mask for each requested condition
        self.condition_masks = {}  # Effective masks (condition & global)
        self.raw_condition_masks = {}  # Raw condition timing (for visualization)
        
        for condition in conditions_to_process:
            # Start with all False
            cond_mask = np.zeros(self.n_volumes, dtype=bool)
            
            # Get events for this condition
            cond_events = events_df[events_df[condition_col] == condition]
            
            for _, event in cond_events.iterrows():
                onset = event['onset']
                duration = event['duration']
                
                # Apply transition buffer
                buffered_onset = onset + cs.transition_buffer
                buffered_end = onset + duration - cs.transition_buffer
                
                if buffered_end <= buffered_onset:
                    # Buffer too large, skip this event
                    continue
                
                # Find volumes within this event
                in_event = (volume_times >= buffered_onset) & (volume_times < buffered_end)
                cond_mask |= in_event
            
            # Store raw condition timing (before applying global mask)
            self.raw_condition_masks[condition] = cond_mask.copy()
            
            # Apply the global censoring mask to get effective mask
            cond_mask &= self.mask
            
            self.condition_masks[condition] = cond_mask
            n_volumes_cond = np.sum(cond_mask)
            
            self._logger.info(
                f"Condition '{condition}': {n_volumes_cond} volumes "
                f"({100 * n_volumes_cond / self.n_volumes:.1f}%)"
            )
        
        # Add baseline if requested (via --conditions baseline or --include-baseline)
        if baseline_requested:
            # Baseline = not in any condition from events file (raw timing)
            raw_baseline_mask = ~all_events_mask
            self.raw_condition_masks['baseline'] = raw_baseline_mask
            
            # Effective baseline = raw baseline & global mask
            baseline_mask = raw_baseline_mask & self.mask
            self.condition_masks['baseline'] = baseline_mask
            
            n_baseline = np.sum(baseline_mask)
            self._logger.info(
                f"Condition 'baseline' (inter-trial intervals): {n_baseline} volumes "
                f"({100 * n_baseline / self.n_volumes:.1f}%)"
            )
        
        return self.condition_masks
    
    def apply_custom_mask(self, mask_file: Path) -> int:
        """Apply user-provided censoring mask.
        
        Args:
            mask_file: Path to TSV file with 'censor' column (1=keep, 0=drop).
            
        Returns:
            Number of volumes marked for censoring.
        """
        if mask_file is None:
            return 0
        
        mask_file = Path(mask_file)
        if not mask_file.exists():
            raise PreprocessingError(f"Custom mask file not found: {mask_file}")
        
        # Load mask
        mask_df = pd.read_csv(mask_file, sep='\t')
        
        if 'censor' not in mask_df.columns:
            raise PreprocessingError(
                f"Custom mask file must have 'censor' column. "
                f"Found columns: {list(mask_df.columns)}"
            )
        
        custom_mask = mask_df['censor'].values.astype(bool)
        
        if len(custom_mask) != self.n_volumes:
            raise PreprocessingError(
                f"Custom mask length ({len(custom_mask)}) doesn't match "
                f"number of volumes ({self.n_volumes})"
            )
        
        # Apply to global mask
        n_censored = 0
        for i, keep in enumerate(custom_mask):
            if not keep:
                if self.mask[i]:
                    n_censored += 1
                self.mask[i] = False
                self._add_censoring_reason(i, "custom_mask")
        
        self._logger.info(f"Temporal censoring: marked {n_censored} volumes from custom mask")
        return n_censored
    
    def _add_censoring_reason(self, volume_idx: int, reason: str) -> None:
        """Record reason for censoring a volume."""
        if volume_idx not in self.censoring_log:
            self.censoring_log[volume_idx] = []
        if reason not in self.censoring_log[volume_idx]:
            self.censoring_log[volume_idx].append(reason)
    
    def get_mask(self) -> np.ndarray:
        """Return the combined boolean mask.
        
        Returns:
            Boolean array where True = keep, False = censor.
        """
        return self.mask.copy()
    
    def validate(self) -> None:
        """Check if enough volumes remain after censoring.
        
        When condition selection is active, validates that each condition
        has sufficient volumes. Otherwise validates the global mask.
        
        Raises:
            PreprocessingError: If too few volumes remain.
        """
        # Track warnings for reporting
        self._warnings: List[str] = []
        
        # When condition selection is active, validate each condition
        if self.condition_masks:
            for cond_name, cond_mask in self.condition_masks.items():
                n_retained = np.sum(cond_mask)
                fraction_retained = n_retained / self.n_volumes
                
                if n_retained < self.config.min_volumes_retained:
                    warning_msg = (
                        f"⚠️ LOW VOLUME COUNT for condition '{cond_name}': only {n_retained} volumes "
                        f"(recommended minimum: {self.config.min_volumes_retained}). "
                        f"Results may be unreliable."
                    )
                    self._warnings.append(warning_msg)
                    self._logger.warning(warning_msg)
                
                if fraction_retained < self.config.min_fraction_retained:
                    warning_msg = (
                        f"⚠️ LOW RETENTION RATE for condition '{cond_name}': {fraction_retained:.1%} "
                        f"(recommended minimum: {self.config.min_fraction_retained:.0%}). "
                        f"Results may be unreliable."
                    )
                    self._warnings.append(warning_msg)
                    self._logger.warning(warning_msg)
                elif fraction_retained < self.config.warn_fraction_retained:
                    self._logger.warning(
                        f"⚠️ Only {fraction_retained:.1%} of volumes retained for condition "
                        f"'{cond_name}'. Interpret results with caution."
                    )
        else:
            # Validate global mask
            n_retained = np.sum(self.mask)
            fraction_retained = n_retained / self.n_volumes
            
            if n_retained < self.config.min_volumes_retained:
                warning_msg = (
                    f"⚠️ LOW VOLUME COUNT after censoring: only {n_retained} volumes remaining "
                    f"(recommended minimum: {self.config.min_volumes_retained}). "
                    f"Results may be unreliable."
                )
                self._warnings.append(warning_msg)
                self._logger.warning(warning_msg)
            
            if fraction_retained < self.config.min_fraction_retained:
                warning_msg = (
                    f"⚠️ LOW RETENTION RATE after censoring: {fraction_retained:.1%} remaining "
                    f"(recommended minimum: {self.config.min_fraction_retained:.0%}). "
                    f"Results may be unreliable."
                )
                self._warnings.append(warning_msg)
                self._logger.warning(warning_msg)
            elif fraction_retained < self.config.warn_fraction_retained:
                self._logger.warning(
                    f"⚠️ Only {fraction_retained:.1%} of volumes retained after censoring. "
                    f"Interpret results with caution."
                )
    
    def apply_to_image(
        self,
        img: nib.Nifti1Image,
        condition: Optional[str] = None,
    ) -> nib.Nifti1Image:
        """Apply censoring mask to 4D image.
        
        Args:
            img: 4D NIfTI image.
            condition: If specified, use condition-specific mask.
            
        Returns:
            New image with censored volumes removed.
        """
        data = img.get_fdata()
        
        if data.ndim != 4:
            raise PreprocessingError(
                f"Expected 4D image, got {data.ndim}D"
            )
        
        # Select mask
        if condition is not None:
            if condition not in self.condition_masks:
                raise PreprocessingError(
                    f"Condition '{condition}' not found. "
                    f"Available: {list(self.condition_masks.keys())}"
                )
            mask = self.condition_masks[condition]
        else:
            mask = self.mask
        
        # Apply mask
        censored_data = data[..., mask]
        
        # Create new image
        new_img = nib.Nifti1Image(censored_data, img.affine, img.header)
        
        # Update header for new number of volumes
        new_img.header.set_data_shape(censored_data.shape)
        
        n_original = data.shape[-1]
        n_retained = censored_data.shape[-1]
        self._logger.debug(
            f"Applied censoring: {n_original} -> {n_retained} volumes"
        )
        
        return new_img
    
    def apply_to_confounds(
        self,
        df: pd.DataFrame,
        condition: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply censoring mask to confounds DataFrame.
        
        Args:
            df: Confounds DataFrame.
            condition: If specified, use condition-specific mask.
            
        Returns:
            New DataFrame with censored rows removed.
        """
        # Select mask
        if condition is not None:
            if condition not in self.condition_masks:
                raise PreprocessingError(
                    f"Condition '{condition}' not found. "
                    f"Available: {list(self.condition_masks.keys())}"
                )
            mask = self.condition_masks[condition]
        else:
            mask = self.mask
        
        return df.iloc[mask].reset_index(drop=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Return censoring statistics for reporting.
        
        Returns:
            Dictionary with censoring summary statistics.
        """
        n_global_retained = np.sum(self.mask)
        n_global_censored = self.n_volumes - n_global_retained
        
        # Count by reason (for global censoring: motion, initial drop, etc.)
        reason_counts = {}
        for reasons in self.censoring_log.values():
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # When condition selection is active, report condition-specific stats as "retained"
        # because those are the volumes actually used for connectivity analysis
        if self.condition_masks:
            # Calculate total volumes used across all conditions
            # (union of all condition masks - volumes used at least once)
            combined_mask = np.zeros(self.n_volumes, dtype=bool)
            for mask in self.condition_masks.values():
                combined_mask |= mask
            n_retained = int(np.sum(combined_mask))
            n_censored = self.n_volumes - n_retained
            fraction_retained = n_retained / self.n_volumes
        else:
            n_retained = int(n_global_retained)
            n_censored = int(n_global_censored)
            fraction_retained = float(n_global_retained / self.n_volumes)
        
        summary = {
            'enabled': self.config.enabled,
            'n_original': self.n_volumes,
            'n_retained': n_retained,
            'n_censored': n_censored,
            'fraction_retained': fraction_retained,
            'reason_counts': reason_counts,
            'mask': self.mask.tolist(),
            # Also include global censoring stats separately
            'global_censoring': {
                'n_retained': int(n_global_retained),
                'n_censored': int(n_global_censored),
                'fraction_retained': float(n_global_retained / self.n_volumes),
            },
        }
        
        # Add condition info if applicable
        if self.condition_masks:
            summary['conditions'] = {}
            for name, mask in self.condition_masks.items():
                raw_mask = self.raw_condition_masks.get(name, mask)
                summary['conditions'][name] = {
                    'n_volumes': int(np.sum(mask)),
                    'fraction': float(np.sum(mask) / self.n_volumes),
                    'mask': mask.tolist(),  # Effective mask (for connectivity)
                    'raw_mask': raw_mask.tolist(),  # Raw timing (for visualization)
                }
        
        # Add any warnings that were generated during validation
        if hasattr(self, '_warnings') and self._warnings:
            summary['warnings'] = self._warnings
        
        return summary
    
    def get_censoring_entity(self) -> Optional[str]:
        """Generate BIDS-style entity string for censoring.
        
        Returns:
            Entity string like "drop4fd05" or None if no censoring.
        """
        if not self.config.enabled:
            return None
        
        parts = []
        
        # Initial drop
        if self.config.drop_initial_volumes > 0:
            parts.append(f"drop{self.config.drop_initial_volumes}")
        
        # Motion censoring
        if self.config.motion_censoring.enabled:
            # Format threshold without decimal (0.5 -> "05")
            fd_str = str(self.config.motion_censoring.fd_threshold).replace('.', '')
            parts.append(f"fd{fd_str}")
        
        # Custom mask
        if self.config.custom_mask_file:
            parts.append("custom")
        
        if not parts:
            return None
        
        return "".join(parts)


def load_events_file(
    events_path: Path,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Load BIDS events TSV file.
    
    Args:
        events_path: Path to events.tsv file.
        logger: Optional logger.
        
    Returns:
        Events DataFrame.
    """
    _logger = logger or logging.getLogger(__name__)
    
    if not events_path.exists():
        raise PreprocessingError(f"Events file not found: {events_path}")
    
    events_df = pd.read_csv(events_path, sep='\t')
    _logger.debug(f"Loaded events file: {events_path.name} ({len(events_df)} events)")
    
    return events_df


def find_events_file(
    func_path: Path,
    layout: "BIDSLayout",
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Find BIDS events file matching a functional file using BIDSLayout.
    
    Uses BIDSLayout to properly query for events.tsv in the raw BIDS dataset.
    BIDS allows two types of events files:
    
    1. Subject-specific: sub-01_task-rest_events.tsv (in sub-01/func/)
    2. Dataset-wide: task-rest_events.tsv (in root, shared by all subjects)
    
    This function first queries without subject filter. If multiple matches
    are found, it narrows down by adding subject (and session/run if needed).
    
    Args:
        func_path: Path to functional file (from fMRIPrep derivatives).
        layout: BIDSLayout object with access to raw BIDS data.
        logger: Optional logger.
        
    Returns:
        Path to events file, or None if not found.
    """
    _logger = logger or logging.getLogger(__name__)
    
    # Extract entities from functional filename
    # e.g., sub-01_task-rest_space-MNI_desc-preproc_bold.nii.gz
    func_name = func_path.name
    
    # Parse entities
    entities = {}
    for part in func_name.split('_'):
        if '-' in part:
            key, value = part.split('-', 1)
            entities[key] = value
    
    if 'sub' not in entities or 'task' not in entities:
        _logger.warning("Cannot find events file: missing sub or task entity")
        return None
    
    # Clean up task name (remove any suffix contamination)
    task_name = entities['task'].replace('_bold.nii.gz', '').replace('_bold.nii', '')
    
    # Build base query WITHOUT subject (to find dataset-wide events files)
    base_query = {
        'task': task_name,
        'suffix': 'events',
        'extension': '.tsv',
    }
    
    # Add session if present
    if 'ses' in entities:
        base_query['session'] = entities['ses']
    
    # Add run if present
    if 'run' in entities:
        base_query['run'] = entities['run']
    
    _logger.debug(f"Querying BIDSLayout for events file (without subject): {base_query}")
    
    try:
        # First query WITHOUT subject - catches dataset-wide events files
        events_files = layout.get(**base_query)
        
        if len(events_files) == 1:
            # Single match - use it (likely dataset-wide events file)
            events_path = Path(events_files[0].path)
            _logger.debug(f"Found single events file: {events_path}")
            return events_path
        
        elif len(events_files) > 1:
            # Multiple matches - narrow down by subject
            query_with_subject = {**base_query, 'subject': entities['sub']}
            _logger.debug(f"Multiple events files found, narrowing by subject: {query_with_subject}")
            
            events_files = layout.get(**query_with_subject)
            
            if events_files:
                events_path = Path(events_files[0].path)
                _logger.debug(f"Found subject-specific events file: {events_path}")
                return events_path
        
        # No matches - try without run entity (events may be shared across runs)
        if 'run' in base_query:
            query_no_run = {k: v for k, v in base_query.items() if k != 'run'}
            _logger.debug(f"No events file found, trying without run: {query_no_run}")
            
            events_files = layout.get(**query_no_run)
            
            if len(events_files) == 1:
                events_path = Path(events_files[0].path)
                _logger.debug(f"Found events file (ignoring run): {events_path}")
                return events_path
            
            elif len(events_files) > 1:
                # Multiple - narrow by subject
                query_no_run['subject'] = entities['sub']
                events_files = layout.get(**query_no_run)
                
                if events_files:
                    events_path = Path(events_files[0].path)
                    _logger.debug(f"Found subject-specific events file (ignoring run): {events_path}")
                    return events_path
        
        _logger.debug(f"No events file found for task '{task_name}'")
        return None
            
    except Exception as e:
        _logger.warning(f"Error querying BIDSLayout for events: {e}")
        return None
    except Exception as e:
        _logger.warning(f"Error querying BIDSLayout for events: {e}")
        return None
