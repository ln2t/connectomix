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
                self._add_censoring_reason(i, f"motion_fd>{mc.fd_threshold}")
        
        self._logger.info(
            f"Temporal censoring: marked {n_censored} volumes for motion "
            f"(FD > {mc.fd_threshold}mm)"
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
        
        # Determine which conditions to process
        if cs.conditions:
            # User specified conditions
            conditions_to_process = cs.conditions
            # Validate
            for cond in conditions_to_process:
                if cond not in all_conditions:
                    raise PreprocessingError(
                        f"Condition '{cond}' not found in events file. "
                        f"Available conditions: {all_conditions}"
                    )
        else:
            # Process all conditions
            conditions_to_process = all_conditions
        
        # Create volume times (center of each volume)
        volume_times = np.arange(self.n_volumes) * self.tr + self.tr / 2
        
        # Create mask for each condition
        self.condition_masks = {}
        
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
            
            # Also apply the global censoring mask
            cond_mask &= self.mask
            
            self.condition_masks[condition] = cond_mask
            n_volumes_cond = np.sum(cond_mask)
            
            self._logger.info(
                f"Condition '{condition}': {n_volumes_cond} volumes "
                f"({100 * n_volumes_cond / self.n_volumes:.1f}%)"
            )
        
        # Optionally include baseline (time not in any condition)
        if cs.include_baseline:
            # Baseline = not in any condition
            any_condition = np.zeros(self.n_volumes, dtype=bool)
            for _, cond_mask in self.condition_masks.items():
                any_condition |= cond_mask
            
            baseline_mask = ~any_condition & self.mask
            self.condition_masks['baseline'] = baseline_mask
            
            n_baseline = np.sum(baseline_mask)
            self._logger.info(
                f"Condition 'baseline': {n_baseline} volumes "
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
        
        Raises:
            PreprocessingError: If too few volumes remain.
        """
        n_retained = np.sum(self.mask)
        fraction_retained = n_retained / self.n_volumes
        
        if n_retained < self.config.min_volumes_retained:
            raise PreprocessingError(
                f"Too few volumes after censoring: {n_retained} volumes remaining "
                f"(minimum required: {self.config.min_volumes_retained}). "
                f"Consider relaxing censoring parameters."
            )
        
        if fraction_retained < self.config.min_fraction_retained:
            raise PreprocessingError(
                f"Too few volumes after censoring: {fraction_retained:.1%} remaining "
                f"(minimum required: {self.config.min_fraction_retained:.0%}). "
                f"Consider relaxing censoring parameters."
            )
        
        if fraction_retained < self.config.warn_fraction_retained:
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
        n_retained = np.sum(self.mask)
        n_censored = self.n_volumes - n_retained
        
        # Count by reason
        reason_counts = {}
        for reasons in self.censoring_log.values():
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        summary = {
            'enabled': self.config.enabled,
            'n_original': self.n_volumes,
            'n_retained': int(n_retained),
            'n_censored': int(n_censored),
            'fraction_retained': float(n_retained / self.n_volumes),
            'reason_counts': reason_counts,
            'mask': self.mask.tolist(),
        }
        
        # Add condition info if applicable
        if self.condition_masks:
            summary['conditions'] = {
                name: {
                    'n_volumes': int(np.sum(mask)),
                    'fraction': float(np.sum(mask) / self.n_volumes),
                }
                for name, mask in self.condition_masks.items()
            }
        
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
    
    Uses BIDSLayout to properly query for events.tsv in the raw BIDS dataset,
    matching the subject, session, task, and run entities from the functional file.
    
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
    
    # Build query for BIDSLayout
    query = {
        'subject': entities['sub'],
        'task': entities['task'].replace('_bold.nii.gz', '').replace('_bold.nii', ''),
        'suffix': 'events',
        'extension': '.tsv',
    }
    
    # Add session if present
    if 'ses' in entities:
        query['session'] = entities['ses']
    
    # Add run if present
    if 'run' in entities:
        query['run'] = entities['run']
    
    _logger.debug(f"Querying BIDSLayout for events file: {query}")
    
    try:
        # Query the layout - this will search in raw BIDS directory
        events_files = layout.get(**query)
        
        if events_files:
            events_path = Path(events_files[0].path)
            _logger.debug(f"Found events file via BIDSLayout: {events_path}")
            return events_path
        else:
            _logger.debug(f"No events file found matching query: {query}")
            
            # Try without run entity (sometimes events are shared across runs)
            if 'run' in query:
                query_no_run = {k: v for k, v in query.items() if k != 'run'}
                events_files = layout.get(**query_no_run)
                if events_files:
                    events_path = Path(events_files[0].path)
                    _logger.debug(f"Found events file (ignoring run): {events_path}")
                    return events_path
            
            return None
            
    except Exception as e:
        _logger.warning(f"Error querying BIDSLayout for events: {e}")
        return None
