# Claude Coding Agent Instructions for Connectomix Rewrite

## Project Context

You are rewriting Connectomix, a neuroimaging analysis tool for functional connectivity analysis. The complete functional specification is in `prompt.txt`. This document provides Claude-specific implementation guidance.

## Development Approach

### Code Organization Strategy

Create a clean, modular architecture with clear separation of concerns:

```
connectomix/
├── __init__.py
├── __main__.py              # Entry point for python -m connectomix
├── cli.py                   # Command-line interface and argument parsing
├── config/
│   ├── __init__.py
│   ├── loader.py            # Load and parse configuration files
│   ├── validator.py         # Validate configuration parameters
│   ├── defaults.py          # Default values and schemas
│   └── strategies.py        # Predefined denoising strategies
├── core/
│   ├── __init__.py
│   ├── participant.py       # Participant-level pipeline orchestration
│   ├── group.py             # Group-level pipeline orchestration
│   └── version.py           # Version string
├── preprocessing/
│   ├── __init__.py
│   ├── resampling.py        # Image resampling functions
│   ├── denoising.py         # Signal cleaning and filtering
│   └── canica.py            # CanICA atlas generation
├── connectivity/
│   ├── __init__.py
│   ├── extraction.py        # Time series extraction (seeds, ROIs)
│   ├── seed_to_voxel.py     # Seed-to-voxel analysis
│   ├── roi_to_voxel.py      # ROI-to-voxel analysis
│   ├── seed_to_seed.py      # Seed-to-seed analysis
│   └── roi_to_roi.py        # ROI-to-ROI analysis
├── statistics/
│   ├── __init__.py
│   ├── glm.py               # GLM fitting and contrasts
│   ├── permutation.py       # Permutation testing
│   ├── thresholding.py      # Multiple comparison correction
│   └── clustering.py        # Cluster analysis and labeling
├── io/
│   ├── __init__.py
│   ├── bids.py              # BIDS layout and querying
│   ├── readers.py           # Read various file formats
│   ├── writers.py           # Write outputs with BIDS naming
│   └── paths.py             # Path construction and validation
├── utils/
│   ├── __init__.py
│   ├── logging.py           # Logging configuration and utilities
│   ├── validation.py        # Input validation functions
│   ├── matrix.py            # Matrix operations (sym to vec, etc.)
│   ├── visualization.py     # Plotting functions
│   ├── geometry.py          # Geometric consistency checking
│   └── reports.py           # HTML report generation
└── data/
    ├── __init__.py
    ├── atlases.py           # Atlas loading and management
    └── nilearn_data/        # Bundled atlas files
        ├── aal_SPM12/
        ├── fsl/
        └── schaefer_2018/
```

### Python Best Practices

**Use Type Hints Everywhere:**
```python
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from nibabel import Nifti1Image

def resample_image(
    img: Nifti1Image,
    reference: Nifti1Image,
    interpolation: str = "continuous"
) -> Nifti1Image:
    """Resample image to reference space."""
    ...
```

**Use Dataclasses for Configuration:**
```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ParticipantConfig:
    """Configuration for participant-level analysis."""
    
    # BIDS entities
    subject: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    sessions: Optional[List[str]] = None
    runs: Optional[List[str]] = None
    spaces: Optional[List[str]] = None
    
    # Preprocessing
    confounds: List[str] = field(default_factory=lambda: [
        "trans_x", "trans_y", "trans_z", 
        "rot_x", "rot_y", "rot_z", "csf_wm"
    ])
    high_pass: float = 0.01
    low_pass: float = 0.08
    ica_aroma: bool = False
    
    # Analysis method
    method: str = "roiToRoi"
    
    # Method-specific parameters
    seeds_file: Optional[Path] = None
    radius: float = 5.0
    roi_masks: Optional[List[Path]] = None
    atlas: str = "schaeffer100"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.method not in ["seedToVoxel", "roiToVoxel", "seedToSeed", "roiToRoi"]:
            raise ValueError(f"Invalid method: {self.method}")
        # More validation...
```

**Use pathlib.Path Instead of Strings:**
```python
from pathlib import Path

def load_config(config_path: Path) -> dict:
    """Load configuration from file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix == ".json":
        with config_path.open() as f:
            return json.load(f)
    elif config_path.suffix in [".yaml", ".yml"]:
        with config_path.open() as f:
            return yaml.safe_load(f)
```

**Use Logging Module:**
```python
import logging
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors."""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with color support."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        '%(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger('connectomix')
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
```

**Use Context Managers:**
```python
from contextlib import contextmanager
import time

@contextmanager
def timer(logger: logging.Logger, message: str):
    """Context manager for timing operations."""
    start = time.time()
    logger.info(f"Starting: {message}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {message} ({elapsed:.2f}s)")

# Usage:
with timer(logger, "Denoising functional images"):
    denoise_images(...)
```

### Dependency Management

**Core Dependencies:**
```python
# requirements.txt
nibabel>=5.2.0
nilearn>=0.10.3
numpy>=1.24.0
pandas>=2.0.0
pybids>=0.16.4
PyYAML>=6.0
scipy>=1.10.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
colorama>=0.4.6
tqdm>=4.65.0  # Progress bars
```

**Import Organization:**
```python
# Standard library
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# Third-party
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, maskers, glm
from bids import BIDSLayout

# Local
from connectomix.config import ParticipantConfig
from connectomix.utils.logging import setup_logging
from connectomix.io.bids import build_output_path
```

## Implementation Guidelines

### 1. CLI Implementation

Use `argparse` with clear help messages:

```python
import argparse
from pathlib import Path

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Connectomix: Functional connectivity analysis from fMRIPrep outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Participant-level analysis with default settings
  connectomix /data/bids /data/derivatives/connectomix participant
  
  # Group-level analysis with custom config
  connectomix /data/bids /data/derivatives/connectomix group -c config.yaml
  
  # Specify fMRIPrep location
  connectomix /data/bids /data/derivatives/connectomix participant \\
    --derivatives fmriprep=/data/derivatives/fmriprep
        """
    )
    
    # Required arguments
    parser.add_argument(
        "bids_dir",
        type=Path,
        help="Path to BIDS dataset directory"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output directory for derivatives"
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant", "group"],
        help="Analysis level to perform"
    )
    
    # Optional arguments
    parser.add_argument(
        "-d", "--derivatives",
        action="append",
        help="Derivatives in form name=path (e.g., fmriprep=/path/to/fmriprep)"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to configuration file (.json, .yaml, .yml)"
    )
    parser.add_argument(
        "-p", "--participant_label",
        help="Process only this participant"
    )
    parser.add_argument(
        "-t", "--task",
        help="Process only this task"
    )
    parser.add_argument(
        "-s", "--session",
        help="Process only this session"
    )
    parser.add_argument(
        "--denoising",
        help="Use predefined denoising strategy"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    return parser
```

### 2. Configuration Loading and Validation

**Robust Configuration Loading:**

```python
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
from dataclasses import asdict

def load_config_file(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open() as f:
        if path.suffix == ".json":
            return json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

def config_from_dict(
    data: Dict[str, Any],
    config_class: type
) -> Any:
    """Create config dataclass from dictionary."""
    
    # Filter to only include fields defined in the dataclass
    valid_fields = {f.name for f in config_class.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}
    
    return config_class(**filtered_data)
```

**Configuration Validation:**

```python
from typing import Any, List
import logging

class ConfigValidator:
    """Validate configuration parameters."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.errors: List[str] = []
    
    def validate_alpha(self, value: float, name: str) -> bool:
        """Validate alpha value is in [0, 1]."""
        if not 0 <= value <= 1:
            self.errors.append(f"{name} must be between 0 and 1, got {value}")
            return False
        return True
    
    def validate_positive(self, value: float, name: str) -> bool:
        """Validate value is positive."""
        if value <= 0:
            self.errors.append(f"{name} must be positive, got {value}")
            return False
        return True
    
    def validate_file_exists(self, path: Path, name: str) -> bool:
        """Validate file exists."""
        if not path.exists():
            self.errors.append(f"{name} file not found: {path}")
            return False
        return True
    
    def validate_choice(self, value: Any, choices: List[Any], name: str) -> bool:
        """Validate value is in allowed choices."""
        if value not in choices:
            self.errors.append(
                f"{name} must be one of {choices}, got {value}"
            )
            return False
        return True
    
    def raise_if_errors(self) -> None:
        """Raise ValueError if any validation errors occurred."""
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in self.errors
            )
            raise ValueError(error_msg)
```

### 3. BIDS I/O Operations

**BIDS Layout Setup:**

```python
from bids import BIDSLayout
from pathlib import Path
from typing import Dict, Optional

def create_bids_layout(
    bids_dir: Path,
    derivatives: Optional[Dict[str, Path]] = None,
    logger: logging.Logger = None
) -> BIDSLayout:
    """Create BIDS layout with derivatives."""
    
    if logger:
        logger.info(f"Creating BIDS layout for {bids_dir}")
    
    # Validate BIDS directory
    dataset_desc = bids_dir / "dataset_description.json"
    if not dataset_desc.exists():
        raise ValueError(
            f"Not a valid BIDS dataset: {bids_dir}\n"
            f"Missing dataset_description.json"
        )
    
    # Create layout
    layout = BIDSLayout(
        bids_dir,
        derivatives=derivatives or {},
        validate=False  # Skip validation for speed
    )
    
    if logger:
        logger.info(f"Found {len(layout.get_subjects())} subjects")
    
    return layout
```

**BIDS Path Building:**

```python
from typing import Dict, Any, Optional
from pathlib import Path

def build_bids_path(
    output_dir: Path,
    entities: Dict[str, Any],
    suffix: str,
    extension: str,
    level: str = "participant"
) -> Path:
    """Build BIDS-compliant output path."""
    
    # Start with output directory
    if level == "participant":
        path = output_dir / f"sub-{entities['subject']}"
        if 'session' in entities:
            path = path / f"ses-{entities['session']}"
    else:  # group
        path = output_dir / "group"
        if 'method' in entities:
            path = path / entities['method']
        if 'analysis' in entities:
            path = path / entities['analysis']
        if 'session' in entities:
            path = path / f"ses-{entities['session']}"
    
    # Create filename from entities
    parts = []
    entity_order = [
        'subject', 'session', 'task', 'run', 'space', 
        'method', 'seed', 'data', 'analysis', 'desc',
        'threshold', 'stat'
    ]
    
    for entity_name in entity_order:
        if entity_name in entities and entities[entity_name] is not None:
            parts.append(f"{entity_name}-{entities[entity_name]}")
    
    parts.append(suffix)
    filename = "_".join(parts) + extension
    
    # Ensure directory exists
    path.mkdir(parents=True, exist_ok=True)
    
    return path / filename
```

### 4. Preprocessing Implementation

**Resampling:**

```python
from nilearn import image
import nibabel as nib
from pathlib import Path
import numpy as np

def resample_to_reference(
    img_path: Path,
    reference_path: Path,
    output_path: Path,
    logger: logging.Logger
) -> Path:
    """Resample image to reference space."""
    
    # Skip if already exists
    if output_path.exists():
        logger.debug(f"Resampled file exists, skipping: {output_path}")
        return output_path
    
    logger.info(f"Resampling {img_path.name}")
    
    # Load images
    img = nib.load(img_path)
    ref = nib.load(reference_path)
    
    # Resample
    resampled = image.resample_to_img(
        img,
        ref,
        interpolation='continuous'
    )
    
    # Round affine to avoid numerical precision issues
    resampled = nib.Nifti1Image(
        resampled.get_fdata(),
        np.round(resampled.affine, decimals=6),
        resampled.header
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(resampled, output_path)
    
    return output_path
```

**Denoising:**

```python
from nilearn import image
import pandas as pd
from pathlib import Path
from typing import List
import json

def denoise_image(
    img_path: Path,
    confounds_path: Path,
    confound_names: List[str],
    high_pass: float,
    low_pass: float,
    output_path: Path,
    logger: logging.Logger,
    overwrite: bool = True
) -> Path:
    """Denoise functional image."""
    
    # Skip if exists and not overwriting
    if output_path.exists() and not overwrite:
        logger.debug(f"Denoised file exists, skipping: {output_path}")
        return output_path
    
    logger.info(f"Denoising {img_path.name}")
    
    # Load confounds
    confounds_df = pd.read_csv(confounds_path, sep='\t')
    
    # Validate confound columns exist
    missing = set(confound_names) - set(confounds_df.columns)
    if missing:
        raise ValueError(
            f"Confounds not found in {confounds_path}: {missing}"
        )
    
    # Extract selected confounds
    confounds = confounds_df[confound_names].values
    
    # Clean image
    cleaned = image.clean_img(
        img_path,
        confounds=confounds,
        high_pass=high_pass,
        low_pass=low_pass,
        standardize=True,
        detrend=True,
        t_r=None  # Will be read from image header
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_filename(output_path)
    
    # Save sidecar with confound info
    sidecar_path = output_path.with_suffix('.json')
    sidecar_data = {
        'Confounds': confound_names,
        'HighPass': high_pass,
        'LowPass': low_pass,
        'Standardized': True,
        'Detrended': True
    }
    with sidecar_path.open('w') as f:
        json.dump(sidecar_data, f, indent=2)
    
    return output_path
```

### 5. Progress Tracking

Use `tqdm` for progress bars:

```python
from tqdm import tqdm
from typing import Iterable, Any

def process_subjects(subjects: List[str], func, **kwargs):
    """Process subjects with progress bar."""
    
    results = []
    for subject in tqdm(subjects, desc="Processing subjects", unit="subject"):
        result = func(subject, **kwargs)
        results.append(result)
    
    return results

# For nested progress:
def permutation_test(n_permutations: int):
    """Run permutation test with progress bar."""
    
    null_dist = []
    for i in tqdm(range(n_permutations), desc="Permutations", unit="perm"):
        # Permute and compute
        max_stat = compute_null_permutation()
        null_dist.append(max_stat)
    
    return np.array(null_dist)
```

### 6. Error Handling Patterns

**Custom Exceptions:**

```python
class ConnectomixError(Exception):
    """Base exception for Connectomix."""
    pass

class ConfigurationError(ConnectomixError):
    """Error in configuration."""
    pass

class BIDSError(ConnectomixError):
    """Error related to BIDS dataset."""
    pass

class PreprocessingError(ConnectomixError):
    """Error during preprocessing."""
    pass

# Usage:
def validate_bids_dir(path: Path):
    """Validate BIDS directory."""
    if not path.exists():
        raise BIDSError(
            f"BIDS directory not found: {path}\n"
            f"Please check the path and try again."
        )
    
    dataset_desc = path / "dataset_description.json"
    if not dataset_desc.exists():
        raise BIDSError(
            f"Not a valid BIDS dataset: {path}\n"
            f"Missing dataset_description.json\n"
            f"See https://bids.neuroimaging.io for BIDS specification."
        )
```

### 7. Testing Strategy (For Future)

While you requested no tests in this rewrite, here's a structure for when you add them later:

```python
# Structure for future testing
"""
tests/
├── conftest.py              # Pytest fixtures
├── test_config.py           # Configuration tests
├── test_preprocessing.py    # Preprocessing tests
├── test_connectivity.py     # Connectivity analysis tests
├── test_statistics.py       # Statistical tests
├── test_io.py              # I/O tests
└── test_integration.py     # End-to-end tests
"""

# Example fixture pattern:
import pytest
from pathlib import Path

@pytest.fixture
def temp_bids_dataset(tmp_path):
    """Create temporary BIDS dataset for testing."""
    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()
    
    # Create minimal BIDS structure
    dataset_desc = {
        "Name": "Test Dataset",
        "BIDSVersion": "1.6.0"
    }
    with (bids_dir / "dataset_description.json").open('w') as f:
        json.dump(dataset_desc, f)
    
    return bids_dir
```

### 8. Documentation Strings

Use Google-style docstrings:

```python
def compute_connectivity_matrix(
    time_series: np.ndarray,
    kind: str = "correlation"
) -> np.ndarray:
    """Compute connectivity matrix from time series.
    
    Args:
        time_series: Time series array of shape (n_timepoints, n_regions)
        kind: Type of connectivity measure. Options: 'correlation', 
            'partial correlation', 'covariance'. Default: 'correlation'
    
    Returns:
        Connectivity matrix of shape (n_regions, n_regions). Symmetric 
        matrix with zeros on diagonal.
    
    Raises:
        ValueError: If time_series has wrong shape or kind is invalid
    
    Examples:
        >>> ts = np.random.randn(100, 5)  # 100 timepoints, 5 regions
        >>> conn = compute_connectivity_matrix(ts)
        >>> conn.shape
        (5, 5)
        >>> np.allclose(conn, conn.T)  # Symmetric
        True
        >>> np.allclose(np.diag(conn), 0)  # Zero diagonal
        True
    """
    if time_series.ndim != 2:
        raise ValueError(
            f"time_series must be 2D, got shape {time_series.shape}"
        )
    
    if kind == "correlation":
        conn = np.corrcoef(time_series.T)
    elif kind == "partial correlation":
        # Implementation
        pass
    else:
        raise ValueError(f"Unknown connectivity kind: {kind}")
    
    # Zero diagonal
    np.fill_diagonal(conn, 0)
    
    return conn
```

### 9. Memory Management

For large datasets, process in chunks:

```python
def process_large_dataset(
    files: List[Path],
    process_func,
    chunk_size: int = 10
):
    """Process large number of files in chunks to manage memory."""
    
    results = []
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        
        # Process chunk
        chunk_results = [process_func(f) for f in chunk]
        results.extend(chunk_results)
        
        # Explicit garbage collection
        import gc
        gc.collect()
    
    return results
```

### 10. Parallel Processing

Use joblib for parallelization:

```python
from joblib import Parallel, delayed
from typing import List, Callable, Any

def parallel_map(
    func: Callable,
    items: List[Any],
    n_jobs: int = -1,
    desc: str = "Processing"
) -> List[Any]:
    """Apply function to items in parallel.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        desc: Description for progress bar
    
    Returns:
        List of results
    """
    from tqdm import tqdm
    
    if n_jobs == 1:
        # Serial processing with progress bar
        return [func(item) for item in tqdm(items, desc=desc)]
    
    # Parallel processing
    return Parallel(n_jobs=n_jobs)(
        delayed(func)(item) 
        for item in tqdm(items, desc=desc)
    )

# Usage:
results = parallel_map(
    process_subject,
    subjects,
    n_jobs=4,
    desc="Processing subjects"
)
```

## Key Implementation Priorities

### Phase 1: Core Infrastructure
1. CLI argument parsing
2. Configuration loading and validation
3. BIDS layout creation
4. Logging setup
5. Output directory structure

### Phase 2: Participant-Level Analysis
1. File discovery and matching
2. Resampling functionality
3. Denoising functionality
4. Time series extraction
5. Connectivity computation (all 4 methods)
6. Output writing with BIDS naming

### Phase 3: Group-Level Analysis
1. Second-level input preparation
2. Design matrix construction
3. GLM fitting
4. Contrast computation
5. Permutation testing
6. Thresholding (uncorrected, FDR, FWE)
7. Cluster analysis
8. Output writing

### Phase 4: Polish
1. Comprehensive error handling
2. Progress bars and user feedback
3. Performance optimization
4. HTML report generation
5. Geometric consistency validation
6. Documentation (README.md)
7. Examples and tutorials

## Common Pitfalls to Avoid

1. **Don't load all data into memory**: Process subjects one at a time
2. **Validate early**: Check configuration and inputs before heavy computation
3. **Handle missing data gracefully**: Not all subjects have all sessions/runs
4. **Avoid hardcoded paths**: Use Path objects and make everything configurable
5. **Round affine matrices**: Avoid numerical precision mismatches
6. **Check for existing outputs**: Don't recompute if file exists (unless forced)
7. **Save metadata**: Always write JSON sidecars with processing parameters
8. **Use appropriate logging levels**: DEBUG for detailed info, INFO for progress, WARNING for issues
9. **Maintain BIDS compliance**: Follow naming conventions strictly
10. **Document assumptions**: Make defaults explicit and well-documented
11. **Ensure geometric consistency**: Check dimensions across ALL subjects, not just selected ones
12. **Generate comprehensive reports**: HTML reports enhance transparency and reproducibility

## Additional Implementation Guidelines for New Features

### 1. Geometric Consistency Checking

**Function to check all functional images:**

```python
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib
import json

def check_geometric_consistency(
    layout: BIDSLayout,
    entities: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[bool, Dict[str, np.ndarray]]:
    """Check if all functional images have consistent geometry.
    
    Args:
        layout: BIDS layout with fMRIPrep derivatives
        entities: BIDS entities for filtering
        logger: Logger instance
    
    Returns:
        Tuple of (is_consistent, geometry_dict)
        where geometry_dict maps subject -> {'shape': shape, 'affine': affine}
    """
    
    # Query ALL functional images (ignore participant_label filter)
    entities_copy = entities.copy()
    entities_copy.pop('subject', None)
    
    func_files = layout.get(
        extension='nii.gz',
        desc='preproc',
        suffix='bold',
        **entities_copy
    )
    
    if not func_files:
        raise ValueError("No functional images found in dataset")
    
    logger.info(f"Checking geometric consistency across {len(func_files)} images")
    
    # Collect geometry information
    geometries = {}
    reference_shape = None
    reference_affine = None
    
    for func_file in func_files:
        img = nib.load(func_file.path)
        shape = img.shape[:3]  # Spatial dimensions only
        affine = np.round(img.affine, decimals=6)
        
        # Get subject
        subject = func_file.entities['subject']
        
        if subject not in geometries:
            geometries[subject] = {'shape': shape, 'affine': affine}
        
        # Set reference from first image
        if reference_shape is None:
            reference_shape = shape
            reference_affine = affine
    
    # Check consistency
    is_consistent = True
    for subject, geom in geometries.items():
        if not np.array_equal(geom['shape'], reference_shape):
            logger.warning(
                f"Shape mismatch: sub-{subject} has {geom['shape']} "
                f"vs reference {reference_shape}"
            )
            is_consistent = False
        
        if not np.allclose(geom['affine'], reference_affine, rtol=1e-5):
            logger.warning(
                f"Affine mismatch: sub-{subject} differs from reference"
            )
            is_consistent = False
    
    if is_consistent:
        logger.info("All images have consistent geometry - no resampling needed")
    else:
        logger.warning("Geometric inconsistencies detected - resampling required")
    
    return is_consistent, geometries

def save_geometry_info(
    img: nib.Nifti1Image,
    output_path: Path,
    reference_path: Optional[Path] = None
) -> None:
    """Save geometric information to JSON file.
    
    Args:
        img: NIfTI image
        output_path: Path for JSON file
        reference_path: Path to reference image if resampling was used
    """
    
    geometry = {
        'Shape': list(img.shape),
        'Affine': np.round(img.affine, decimals=6).tolist(),
        'Voxel_Size_mm': [
            float(img.header.get_zooms()[i]) 
            for i in range(3)
        ],
        'Reference_Used': str(reference_path) if reference_path else None,
        'Creation_Date': datetime.now().isoformat()
    }
    
    with output_path.open('w') as f:
        json.dump(geometry, f, indent=2)

def validate_group_geometry(
    layout: BIDSLayout,
    subjects: List[str],
    entities: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Validate all subjects have consistent geometry for group analysis.
    
    Args:
        layout: BIDS layout with Connectomix derivatives
        subjects: List of subject IDs
        entities: BIDS entities
        logger: Logger instance
    
    Raises:
        ValueError: If geometric inconsistency detected
    """
    
    logger.info("Validating geometric consistency for group analysis")
    
    geometries = {}
    reference_shape = None
    reference_affine = None
    
    for subject in subjects:
        # Find geometry file
        geom_files = layout.get(
            subject=subject,
            desc='geometry',
            extension='json',
            **entities
        )
        
        if not geom_files:
            raise ValueError(
                f"Geometry file not found for sub-{subject}. "
                f"Re-run participant-level analysis."
            )
        
        with open(geom_files[0].path) as f:
            geom = json.load(f)
        
        shape = geom['Shape'][:3]
        affine = np.array(geom['Affine'])
        
        geometries[subject] = {'shape': shape, 'affine': affine}
        
        if reference_shape is None:
            reference_shape = shape
            reference_affine = affine
    
    # Check consistency
    mismatched = []
    for subject, geom in geometries.items():
        if geom['shape'] != reference_shape:
            mismatched.append(
                f"sub-{subject}: shape {geom['shape']} != {reference_shape}"
            )
        
        if not np.allclose(geom['affine'], reference_affine, rtol=1e-5):
            mismatched.append(
                f"sub-{subject}: affine differs from reference"
            )
    
    if mismatched:
        error_msg = (
            "Geometric inconsistency detected for group analysis:\n" +
            "\n".join(f"  - {m}" for m in mismatched) +
            "\n\nAll participants must have identical image geometry. "
            "Re-run participant-level analysis with consistent resampling."
        )
        raise ValueError(error_msg)
    
    logger.info(f"Geometry validation passed for {len(subjects)} subjects")
```

### 2. HTML Report Generation

**HTML Report Generator:**

```python
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

class HTMLReportGenerator:
    """Generate professional HTML reports for Connectomix analyses."""
    
    def __init__(self, version: str):
        self.version = version
        self.sections = []
    
    def add_header(
        self,
        title: str,
        analysis_level: str,
        command_line: str,
        config: Dict[str, Any]
    ) -> None:
        """Add report header with metadata."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <div class="header">
            <h1>{title}</h1>
            <div class="metadata">
                <p><strong>Connectomix Version:</strong> {self.version}</p>
                <p><strong>Analysis Level:</strong> {analysis_level}</p>
                <p><strong>Date:</strong> {timestamp}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Command Line</h2>
            <pre class="command">{command_line}</pre>
        </div>
        
        <div class="section">
            <h2>Configuration</h2>
            {self._format_config(config)}
        </div>
        """
        
        self.sections.append(html)
    
    def add_qa_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Add quality assurance metrics section."""
        
        html = f"""
        <div class="section">
            <h2>Quality Assurance Metrics</h2>
            <table class="qa-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
        """
        
        for metric_name, metric_data in metrics.items():
            html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{metric_data['value']}</td>
                    <td>{metric_data.get('interpretation', '-')}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        self.sections.append(html)
    
    def add_image(
        self,
        title: str,
        fig: plt.Figure,
        description: Optional[str] = None
    ) -> None:
        """Add matplotlib figure as embedded image."""
        
        # Convert figure to base64
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        html = f"""
        <div class="section">
            <h2>{title}</h2>
            {f'<p>{description}</p>' if description else ''}
            <img src="data:image/png;base64,{img_base64}" 
                 alt="{title}" class="result-image">
        </div>
        """
        
        self.sections.append(html)
    
    def add_connectivity_matrix(
        self,
        title: str,
        matrix: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> None:
        """Add connectivity matrix visualization."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
        ax.set_title(title)
        
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
        
        plt.colorbar(im, ax=ax, label='Correlation')
        
        self.add_image(title, fig)
    
    def add_statistical_summary(
        self,
        summary: Dict[str, Any]
    ) -> None:
        """Add statistical summary section."""
        
        html = f"""
        <div class="section">
            <h2>Statistical Summary</h2>
            <div class="summary-grid">
        """
        
        for key, value in summary.items():
            html += f"""
                <div class="summary-item">
                    <div class="summary-label">{key}</div>
                    <div class="summary-value">{value}</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        self.sections.append(html)
    
    def generate(self, output_path: Path) -> None:
        """Generate and save complete HTML report."""
        
        css = self._get_css()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Connectomix Analysis Report</title>
            <style>{css}</style>
        </head>
        <body>
            <div class="container">
                {''.join(self.sections)}
                
                <div class="footer">
                    <p>Generated by Connectomix v{self.version}</p>
                    <p>Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as f:
            f.write(html)
    
    def _format_config(self, config: Dict[str, Any]) -> str:
        """Format configuration as HTML table."""
        
        html = '<table class="config-table">'
        
        for key, value in config.items():
            if isinstance(value, dict):
                html += f'<tr><th colspan="2">{key}</th></tr>'
                for subkey, subvalue in value.items():
                    html += f'<tr><td>&nbsp;&nbsp;{subkey}</td><td>{subvalue}</td></tr>'
            else:
                html += f'<tr><td>{key}</td><td>{value}</td></tr>'
        
        html += '</table>'
        return html
    
    def _get_css(self) -> str:
        """Return CSS styles for report."""
        
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .header {
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2c3e50;
            margin: 0 0 10px 0;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        
        .metadata {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .command {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-left: 4px solid #007bff;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .config-table td:first-child {
            font-weight: bold;
            width: 30%;
        }
        
        .qa-table th {
            background-color: #27ae60;
        }
        
        .result-image {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .summary-item {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        
        .summary-label {
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        
        .summary-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }
        
        .info {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 20px 0;
        }
        """

# Usage example:
def generate_participant_report(
    subject: str,
    config: Dict[str, Any],
    command_line: str,
    qa_metrics: Dict[str, Any],
    connectivity_matrix: Optional[np.ndarray],
    output_path: Path,
    version: str
) -> None:
    """Generate participant-level HTML report."""
    
    report = HTMLReportGenerator(version)
    
    report.add_header(
        title=f"Connectomix Participant Report: sub-{subject}",
        analysis_level="participant",
        command_line=command_line,
        config=config
    )
    
    report.add_qa_metrics(qa_metrics)
    
    if connectivity_matrix is not None:
        report.add_connectivity_matrix(
            title="Connectivity Matrix",
            matrix=connectivity_matrix
        )
    
    report.generate(output_path)
```

### 3. README.md Generation

Create a comprehensive README using templates:

```python
def generate_readme(output_path: Path) -> None:
    """Generate comprehensive README.md documentation."""
    
    readme_content = """
# Connectomix

[Insert badges: License, Version, Coverage, etc.]

## Overview

Connectomix is a BIDS-compliant application for functional connectivity analysis...

[Continue with all sections as outlined in prompt.txt]
"""
    
    with output_path.open('w') as f:
        f.write(readme_content)
```

## Example Entry Point

```python
# connectomix/__main__.py

import sys
import logging
from pathlib import Path

from connectomix.cli import create_parser
from connectomix.utils.logging import setup_logging
from connectomix.core.participant import run_participant_pipeline
from connectomix.core.group import run_group_pipeline
from connectomix.core.version import __version__

def main():
    """Main entry point for Connectomix."""
    
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
        if args.analysis_level == "participant":
            run_participant_pipeline(
                bids_dir=args.bids_dir,
                output_dir=args.output_dir,
                derivatives=args.derivatives,
                config_path=args.config,
                participant_label=args.participant_label,
                task=args.task,
                session=args.session,
                denoising=args.denoising,
                logger=logger
            )
        else:  # group
            run_group_pipeline(
                bids_dir=args.bids_dir,
                output_dir=args.output_dir,
                config_path=args.config,
                logger=logger
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
```

## Summary

This rewrite should produce a modern, maintainable, well-structured Python package that:

- Uses type hints throughout
- Has clear module separation
- Provides excellent user feedback via logging
- Handles errors gracefully with actionable messages
- Follows Python best practices
- Is easy to extend and modify
- Performs efficiently with large datasets
- Produces BIDS-compliant outputs

Focus on code clarity and correctness first, then optimize for performance. The modular structure makes it easy to test components individually and add features incrementally.
