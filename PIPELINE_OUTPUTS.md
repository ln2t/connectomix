# Connectomix Pipeline Outputs: Comprehensive Summary

**Document Version**: 1.0  
**Date**: January 19, 2026  
**Purpose**: Detailed specification of all outputs produced by the Connectomix functional connectivity analysis pipeline for use in implementing group-level analyses.

---

## Table of Contents

1. [Overview](#overview)
2. [Connectivity Measures](#connectivity-measures)
3. [Output File Organization](#output-file-organization)
4. [File Naming Conventions](#file-naming-conventions)
5. [Data Formats and Specifications](#data-formats-and-specifications)
6. [Method-Specific Outputs](#method-specific-outputs)
7. [Temporal Censoring Outputs](#temporal-censoring-outputs)
8. [Quality Control and Reports](#quality-control-and-reports)

---

## Overview

Connectomix processes fMRIPrep-preprocessed functional MRI data to compute functional connectivity matrices at the **participant level**. The pipeline produces:

- **Connectivity matrices** in NumPy format (.npy) with JSON metadata sidecars
- **Denoised functional images** (optional, for further analysis)
- **ROI time series** for reanalysis
- **HTML quality control reports** with comprehensive visualizations

The outputs are organized in a **BIDS-derivatives-compliant** structure, making them suitable for group-level analysis pipelines.

---

## Connectivity Measures

Connectomix computes **four complementary connectivity measures**, each providing different information about functional brain networks:

### 1. **Pearson Correlation** (default)

**Description**: The most commonly used measure. Normalizes covariance by the standard deviations of both regions.

**Properties**:
- Range: -1 to +1 (unitless)
- Symmetric: Yes (ρ_ij = ρ_ji)
- Diagonal: Set to 0 (self-correlation excluded)
- Interpretation: Strength and direction of linear relationship between regions

**Formula**: 
$$\rho_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sigma_i \sigma_j}$$

**Use case**: General connectivity analysis; comparing connectivity across subjects with different signal amplitudes; easily interpretable values.

**File suffix**: `desc-correlation_connectivity`

---

### 2. **Covariance**

**Description**: Raw joint variability between regions, without normalization by standard deviations.

**Properties**:
- Range: Unbounded (real numbers)
- Symmetric: Yes (Cov_ij = Cov_ji)
- Diagonal: Contains region variance (not set to 0)
- Interpretation: Joint variability; retains information about signal magnitude

**Formula**: 
$$\text{Cov}(X_i, X_j) = \frac{1}{n-1}\sum_{t=1}^{n}(x_i^t - \bar{x}_i)(x_j^t - \bar{x}_j)$$

**Use case**: Analyses where variance magnitude is meaningful; preservation of amplitude information across subjects; statistical tests assuming Gaussian distributions.

**File suffix**: `desc-covariance_connectivity`

**Note**: Covariance values are highly sensitive to the signal scaling and preprocessing pipeline. Raw time series variance information is retained.

---

### 3. **Partial Correlation**

**Description**: Measures the relationship between two regions while controlling for the influence of all other regions in the atlas. Reveals direct connections by removing indirect effects.

**Properties**:
- Range: -1 to +1 (unitless)
- Symmetric: Yes (ρ_ij|Z = ρ_ji|Z)
- Diagonal: Set to 0
- Interpretation: Direct functional connectivity after accounting for connections through other regions

**Formula**: 
$$\rho_{ij|Z} = -\frac{\Theta_{ij}}{\sqrt{\Theta_{ii}\Theta_{jj}}}$$
where Θ is the precision matrix (inverse covariance)

**Use case**: Identifying direct functional connections; distinguishing direct effects from indirect mediation through other brain regions; network analyses requiring sparsity.

**File suffix**: `desc-partial-correlation_connectivity`

---

### 4. **Precision (Inverse Covariance)**

**Description**: The inverse of the covariance matrix. Encodes conditional independence relationships.

**Properties**:
- Range: Unbounded (real numbers)
- Symmetric: Yes (Θ_ij = Θ_ji)
- Diagonal: Contains inverse variances
- Sparse structure: Zero entries indicate conditional independence
- Interpretation: Direct statistical dependencies; sparse representation of network structure

**Formula**: 
$$\Theta = \Sigma^{-1}$$

**Use case**: Graph-theoretical network analyses; identifying sparse direct connections; conditional independence structures; machine learning feature extraction.

**File suffix**: `desc-precision_connectivity`

**Note**: Computed using nilearn's robust implementation (handles singular or near-singular covariance matrices). May be sparse depending on atlas resolution and sample size.

---

## Output File Organization

### Directory Structure

```
output_dir/
├── dataset_description.json              # BIDS derivative metadata
├── config/
│   └── backups/
│       └── config_<TIMESTAMP>.json       # Configuration backup with execution timestamp
│
├── sub-<SUBJECT>/
│   ├── func/
│   │   ├── sub-<SUBJECT>[_ses-<SESSION>]_task-<TASK>[_run-<RUN>]_space-<SPACE>
│   │   │   _desc-denoised_bold.nii.gz                          # Denoised functional image
│   │   └── sub-<SUBJECT>[_ses-<SESSION>]_task-<TASK>[_run-<RUN>]_space-<SPACE>
│   │       _desc-denoised_bold.json                            # JSON sidecar
│   │
│   ├── connectivity_data/
│   │   ├── sub-<SUBJECT>[_ses-<SESSION>]_task-<TASK>[_run-<RUN>]_space-<SPACE>
│   │   │   [_condition-<CONDITION>][_atlas-<ATLAS>]_desc-correlation_connectivity.npy
│   │   ├── sub-<SUBJECT>[_ses-<SESSION>]_task-<TASK>[_run-<RUN>]_space-<SPACE>
│   │   │   [_condition-<CONDITION>][_atlas-<ATLAS>]_desc-correlation_connectivity.json
│   │   ├── sub-<SUBJECT>[_ses-<SESSION>]_task-<TASK>[_run-<RUN>]_space-<SPACE>
│   │   │   [_condition-<CONDITION>][_atlas-<ATLAS>]_desc-covariance_connectivity.npy
│   │   ├── ... (other connectivity measures)
│   │   └── sub-<SUBJECT>[_ses-<SESSION>]_task-<TASK>[_run-<RUN>]_space-<SPACE>
│   │       [_condition-<CONDITION>][_atlas-<ATLAS>]_timeseries.npy
│   │
│   ├── figures/                          # Report figures (downloadable from HTML)
│   │   ├── connectivity_correlation.png
│   │   ├── connectivity_covariance.png
│   │   ├── connectivity_partial-correlation.png
│   │   ├── connectivity_precision.png
│   │   ├── connectome_correlation.png         # Glass brain connectivity visualization
│   │   ├── histogram_correlation.png          # Distribution of connectivity values
│   │   ├── confounds_timeseries.png
│   │   ├── confounds_correlation.png
│   │   ├── denoising-histogram.png            # Before/after denoising QA
│   │   └── temporal_censoring.png             # If temporal censoring applied
│   │
│   └── sub-<SUBJECT>[_ses-<SESSION>]_task-<TASK>[_run-<RUN>]_space-<SPACE>
│       [_condition-<CONDITION>][_atlas-<ATLAS>]_desc-schaefer_report.html
│
├── sub-<SUBJECT2>/
│   └── ...
│
└── group/
    └── [future group-level outputs]
```

### Key Characteristics

- **Hierarchical organization**: Participant ID → optional session → subdirectory by data type
- **BIDS compliance**: Entity ordering and naming follows BIDS Derivatives specification
- **Optional entities**: Brackets [ ] indicate conditional entities appearing only when relevant
  - `[_ses-<SESSION>]` - Only if multiple sessions
  - `[_run-<RUN>]` - Only if multiple runs
  - `[_condition-<CONDITION>]` - Only if temporal censoring with conditions is applied
- **JSON sidecars**: All data files (.npy, .nii.gz) have accompanying .json files with metadata
- **Automatic figure generation**: Figures are automatically generated from report templates

---

## File Naming Conventions

### BIDS Entity Ordering

Connectomix follows strict BIDS entity ordering for all output files. The standard order is:

```
sub-<SUBJECT>[_ses-<SESSION>][_task-<TASK>][_run-<RUN>][_space-<SPACE>]
[_condition-<CONDITION>][_atlas-<ATLAS>][_seed-<SEED>][_roi-<ROI>]
[_label-<LABEL>]_desc-<DESC>_<SUFFIX><EXTENSION>
```

**Entity definitions**:
- `sub`: Subject ID (required)
- `ses`: Session number (optional)
- `task`: Task name (optional, for task fMRI)
- `run`: Run number (optional)
- `space`: MNI space name (e.g., MNI152NLin2009cAsym)
- `condition`: Experimental condition (only with temporal censoring)
- `atlas`: Atlas name (e.g., schaefer2018n100)
- `seed`: Seed region (only for seed-based methods)
- `roi`: ROI identifier (only for ROI-based methods)
- `label`: Custom analysis label
- `desc`: Description (required for connectivity outputs)
- `suffix`: File type (e.g., connectivity, timeseries, bold)

### Examples

#### ROI-to-ROI with Schaefer atlas (resting state)
```
sub-01_task-rest_space-MNI152NLin2009cAsym_atlas-schaefer2018n100_desc-correlation_connectivity.npy
sub-01_task-rest_space-MNI152NLin2009cAsym_atlas-schaefer2018n100_desc-correlation_connectivity.json
```

#### Task fMRI with temporal censoring (condition-specific)
```
sub-02_task-faces_condition-face_space-MNI152NLin2009cAsym_atlas-schaefer2018n100_desc-correlation_connectivity.npy
sub-02_task-faces_condition-house_space-MNI152NLin2009cAsym_atlas-schaefer2018n100_desc-correlation_connectivity.npy
sub-02_task-faces_condition-baseline_space-MNI152NLin2009cAsym_atlas-schaefer2018n100_desc-correlation_connectivity.npy
```

#### Seed-to-voxel analysis
```
sub-03_task-rest_space-MNI152NLin2009cAsym_seed-pcc_desc-correlation_connectivity.nii.gz
```

#### All four connectivity measures (same dataset)
```
sub-01_task-rest_atlas-schaefer2018n100_desc-correlation_connectivity.npy
sub-01_task-rest_atlas-schaefer2018n100_desc-covariance_connectivity.npy
sub-01_task-rest_atlas-schaefer2018n100_desc-partial-correlation_connectivity.npy
sub-01_task-rest_atlas-schaefer2018n100_desc-precision_connectivity.npy
```

#### Custom analysis label
```
sub-01_task-rest_label-myanalysis_atlas-schaefer2018n100_desc-correlation_connectivity.npy
```

---

## Data Formats and Specifications

### NumPy Connectivity Matrices (.npy)

**Format**: Binary NumPy compressed array

**Specifications**:
- **Dimensionality**: 2D square matrix (N_regions × N_regions)
- **Data type**: float64 (double precision)
- **Symmetry**: All matrices are symmetric (M_ij = M_ji)
- **Diagonal**: For correlation and partial correlation, diagonal is set to 0
- **Access**: `matrix = np.load('filename.npy')`

**Expected file sizes** (uncompressed):
- Schaefer 100 regions: 80 KB (100 × 100 × 8 bytes)
- Schaefer 200 regions: 320 KB (200 × 200 × 8 bytes)
- AAL 116 regions: ~107 KB (116 × 116 × 8 bytes)

**Value ranges** (typical):
- Correlation: -1.0 to 1.0
- Covariance: varies with signal scaling (typically -10 to 10)
- Partial correlation: -1.0 to 1.0
- Precision: varies with network structure

### ROI Time Series (.npy)

**Filename**: `sub-<ID>[_conditions...]_atlas-<ATLAS>_timeseries.npy`

**Format**: 2D NumPy array

**Specifications**:
- **Dimensionality**: 2D array (N_timepoints × N_regions)
- **Data type**: float64
- **Time axis**: First dimension (rows = time)
- **Region axis**: Second dimension (columns = ROI)
- **Preprocessing applied**: Same denoising and filtering as used for connectivity computation
- **Volume censoring**: If applied, only retained volumes are included

**Example interpretation**:
```python
timeseries = np.load('sub-01_atlas-schaefer2018n100_timeseries.npy')
print(timeseries.shape)  # (350, 100) = 350 timepoints, 100 regions

# Extract a single region's time series
region_ts = timeseries[:, 0]  # First region

# Compute custom connectivity from raw time series
custom_conn = np.corrcoef(timeseries.T)  # 100 × 100 correlation
```

**Use cases**:
- Re-analysis with alternative connectivity measures
- Custom statistical models
- Network analysis requiring explicit time series
- Validation and quality control

### NIfTI Connectivity Maps (Voxelwise Methods) (.nii.gz)

**Filename pattern**: `sub-<ID>_task-<TASK>_seed-<SEED>_desc-correlation_connectivity.nii.gz`

**Format**: NIfTI-1 compressed format

**Specifications**:
- **Dimensionality**: 3D spatial volume
- **Data type**: float32
- **Affine**: Identical to input fMRI (MNI space)
- **Spatial resolution**: Matches preprocessed fMRI (typically 2mm isotropic)
- **Values**: Correlation coefficients at each voxel
- **Range**: -1.0 to 1.0

**Methods producing voxelwise outputs**:
1. **Seed-to-voxel**: Correlation between seed region and all voxels
2. **ROI-to-voxel**: Correlation between ROI and all voxels

### JSON Metadata Sidecars

**Filename**: Same as data file with `.json` extension (instead of `.npy` or `.nii.gz`)

**Example content for connectivity matrix**:
```json
{
  "AtlasName": "schaefer2018n100",
  "ConnectivityKind": "correlation",
  "Description": "ROI-to-ROI correlation matrix using Schaefer 100-parcel atlas",
  "AnalysisMethod": "roiToRoi",
  "MatrixShape": [100, 100],
  "NumberOfRegions": 100,
  "Dtype": "float64",
  "CreationTime": "2026-01-19T14:32:15.123456",
  "ROINames": ["7Networks_LH_Vis_1", "7Networks_LH_Vis_2", "..." ],
  "ROICoordinates": [
    [-22.5, -94.2, -12.1],
    [-14.3, -82.7, -8.4],
    "..."
  ],
  "CoordinateSpace": "MNI152NLin2009cAsym",
  "Shape": [100, 100]
}
```

**Key fields for group-level connectome visualization**:
- `ROICoordinates`: Array of [x, y, z] MNI coordinates for each ROI centroid (same order as matrix rows/columns). Required by `nilearn.plotting.plot_connectome()` to position nodes on glass brain visualizations.
- `CoordinateSpace`: The MNI space name (e.g., "MNI152NLin2009cAsym") for documentation and reproducibility.

**Example content for time series**:
```json
{
  "AtlasName": "schaefer2018n100",
  "Description": "ROI time series extracted from denoised fMRI",
  "TimeAxis": "first",
  "NumberOfTimepoints": 350,
  "NumberOfRegions": 100,
  "RepetitionTime": 2.0,
  "HighPassFrequency": 0.01,
  "LowPassFrequency": 0.08,
  "DenoisingStrategy": "csfwm_6p",
  "TemporalCensoringApplied": false,
  "ROICoordinates": [
    [-22.5, -94.2, -12.1],
    [-14.3, -82.7, -8.4],
    "..."
  ],
  "CoordinateSpace": "MNI152NLin2009cAsym",
  "Shape": [350, 100],
  "Dtype": "float64",
  "CreationTime": "2026-01-19T14:32:15.123456"
}
```

**Best practices for group analysis**:
- Use JSON sidecars to verify preprocessing consistency across subjects
- Check `HighPassFrequency`, `LowPassFrequency`, `DenoisingStrategy` for consistency
- Track `TemporalCensoringApplied` and `NumberOfTimepoints` for censoring awareness
- Use `ROINames` for anatomical labeling in results
- Use `ROICoordinates` for connectome visualization (glass brain plots)

---

## Method-Specific Outputs

### 1. ROI-to-ROI Analysis (Default)

**Output structure**:
```
sub-01/connectivity_data/
├── sub-01_task-rest_atlas-schaefer2018n100_desc-correlation_connectivity.npy
├── sub-01_task-rest_atlas-schaefer2018n100_desc-correlation_connectivity.json
├── sub-01_task-rest_atlas-schaefer2018n100_desc-covariance_connectivity.npy
├── sub-01_task-rest_atlas-schaefer2018n100_desc-covariance_connectivity.json
├── sub-01_task-rest_atlas-schaefer2018n100_desc-partial-correlation_connectivity.npy
├── sub-01_task-rest_atlas-schaefer2018n100_desc-partial-correlation_connectivity.json
├── sub-01_task-rest_atlas-schaefer2018n100_desc-precision_connectivity.npy
├── sub-01_task-rest_atlas-schaefer2018n100_desc-precision_connectivity.json
└── sub-01_task-rest_atlas-schaefer2018n100_timeseries.npy
```

**Connectivity matrices**: Always produces **all four connectivity measures** (correlation, covariance, partial correlation, precision)

**Time series**: Single file with shape (N_timepoints, N_regions)

**Typical use for group analysis**: All connectivity matrices are available for flexibility in statistical modeling

---

### 2. Seed-to-Voxel Analysis

**Output structure**:
```
sub-01/connectivity_data/
├── sub-01_task-rest_seed-PCC_desc-correlation_connectivity.nii.gz
├── sub-01_task-rest_seed-PCC_desc-correlation_connectivity.json
├── sub-01_task-rest_seed-mPFC_desc-correlation_connectivity.nii.gz
├── sub-01_task-rest_seed-mPFC_desc-correlation_connectivity.json
└── ... (one per seed)
```

**Number of outputs**: One 3D NIfTI file per seed region (defined in seeds.tsv)

**Data format**: 3D voxel-wise correlation maps

**Metadata includes**: Seed coordinates, sphere radius

---

### 3. ROI-to-Voxel Analysis

**Output structure**:
```
sub-01/connectivity_data/
├── sub-01_task-rest_roi-roi1_desc-correlation_connectivity.nii.gz
├── sub-01_task-rest_roi-roi1_desc-correlation_connectivity.json
├── sub-01_task-rest_roi-roi2_desc-correlation_connectivity.nii.gz
├── sub-01_task-rest_roi-roi2_desc-correlation_connectivity.json
└── ... (one per ROI mask)
```

**Number of outputs**: One 3D NIfTI file per ROI mask file

**Data format**: 3D voxel-wise correlation maps

---

### 4. Seed-to-Seed Analysis

**Output structure**:
```
sub-01/connectivity_data/
├── sub-01_task-rest_desc-seedtoSeed_connectivity.npy
├── sub-01_task-rest_desc-seedtoSeed_connectivity.json
└── sub-01_task-rest_timeseries.npy
```

**Connectivity matrix**: N_seeds × N_seeds square matrix

**Metadata**: Seed names, coordinates, sphere radii

---

## Temporal Censoring Outputs

When temporal censoring is enabled (for task fMRI condition-based analysis or motion scrubbing), output filenames include a `condition-<NAME>` entity.

### Condition-Based Analysis (Task fMRI)

**Trigger**: Use `--conditions` flag with condition names from events.tsv

**Output modification**: Inserts `condition-<CONDITION>` entity in filename

**Example without temporal censoring**:
```
sub-01_task-faces_atlas-schaefer2018n100_desc-correlation_connectivity.npy
```

**Example with temporal censoring (all conditions)**:
```
sub-01_task-faces_condition-face_atlas-schaefer2018n100_desc-correlation_connectivity.npy
sub-01_task-faces_condition-house_atlas-schaefer2018n100_desc-correlation_connectivity.npy
sub-01_task-faces_condition-scrambled_atlas-schaefer2018n100_desc-correlation_connectivity.npy
sub-01_task-faces_condition-baseline_atlas-schaefer2018n100_desc-correlation_connectivity.npy
```

**Example with temporal censoring (subset of conditions)**:
```
sub-01_task-faces_condition-face_atlas-schaefer2018n100_desc-correlation_connectivity.npy
sub-01_task-faces_condition-house_atlas-schaefer2018n100_desc-correlation_connectivity.npy
```

### JSON Metadata for Censored Data

Additional fields in JSON sidecars for temporally censored data:

```json
{
  "TemporalCensoringApplied": true,
  "CensoringType": "condition_selection",
  "ConditionName": "face",
  "OriginalNumberOfTimepoints": 420,
  "RetainedNumberOfTimepoints": 84,
  "RetentionFraction": 0.20,
  "EventsFile": "events.tsv",
  "ConditionOnset": [0.0, 3.0, 6.0],
  "ConditionDuration": [2.5, 2.5, 2.5],
  "TransitionBuffer": 0.0,
  "Warning": "Low retention (20%): Results may be unreliable with very few volumes"
}
```

### Motion Censoring

When motion censoring is enabled with `--fd-threshold`, the JSON metadata includes:

```json
{
  "TemporalCensoringApplied": true,
  "CensoringType": "motion_censoring",
  "FramewiseDisplacementThreshold": 0.5,  # units: cm (fMRIPrep reports FD in cm)
  "FramewiseDisplacementExtendBefore": 1,
  "FramewiseDisplacementExtendAfter": 1,
  "OriginalNumberOfTimepoints": 420,
  "RetainedNumberOfTimepoints": 356,
  "RetentionFraction": 0.85,
  "FramesDueToMotion": 64,
  "CensoringMaskFile": "sub-01_task-rest_desc-censoring_mask.npy"
}
```

---

## Quality Control and Reports

### HTML Report Structure

Each participant generates one comprehensive HTML report: `sub-<ID>_task-<TASK>_desc-<ATLAS>_report.html`

**Report sections**:

1. **Summary**
   - Subject/session/task information
   - Processing parameters
   - Key metrics (number of timepoints, TR, filters)

2. **Denoising QA**
   - Confound time series plot
   - Confound correlation matrix
   - Before/after denoising histogram (signal intensity comparison)
   - tSNR (temporal signal-to-noise ratio) statistics

3. **Temporal Censoring** (if applied)
   - Original vs. retained volume counts
   - Breakdown by censoring reason
   - Condition-specific volume counts
   - Visual censoring mask timeline

4. **Connectivity Matrices** (one section per connectivity measure)
   - Heatmap visualization (colorbar with min/max values)
   - Connectome glass brain plot
   - Connectivity value histogram with summary statistics
   - Mean, median, std deviation of connectivity values

5. **References**
   - Citations for methods and software used

### Downloadable Figures

HTML reports include a `figures/` subdirectory with high-quality PNG versions:

- `connectivity_<MEASURE>.png` - Heatmap matrices (300 dpi, publication-ready)
- `connectome_<MEASURE>.png` - Glass brain visualizations
- `histogram_<MEASURE>.png` - Value distributions
- `confounds_timeseries.png` - Time series of denoising regressors
- `confounds_correlation.png` - Confound inter-correlation
- `denoising-histogram.png` - Before/after comparison
- `temporal_censoring.png` - Censoring visualization (if applicable)

**Format**: PNG, 300 DPI for publication
**Size**: ~100-500 KB each
**Use**: Can be directly incorporated into supplementary materials

---

## Atlas-Specific Details

### Schaefer 2018 (Default)

**Available versions**:
- `schaefer2018n100` - 100 parcels
- `schaefer2018n200` - 200 parcels

**Properties**:
- Cortical only
- 7-network organization
- Symmetric parcellation
- ROI names follow format: `7Networks_LH_<Network>_<Number>` or `7Networks_RH_<Network>_<Number>`
- Left hemisphere (LH) and right hemisphere (RH) labeled

**Expected matrix shapes**:
- n100: 100 × 100
- n200: 200 × 200

### AAL (Automated Anatomical Labeling)

**Identifier**: `atlas-aal`

**Properties**:
- 116 regions (cortical + subcortical)
- Anatomically defined
- 58 regions per hemisphere
- ROI names: Anatomical region names with LH/RH suffix

**Expected matrix shape**: 116 × 116

### Harvard-Oxford

**Identifier**: `atlas-harvardoxford`

**Properties**:
- 96 regions
- Probabilistic atlas (cortical + subcortical)
- Derived from anatomical MRI data
- Well-validated structure

**Expected matrix shape**: 96 × 96

### CanICA (Data-Driven)

**Identifier**: `atlas-canica`

**Properties**:
- Automatically generated from participant data
- Independent component analysis decomposition
- Number of components configurable (default: 20)
- Subject-specific; not directly comparable across subjects

**Expected matrix shape**: N_components × N_components (typically 20 × 20)

---

## Important Notes for Group-Level Analysis

### Data Consistency Checks

Before implementing group-level analysis, verify:

1. **Same atlas across all participants**: Check that `AtlasName` in JSON metadata is identical
2. **Same connectivity kind** (if using single measure): Verify `ConnectivityKind` consistency
3. **Consistent preprocessing**: Check `HighPassFrequency`, `LowPassFrequency`, `DenoisingStrategy`
4. **Temporal censoring awareness**: Check `TemporalCensoringApplied` and `RetentionFraction`
   - If some subjects have ~85% retention and others have ~20%, statistical modeling needs adjustment
5. **Matrix dimensionality**: All matrices should have identical shapes (N_regions × N_regions)

### Loading Connectivity Matrices for Group Analysis

```python
import numpy as np
import json
from pathlib import Path

# Load connectivity matrix
connectivity = np.load('sub-01_atlas-schaefer2018n100_desc-correlation_connectivity.npy')

# Load metadata
with open('sub-01_atlas-schaefer2018n100_desc-correlation_connectivity.json') as f:
    metadata = json.load(f)

print(f"Matrix shape: {metadata['Shape']}")
print(f"Atlas: {metadata['AtlasName']}")
print(f"Temporal censoring applied: {metadata.get('TemporalCensoringApplied', False)}")

# Organize for group analysis
group_data = []
subject_ids = []
for sub_dir in Path('/output').glob('sub-*'):
    conn_file = sub_dir / 'connectivity_data' / 'connectivity_correlation.npy'
    if conn_file.exists():
        connectivity = np.load(conn_file)
        group_data.append(connectivity)
        subject_ids.append(sub_dir.name)

group_data = np.array(group_data)  # (N_subjects, N_regions, N_regions)
```

### Handling Multiple Connectivity Measures

For more flexible group-level analysis, all four connectivity measures are computed simultaneously:

```python
# Load all connectivity measures
measures = ['correlation', 'covariance', 'partial-correlation', 'precision']
connectivities = {}

for measure in measures:
    filename = f'sub-01_atlas-schaefer2018n100_desc-{measure}_connectivity.npy'
    connectivities[measure] = np.load(filename)

# Example: Use partial correlation for direct connections
partial_conn = connectivities['partial-correlation']
```

### Vectorization for Machine Learning

For machine learning pipelines, connectivity matrices are typically vectorized:

```python
def matrix_to_vector(matrix):
    """Convert symmetric matrix to upper triangle vector."""
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]

def vector_to_matrix(vector, n_regions):
    """Reconstruct symmetric matrix from vector."""
    matrix = np.zeros((n_regions, n_regions))
    indices = np.triu_indices(n_regions, k=1)
    matrix[indices] = vector
    matrix = matrix + matrix.T
    return matrix

# For group analysis
n_regions = 100
n_subjects = 50
connectivity_vectors = np.zeros((n_subjects, n_regions*(n_regions-1)//2))

for i, sub_id in enumerate(subject_ids):
    conn = np.load(f'sub-{sub_id:02d}_connectivity_correlation.npy')
    connectivity_vectors[i] = matrix_to_vector(conn)
```

### Tangent Space Representation

For statistical analysis, Connectomix uses tangent space geometry:

```python
# Connectomix produces covariance-like matrices that can be analyzed in tangent space
# This is particularly useful for multivariate statistical tests

# Load covariance matrices
covariances = []
for sub_dir in Path('/output').glob('sub-*'):
    cov = np.load(sub_dir / 'connectivity_data' / '*desc-covariance_connectivity.npy')
    covariances.append(cov)

# Compute group-level covariance (geometric mean in SPD manifold)
# This requires specialized functions from nilearn.connectome.ConnectivityMeasure
```

---

## Summary Reference Table

| Aspect | Details |
|--------|---------|
| **Connectivity Measures** | 4 types: correlation, covariance, partial correlation, precision |
| **Matrix Dimensionality** | N_regions × N_regions (symmetric, float64) |
| **Time Series Format** | N_timepoints × N_regions (float64) |
| **File Format** | .npy (NumPy binary), .nii.gz (NIfTI for voxelwise), .json (metadata) |
| **Naming Convention** | BIDS-compliant with entities: sub, ses, task, run, space, condition, atlas, desc |
| **Methods** | ROI-to-ROI, Seed-to-Voxel, ROI-to-Voxel, Seed-to-Seed |
| **Standard Atlases** | Schaefer 2018 (100/200 parcels), AAL (116), Harvard-Oxford (96) |
| **ROI Coordinates** | MNI centroids in JSON sidecar (`ROICoordinates` field) for connectome plots |
| **Temporal Censoring** | Condition-specific outputs with condition- entity in filename |
| **Quality Control** | HTML reports with figures, metadata JSON, before/after histograms |
| **Metadata Location** | .json files alongside data with comprehensive processing information |
| **Typical File Sizes** | Connectivity matrix: 80 KB (100 regions), Time series: variable (MB scale) |

---

## Recommendations for Group-Level Implementation

1. **Always load and check JSON sidecars** to verify consistency across subjects before group analysis
2. **Use correlation or partial correlation** for most statistical tests (bounded, interpretable)
3. **Preserve covariance matrices** for analyses requiring variance magnitude information
4. **Be aware of temporal censoring** when sample sizes vary across subjects
5. **Utilize provided time series** (.npy) for validation and alternative connectivity computations
6. **Check HTML reports** for quality and identify subjects with unusual patterns
7. **Consider multiple connectivity measures** in exploratory analyses to test robustness
8. **Document data provenance** (preprocessing version, atlas, denoising strategy) for reproducibility
9. **Use ROICoordinates** from JSON sidecars for connectome glass brain visualizations

