# Connectomix

<p align="center">
  <strong>Functional Connectivity Analysis from fMRIPrep Outputs</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#analysis-methods">Methods</a> •
  <a href="#temporal-censoring">Temporal Censoring</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#documentation">Documentation</a>
</p>

---

## Overview

**Connectomix** is a BIDS-compliant tool for computing functional connectivity from fMRIPrep-preprocessed fMRI data. It supports multiple connectivity methods at both participant and group levels, with comprehensive HTML reports for quality assurance.

### Key Features

- 🧠 **Four connectivity methods**: seed-to-voxel, ROI-to-voxel, seed-to-seed, ROI-to-ROI
- � **Four connectivity measures**: correlation, covariance, partial correlation, precision
- 📊 **Two analysis levels**: participant-level and group-level statistical inference
- ⏱️ **Temporal censoring**: condition-based analysis for task fMRI, motion scrubbing
- 🔧 **Flexible preprocessing**: predefined denoising strategies or custom confound selection
- 📋 **BIDS-compliant**: standardized input/output structure
- 📄 **HTML reports**: connectivity matrices, connectome plots, denoising QA histograms

### Technology Stack

- Python 3.8+
- [Nilearn](https://nilearn.github.io/) for neuroimaging operations
- [PyBIDS](https://bids-standard.github.io/pybids/) for BIDS compliance
- [Nibabel](https://nipy.org/nibabel/) for NIfTI I/O
- NumPy, Pandas, SciPy for data processing

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ln2t/connectomix.git
cd connectomix

# Install in development mode
pip install -e .

# Verify installation
connectomix --version
```

**Requirements:**
- Python 3.8+
- fMRIPrep-preprocessed data (BIDS derivatives)

---

## Quick Start

### Basic Usage

```bash
# Participant-level analysis (simplest)
connectomix /data/bids /data/output participant

# With specific subject and task
connectomix /data/bids /data/output participant -p 01 -t rest

# With custom atlas and connectivity method
connectomix /data/bids /data/output participant --atlas aal --method roiToRoi

# With configuration file
connectomix /data/bids /data/output participant -c config.yaml

# Group-level analysis
connectomix /data/bids /data/output group -c group_config.yaml

# Verbose output for debugging
connectomix /data/bids /data/output participant -v
```

### Specifying fMRIPrep Location

```bash
# If fMRIPrep output is not in bids_dir/derivatives/fmriprep
connectomix /data/bids /data/output participant \
  --derivatives fmriprep=/path/to/fmriprep
```

### Common Command-Line Arguments

| Argument | Short | Description | Example |
|----------|-------|-------------|---------|
| `--participant-label` | `-p` | Subject(s) to process | `-p 01` |
| `--task` | `-t` | Task name to process | `-t restingstate` |
| `--session` | `-s` | Session to process | `-s 1` |
| `--run` | `-r` | Run to process | `-r 1` |
| `--space` | | MNI space to use | `--space MNI152NLin2009cAsym` |
| `--config` | `-c` | Config file path | `-c my_config.yaml` |
| `--atlas` | | Atlas for ROI connectivity | `--atlas schaefer2018n200` |
| `--method` | | Connectivity method | `--method roiToRoi` |
| `--denoising` | | Predefined strategy | `--denoising csfwm_6p` |
| `--derivatives` | `-d` | Derivative locations | `-d fmriprep=/path` |
| `--label` | | Custom output label | `--label myanalysis` |
| `--verbose` | `-v` | Enable debug output | `-v` |

---

## Analysis Methods

Connectomix supports four connectivity analysis methods:

### 1. Seed-to-Voxel

Compute correlation between user-defined seed regions and all brain voxels.

```yaml
method: "seedToVoxel"
seeds_file: "seeds.tsv"  # Tab-separated: name, x, y, z
radius: 5.0              # Sphere radius in mm
```

**Seeds file format (seeds.tsv):**
```tsv
name	x	y	z
PCC	0	-52	18
mPFC	0	52	0
LIPL	-45	-70	35
```

**Output:** One NIfTI per seed with correlation values at each voxel.

### 2. ROI-to-Voxel

Like seed-to-voxel but with arbitrary ROI masks instead of spheres.

```yaml
method: "roiToVoxel"
roi_masks: ["/path/to/roi1.nii.gz", "/path/to/roi2.nii.gz"]
```

**Output:** One NIfTI per ROI with correlation values.

### 3. Seed-to-Seed

Compute correlation matrix between multiple seeds.

```yaml
method: "seedToSeed"
seeds_file: "seeds.tsv"
radius: 5.0
```

**Output:** N×N correlation matrix (numpy array).

### 4. ROI-to-ROI

Whole-brain parcellation-based connectivity matrix using a standard atlas.

```yaml
method: "roiToRoi"
atlas: "schaefer2018n100"
```

**Output:** Multiple connectivity matrices (N×N where N = number of atlas regions):
- `*_desc-correlation_connectivity.npy` - Pearson correlation
- `*_desc-covariance_connectivity.npy` - Sample covariance
- `*_desc-partial-correlation_connectivity.npy` - Partial correlation (controlling for other regions)
- `*_desc-precision_connectivity.npy` - Inverse covariance (sparse direct connections)
- `*_timeseries.npy` - Raw ROI time series for reanalysis

### Available Connectivity Measures

For ROI-to-ROI analysis, Connectomix computes **four complementary connectivity measures** to characterize brain network interactions:

| Measure | Values | Interpretation |
|---------|--------|----------------|
| **Correlation** | -1 to +1 | Normalized covariance; strength & direction of linear relationship |
| **Covariance** | Unbounded | Raw joint variability; retains variance magnitude information |
| **Partial Correlation** | -1 to +1 | Correlation controlling for all other regions; reveals direct connections |
| **Precision** | Unbounded | Inverse covariance; sparse matrix revealing direct statistical dependencies |

#### Pearson Correlation

The most commonly used measure. Pearson correlation normalizes the covariance by the standard deviations, yielding values between -1 and +1 that indicate the strength and direction of the linear relationship between two regions.

**Use when:** You want easily interpretable values; comparing connectivity across subjects with different signal variances.

**Formula:** $\rho_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sigma_i \sigma_j}$

#### Covariance

The sample covariance measures how two variables vary together, retaining information about the magnitude of variance. Unlike correlation, covariance is not normalized and can take any real value.

**Use when:** Variance magnitude is meaningful for your analysis; you want to preserve amplitude information.

**Formula:** $\text{Cov}(X_i, X_j) = \frac{1}{n-1}\sum_{t=1}^{n}(x_i^t - \bar{x}_i)(x_j^t - \bar{x}_j)$

#### Partial Correlation

Partial correlation measures the relationship between two regions while controlling for the influence of all other regions. This reveals direct connections by removing indirect effects mediated through other areas.

**Use when:** You want to identify direct functional connections; distinguishing direct from indirect relationships.

**Formula:** $\rho_{ij|Z} = -\frac{\Theta_{ij}}{\sqrt{\Theta_{ii}\Theta_{jj}}}$ where $\Theta$ is the precision matrix

#### Precision (Inverse Covariance)

The precision matrix is the inverse of the covariance matrix. It encodes conditional dependencies: if $\Theta_{ij} = 0$, regions i and j are conditionally independent given all other regions. This provides a sparse representation of direct statistical relationships.

**Use when:** You want sparse networks; identifying direct statistical dependencies; graph-theoretical analyses.

**Formula:** $\Theta = \Sigma^{-1}$

> **Tip:** Correlation and partial correlation are normalized (-1 to +1) and easier to interpret. Covariance and precision preserve variance information but require careful interpretation across subjects.

### Available Atlases

| Name | Regions | Description |
|------|---------|-------------|
| `schaefer2018n100` | 100 | Schaefer 7-network 100 parcels |
| `schaefer2018n200` | 200 | Schaefer 7-network 200 parcels |
| `aal` | 116 | Automated Anatomical Labeling |
| `harvardoxford` | 96 | Harvard-Oxford cortical + subcortical |
| `canica` | Custom | Data-driven ICA (computed from your data) |

### Using a Custom Atlas

Connectomix allows you to use a custom parcellation atlas for ROI-to-ROI or ROI-to-voxel analysis. A custom atlas requires:

1. **A parcellation NIfTI file** — 3D image where each ROI has a unique non-zero integer label (background = 0)
2. **(Optional) A labels file** — human-readable ROI names
3. **(Optional) MNI coordinates** — for connectome (glass brain) visualizations

#### Option 1: Provide a Direct Path

Pass the full path to a NIfTI parcellation file:

```bash
connectomix /data/bids /data/output participant \
  --atlas /path/to/my_atlas.nii.gz
```

If you have a labels file, name it with the **same basename** as your NIfTI file:
- `my_atlas.nii.gz` → `my_atlas.csv`, `my_atlas.tsv`, `my_atlas.txt`, or `my_atlas.json`

#### Option 2: Place the Atlas in Nilearn's Data Directory

Create a folder in `~/nilearn_data` (or `$NILEARN_DATA`) with your atlas:

```bash
mkdir -p ~/nilearn_data/my_custom_atlas
cp /path/to/atlas.nii.gz ~/nilearn_data/my_custom_atlas/
cp /path/to/labels.csv ~/nilearn_data/my_custom_atlas/
```

Then reference it by folder name:

```bash
connectomix /data/bids /data/output participant --atlas my_custom_atlas
```

#### Supported Label File Formats

Connectomix supports multiple formats for specifying ROI names and coordinates:

**CSV with coordinates (recommended for connectome plots):**

```csv
x,y,z,name,network
-53.28,-8.88,32.36,L Auditory,Auditory
53.47,-6.49,27.52,R Auditory,Auditory
-0.15,51.42,7.58,Frontal DMN,DMN
```

Columns `x`, `y`, `z` specify MNI coordinates for each ROI centroid. These are used for:
- Glass brain / connectome visualizations
- Spatial reference in JSON sidecars

**TSV (like Schaefer atlas):**

```tsv
1	7Networks_LH_Vis_1	120	18	131	0
2	7Networks_LH_Vis_2	120	18	132	0
```

The second column is used as the ROI name.

**Plain text (one label per line):**

```text
LeftHippocampus
RightHippocampus
LeftAmygdala
RightAmygdala
```

**JSON array:**

```json
["LeftHippocampus", "RightHippocampus", "LeftAmygdala", "RightAmygdala"]
```

**JSON with coordinates:**

```json
{
  "labels": ["L Auditory", "R Auditory", "Frontal DMN"],
  "coordinates": [[-53.28, -8.88, 32.36], [53.47, -6.49, 27.52], [-0.15, 51.42, 7.58]]
}
```

#### File Naming Convention

Labels files are searched in this priority order:

1. **Same basename as NIfTI**: `my_atlas.csv` for `my_atlas.nii.gz`
2. **Generic labels file**: `labels.csv`, `labels.tsv`, `labels.txt`, `labels.json`

#### What Happens Without a Labels File?

If no labels file is found, Connectomix will:
1. Extract unique integer values from the parcellation image
2. Generate labels as `ROI_1`, `ROI_2`, etc.
3. Compute ROI centroid coordinates automatically using nilearn

> **Tip:** For publication-quality connectome plots, provide a CSV with MNI coordinates and meaningful ROI names.

---

## Temporal Censoring

Temporal censoring removes specific timepoints (volumes) from fMRI data before connectivity analysis. This is essential for:

1. **Condition-based analysis** (task fMRI): Compute separate connectivity matrices for each experimental condition
2. **Motion scrubbing**: Remove high-motion timepoints based on framewise displacement (FD)
3. **Dummy scan removal**: Discard initial volumes during scanner equilibration

**By default, temporal censoring is disabled.** Enable it with CLI options or configuration.


### Condition-Based Analysis (Task fMRI)

Compute connectivity for specific experimental conditions:

```bash
# Compute connectivity per condition
connectomix /data/bids /data/output participant -t faces \
  --conditions face house scrambled

# Compute connectivity for BASELINE only (inter-trial intervals)
# Use when you want to exclude task periods and keep only rest/ITI
connectomix /data/bids /data/output participant -t gas \
  --conditions baseline

# Compute connectivity for both task conditions AND baseline
connectomix /data/bids /data/output participant -t faces \
  --conditions face house baseline

# Legacy: --include-baseline flag (equivalent to adding 'baseline' to --conditions)
connectomix /data/bids /data/output participant -t faces \
  --conditions face house --include-baseline

# Add buffer around condition transitions
connectomix /data/bids /data/output participant -t faces \
  --conditions face house --transition-buffer 2.0
```

**Special condition keywords:**
- `baseline`, `rest`, `iti`, `inter-trial`: Select timepoints NOT covered by any event in events.tsv

**How it works:**
1. Reads the `events.tsv` file for your task (automatically found in BIDS structure)
2. Identifies timepoints belonging to each condition based on onset/duration
3. Creates separate masks for each condition
4. Computes connectivity separately for each condition
5. Outputs one connectivity matrix per condition with `condition-{name}` in the filename

**Example output:**
```
sub-01/
├── connectivity_data/
│   ├── sub-01_task-faces_condition-face_desc-schaefer_correlation.npy
│   ├── sub-01_task-faces_condition-house_desc-schaefer_correlation.npy
│   └── sub-01_task-faces_condition-baseline_desc-schaefer_correlation.npy
└── sub-01_task-faces_condition-face+house+baseline_desc-schaefer_report.html
```

**events.tsv format:**
```tsv
onset	duration	trial_type
0.0	2.5	face
3.0	2.5	house
6.0	2.5	scrambled
```

### Motion Scrubbing

Remove high-motion volumes based on framewise displacement:

```bash
# Remove volumes with FD > 0.5mm
connectomix /data/bids /data/output participant --fd-threshold 0.5

# Also censor ±1 volume around high-FD timepoints
connectomix /data/bids /data/output participant --fd-threshold 0.5 --fd-extend 1
```

FD values are read from fMRIPrep's confounds file (`framewise_displacement` column).

### Dummy Scan Removal

Drop initial volumes during scanner equilibration:

```bash
connectomix /data/bids /data/output participant --drop-initial 4
```

### Combining Censoring Options

```bash
connectomix /data/bids /data/output participant -t faces \
  --conditions face house \
  --fd-threshold 0.3 \
  --fd-extend 1 \
  --drop-initial 4
```

### Temporal Censoring CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--conditions COND [...]` | Condition names from events.tsv (use 'baseline' for inter-trial intervals) | (disabled) |
| `--events-file FILE` | Path to events.tsv | auto-detect |
| `--include-baseline` | Include inter-trial intervals (same as adding 'baseline' to --conditions) | false |
| `--transition-buffer SEC` | Exclude N seconds around transitions | 0 |
| `--fd-threshold MM` | Remove volumes with FD > threshold | (disabled) |
| `--fd-extend N` | Also remove ±N volumes around high-FD | 0 |
| `--drop-initial N` | Drop first N volumes | 0 |

### Quality Control

The HTML report includes a **Temporal Censoring** section showing:
- Original vs retained volume counts
- Censoring breakdown by reason (motion, initial drop, etc.)
- Condition-specific volume counts
- Visual censoring mask

**Important:** If too few volumes remain after censoring (<30% or <50 volumes), Connectomix will issue a prominent warning in the report and logs. Results with very few volumes should be interpreted with caution as connectivity estimates become unreliable.

---

## Configuration

### Configuration File

For complex analyses, use a YAML or JSON configuration file:

```bash
connectomix /data/bids /data/output participant -c config.yaml
```

### Participant-Level Configuration

```yaml
# participant_config.yaml

# BIDS filters
subject: ["01", "02", "03"]
tasks: ["restingstate"]
sessions: null
spaces: ["MNI152NLin2009cAsym"]

# Analysis method
method: "roiToRoi"
atlas: "schaefer2018n100"
connectivity_kind: "correlation"

# Denoising
confounds: ["csf", "white_matter", "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
high_pass: 0.01
low_pass: 0.08

# Temporal censoring (optional)
temporal_censoring:
  enabled: true
  drop_initial_volumes: 4
  condition_selection:
    enabled: true
    conditions: ["face", "house"]
  motion_censoring:
    enabled: true
    fd_threshold: 0.5
```

### Group-Level Configuration

```yaml
# group_config.yaml

# Participants
subject: ["01", "02", "03", "04", "05"]
task: "restingstate"
space: "MNI152NLin2009cAsym"

# Method (must match participant-level)
method: "roiToRoi"
smoothing: 8.0

# Analysis
analysis_name: "patients_vs_controls"

# Design matrix (from participants.tsv)
covariates: ["group"]
add_intercept: true

# Contrast
contrast: "group"

# Statistics
uncorrected_alpha: 0.001
fdr_alpha: 0.05
fwe_alpha: 0.05
thresholding_strategies: ["uncorrected", "fdr", "fwe"]

# Computational
n_permutations: 10000
n_jobs: 4
```

### Denoising Strategies

Use predefined strategies with `--denoising` or define custom confounds:

| Strategy | Confounds | Description |
|----------|-----------|-------------|
| `minimal` | 6 motion parameters | Basic motion correction only |
| `csfwm_6p` | CSF + WM + 6 motion | Standard denoising |
| `csfwm_12p` | CSF + WM + 12 motion | With motion derivatives |
| `gs_csfwm_6p` | Global + CSF + WM + 6 motion | Aggressive |
| `gs_csfwm_12p` | Global + CSF + WM + 12 motion | Very aggressive |
| `csfwm_24p` | CSF + WM + 24 motion | With derivatives and squares |
| `compcor_6p` | 6 aCompCor + 6 motion | CompCor-based |

#### Wildcard Support

Confound names support **shell-style wildcards** for flexible selection:

| Pattern | Matches | Example |
|---------|---------|---------|
| `*` | Any characters | `trans_*` → `trans_x`, `trans_y`, `trans_z` |
| `?` | Single character | `rot_?` → `rot_x`, `rot_y`, `rot_z` |
| `[seq]` | Character in sequence | `a_comp_cor_0[0-5]` → first 6 aCompCor |

**Examples:**

```yaml
# Select all 6 aCompCor components
confounds: ["a_comp_cor_*"]

# Select specific CompCor range
confounds: ["c_comp_cor_0?", "csf", "white_matter"]

# Motion + all cosine regressors
confounds: ["trans_*", "rot_*", "cosine*"]

# First 10 aCompCor components
confounds: ["a_comp_cor_0*"]
```

> **Note:** If a wildcard pattern matches no columns, an error is raised with suggestions.

---

## Output Structure

```
output_dir/
├── dataset_description.json          # BIDS derivative metadata
├── config/
│   └── backups/
│       └── config_TIMESTAMP.json     # Configuration backups
├── sub-01/
│   ├── figures/                      # Report figures
│   │   ├── connectivity_correlation.png
│   │   ├── connectivity_covariance.png
│   │   ├── connectivity_partial-correlation.png
│   │   ├── connectivity_precision.png
│   │   ├── connectome_correlation.png      # Glass brain plots
│   │   ├── histogram_correlation.png       # Value distributions
│   │   ├── confounds_timeseries.png
│   │   ├── confounds_correlation.png
│   │   ├── denoising-histogram.png         # Before/after denoising
│   │   └── temporal_censoring.png
│   ├── func/                         # Denoised functional data
│   │   ├── sub-01_task-rest_desc-denoised_bold.nii.gz
│   │   └── sub-01_task-rest_desc-denoised_bold.json
│   ├── connectivity_data/            # Connectivity matrices & time series
│   │   ├── sub-01_task-rest_atlas-schaefer_desc-correlation_connectivity.npy
│   │   ├── sub-01_task-rest_atlas-schaefer_desc-covariance_connectivity.npy
│   │   ├── sub-01_task-rest_atlas-schaefer_desc-partial-correlation_connectivity.npy
│   │   ├── sub-01_task-rest_atlas-schaefer_desc-precision_connectivity.npy
│   │   └── sub-01_task-rest_atlas-schaefer_timeseries.npy
│   └── sub-01_task-rest_desc-schaefer_report.html
├── sub-02/
│   └── ...
└── group/
    └── roiToRoi/
        └── patients_vs_controls/
            ├── designMatrix.tsv
            ├── stat-t.nii.gz
            ├── threshold-fdr_stat-t.nii.gz
            ├── clusterTable.tsv
            └── report.html
```

### Connectivity Data Files

Each connectivity matrix (`.npy`) has an accompanying JSON sidecar with metadata:

```json
{
    "ConnectivityMeasure": "correlation",
    "AtlasName": "schaefer2018n100",
    "NumRegions": 100,
    "MatrixShape": [100, 100],
    "ROILabels": ["7Networks_LH_Vis_1", "7Networks_LH_Vis_2", "..."],
    "ROICoordinates": [[-22.0, -93.0, -9.0], [-26.0, -81.0, -11.0], "..."],
    "CoordinateSpace": "MNI152NLin2009cAsym",
    "fMRIPrepVersion": "23.1.0",
    "EffectiveVolumeCount": 180,
    "HighPass": 0.01,
    "LowPass": 0.08
}
```

**ROICoordinates** are MNI centroids (x, y, z) for each ROI, enabling connectome glass brain visualization with tools like nilearn's `plot_connectome()`.

**Atlas matrix shapes:**
| Atlas | Regions | Matrix Shape |
|-------|---------|--------------|
| `schaefer2018n100` | 100 | 100 × 100 |
| `schaefer2018n200` | 200 | 200 × 200 |
| `aal` | 116 | 116 × 116 |
| `harvardoxford` | 96 | 96 × 96 |
| `canica` | Custom | N × N (user-defined) |

### Loading Connectivity Data

```python
import numpy as np
import json
from pathlib import Path

# Load connectivity matrix
conn_file = Path('sub-01/connectivity_data/sub-01_task-rest_atlas-schaefer_desc-correlation_connectivity.npy')
connectivity = np.load(conn_file)

# Load metadata from JSON sidecar
json_file = conn_file.with_suffix('.json')
with open(json_file) as f:
    metadata = json.load(f)

# Access ROI coordinates for connectome plotting
roi_coords = np.array(metadata['ROICoordinates'])
roi_labels = metadata['ROILabels']

# Plot connectome using nilearn
from nilearn.plotting import plot_connectome
plot_connectome(connectivity, roi_coords, 
                edge_threshold='95%', 
                node_size=20,
                title=f"Subject 01 - {metadata['ConnectivityMeasure']}")
```

### Vectorization for Machine Learning

Connectivity matrices can be vectorized for group analysis or machine learning:

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

# For group analysis: stack all subjects
n_regions = 100
n_subjects = 50
connectivity_vectors = np.zeros((n_subjects, n_regions*(n_regions-1)//2))

for i, sub_dir in enumerate(Path('/output').glob('sub-*')):
    conn = np.load(sub_dir / 'connectivity_data' / '*_desc-correlation_connectivity.npy')
    connectivity_vectors[i] = matrix_to_vector(conn)
```

### HTML Report Contents

Each participant-level HTML report includes:

| Section | Contents |
|---------|----------|
| **Summary** | Subject info, processing parameters, key metrics |
| **Denoising** | Confound time series, inter-correlation matrix, before/after histogram |
| **Temporal Censoring** | Volume counts, censoring reasons, visual mask (if enabled) |
| **Connectivity** | For each measure: matrix heatmap, connectome glass brain, value histogram |
| **References** | Relevant citations for methods used |

---

## Common Workflows

### Workflow 1: Basic Resting-State Analysis

```bash
# 1. Run participant-level
connectomix /data/bids /data/output participant \
  -c participant_config.yaml -v

# 2. Check HTML reports
ls /data/output/sub-*/

# 3. Run group-level
connectomix /data/bids /data/output group \
  -c group_config.yaml -v
```

### Workflow 2: Task-Based Connectivity

```bash
# Compute connectivity for each condition
connectomix /data/bids /data/output participant \
  -t faces \
  --conditions face house scrambled \
  --fd-threshold 0.5 \
  -v
```

### Workflow 3: Data-Driven Parcellation

```yaml
# Use CanICA to generate subject-specific parcellation
method: "roiToRoi"
atlas: "canica"
n_components: 20
```

---

## Troubleshooting

### "No functional files found"
Check your BIDS entity filters. Use `-v` to see query details.

### "Confound not found"
Check fMRIPrep's confounds TSV columns. Use `--denoising minimal` for basic motion parameters, or use wildcards (e.g., `a_comp_cor_*`) to match multiple components.

### "Too few volumes after censoring"
Relax your censoring thresholds (e.g., increase `--fd-threshold`).

### "Geometric consistency check failed"
Connectomix will automatically resample if subjects have different geometries.

### Slow permutation testing
Reduce `n_permutations` (e.g., 5000) or increase `n_jobs` for parallelization.

---

## Tips and Best Practices

1. **Start small**: Test with 1-2 subjects before full dataset
2. **Use verbose mode** (`-v`) when debugging
3. **Check HTML reports** for quality assurance
4. **Denoising**: Start with `csfwm_6p`, adjust based on data quality
5. **Permutations**: 10000 for publication, 5000 for exploration
6. **Smoothing**: 6-8mm FWHM is typical for group analysis

---

## Configuration Reference

### Participant-Level Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subject` | list | null | Subject IDs (without "sub-") |
| `tasks` | list | null | Task names |
| `sessions` | list | null | Session IDs |
| `runs` | list | null | Run numbers |
| `spaces` | list | null | MNI spaces |
| `method` | string | "roiToRoi" | Analysis method |
| `confounds` | list | [6 motion] | Confound columns (supports wildcards: `*`, `?`) |
| `high_pass` | float | 0.01 | High-pass cutoff (Hz) |
| `low_pass` | float | 0.08 | Low-pass cutoff (Hz) |
| `seeds_file` | path | null | Seeds TSV file |
| `radius` | float | 5.0 | Seed sphere radius (mm) |
| `atlas` | string | "schaefer2018n100" | Atlas name |
| `connectivity_kind` | string | "correlation" | Connectivity measure |

### Temporal Censoring Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable censoring |
| `drop_initial_volumes` | int | 0 | Dummy scans to drop |
| `condition_selection.enabled` | bool | false | Enable condition selection |
| `condition_selection.conditions` | list | [] | Conditions to include |
| `condition_selection.include_baseline` | bool | false | Include baseline |
| `condition_selection.transition_buffer` | float | 0 | Buffer (seconds) |
| `motion_censoring.enabled` | bool | false | Enable FD censoring |
| `motion_censoring.fd_threshold` | float | 0.5 | FD threshold (mm) |
| `motion_censoring.extend_before` | int | 0 | Extend before |
| `motion_censoring.extend_after` | int | 0 | Extend after |
| `min_volumes_retained` | int | 50 | Minimum volumes |
| `min_fraction_retained` | float | 0.3 | Minimum fraction |

### Group-Level Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analysis_name` | string | required | Analysis identifier |
| `smoothing` | float | null | Spatial smoothing FWHM |
| `covariates` | list | [] | Columns from participants.tsv |
| `add_intercept` | bool | true | Add intercept to design |
| `contrast` | string/list | required | Contrast specification |
| `uncorrected_alpha` | float | 0.001 | Uncorrected threshold |
| `fdr_alpha` | float | 0.05 | FDR threshold |
| `fwe_alpha` | float | 0.05 | FWE threshold |
| `n_permutations` | int | 10000 | Permutation count |
| `n_jobs` | int | 1 | Parallel jobs |

---

## Documentation

| File | Purpose |
|------|---------|
| **STATUS.md** | Current implementation status |
| **ROADMAP.md** | Development priorities and plans |
| **CLAUDE.md** | Coding guidelines (for developers) |

---

## Getting Help

```bash
# Check version
connectomix --version

# Get help
connectomix --help
```

**Links:**
- [GitHub Repository](https://github.com/ln2t/connectomix)
- [Report Issues](https://github.com/ln2t/connectomix/issues)

---

## License

[License information here]

---

## Citation

If you use Connectomix in your research, please cite:

```
[Citation information here]
```
