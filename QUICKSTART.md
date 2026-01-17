# Connectomix Quick Start Guide

Fast reference for common Connectomix usage patterns.

## Installation

```bash
# From source (during development)
cd connectomix_AI
pip install -e .

# Required: Python 3.8+, fMRIPrep preprocessed data
```

## Basic Commands

```bash
# Participant-level analysis (simplest)
connectomix /data/bids /data/output participant

# With specific subject and task
connectomix /data/bids /data/output participant -p 01 -t rest

# With configuration file
connectomix /data/bids /data/output participant -c config.yaml

# Group-level analysis
connectomix /data/bids /data/output group -c group_config.yaml

# Verbose output for debugging
connectomix /data/bids /data/output participant -v

# Specify fMRIPrep location (if not in bids_dir/derivatives/fmriprep)
connectomix /data/bids /data/output participant \
  --derivatives fmriprep=/path/to/fmriprep
```

## Common Arguments

| Argument | Short | Description | Example |
|----------|-------|-------------|---------|
| `--participant_label` | `-p` | Subject(s) to process | `-p 01 02 03` |
| `--task` | `-t` | Task name to process | `-t restingstate` |
| `--session` | `-s` | Session to process | `-s 1` |
| `--run` | `-r` | Run to process | `-r 1` |
| `--space` | - | MNI space to use | `--space MNI152NLin2009cAsym` |
| `--config` | `-c` | Config file path | `-c my_config.yaml` |
| `--denoising` | - | Predefined strategy | `--denoising csfwm_6p` |
| `--derivatives` | `-d` | Derivative locations | `-d fmriprep=/path` |
| `--verbose` | `-v` | Enable debug output | `-v` |

## Configuration File Examples

### Participant: Seed-to-Voxel

```yaml
# config_seed.yaml
# BIDS filters
subject: ["01", "02", "03"]
tasks: ["restingstate"]
sessions: null
spaces: ["MNI152NLin2009cAsym"]

# Method
method: "seedToVoxel"
seeds_file: "seeds.tsv"
radius: 5.0

# Denoising
confounds: ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf_wm"]
high_pass: 0.01
low_pass: 0.08
```

**Seeds file (seeds.tsv)**:
```tsv
name    x       y       z
PCC     0       -52     18
mPFC    0       52      0
LIPL    -45     -70     35
```

---

### Participant: ROI-to-ROI

```yaml
# config_roi.yaml
subject: ["01", "02", "03"]
tasks: ["restingstate"]
spaces: ["MNI152NLin2009cAsym"]

# Method
method: "roiToRoi"
atlas: "schaefer2018_100"
connectivity_kind: "correlation"

# Denoising
confounds: ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf_wm"]
high_pass: 0.01
low_pass: 0.08
```

---

### Group: Two-Group Comparison

```yaml
# group_config.yaml
# Participants
subject: ["01", "02", "03", "04", "05"]
task: "restingstate"
session: null
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
contrast: "group"  # or for group difference: [1, -1] if intercept=false

# Statistics
uncorrected_alpha: 0.001
fdr_alpha: 0.05
fwe_alpha: 0.05
two_sided_test: true
thresholding_strategies: ["uncorrected", "fdr", "fwe"]

# Computational
n_permutations: 10000
n_jobs: 4
```

**participants.tsv** must contain:
```tsv
participant_id  group
sub-01          patient
sub-02          patient
sub-03          control
sub-04          control
sub-05          control
```

---

### Group: Correlation with Continuous Variable

```yaml
# group_age_config.yaml
subject: ["01", "02", "03", "04", "05", "06", "07", "08"]
task: "restingstate"
space: "MNI152NLin2009cAsym"

method: "seedToVoxel"
smoothing: 6.0

analysis_name: "age_correlation"

# Include age from participants.tsv
covariates: ["age"]
add_intercept: true

contrast: "age"

uncorrected_alpha: 0.001
fdr_alpha: 0.05
fwe_alpha: 0.05
two_sided_test: true
thresholding_strategies: ["fdr", "fwe"]

n_permutations: 5000
n_jobs: 8
```

## Predefined Denoising Strategies

Use with `--denoising STRATEGY` or `denoising: "STRATEGY"` in config:

| Strategy | Confounds | Description |
|----------|-----------|-------------|
| `minimal` | 6 motion parameters | Basic motion correction only |
| `csfwm_6p` | CSF+WM signal + 6 motion | Standard denoising |
| `gs_csfwm_12p` | Global + CSF + WM + 12 motion | Aggressive (includes derivatives) |
| `ica_aroma` | ICA-AROMA components | Uses fMRIPrep's AROMA |
| `custom` | Define in `confounds` list | Full control |

## Available Atlases

| Name | Regions | Description |
|------|---------|-------------|
| `aal` | 116 | Automated Anatomical Labeling |
| `schaefer2018_100` | 100 | Schaefer 7-network 100 parcels |
| `schaefer2018_200` | 200 | Schaefer 7-network 200 parcels |
| `harvardoxford` | 96 | Harvard-Oxford cortical + subcortical |
| `canica` | Custom | Data-driven ICA (computed from your data) |

## Output Structure

```
output_dir/
├── dataset_description.json          # BIDS derivative metadata
├── config/
│   └── backups/
│       └── config_TIMESTAMP.json     # Configuration backups
├── sub-01/
│   ├── sub-01_task-rest_space-MNI_desc-resampled_bold.nii.gz
│   ├── sub-01_task-rest_space-MNI_desc-denoised_bold.nii.gz
│   ├── sub-01_task-rest_space-MNI_desc-geometry.json
│   ├── sub-01_task-rest_method-seedToVoxel_seed-PCC_effectSize.nii.gz
│   ├── sub-01_task-rest_method-roiToRoi_atlas-schaefer_correlation.npy
│   ├── sub-01_task-rest_method-roiToRoi_atlas-schaefer_correlation.json
│   └── sub-01_task-rest_method-roiToRoi_report.html
├── sub-02/
│   └── ...
└── group/
    └── roiToRoi/
        └── patients_vs_controls/
            ├── task-rest_method-roiToRoi_analysis-patientsVsControls_designMatrix.tsv
            ├── task-rest_method-roiToRoi_analysis-patientsVsControls_stat-t.nii.gz
            ├── task-rest_method-roiToRoi_analysis-patientsVsControls_threshold-fdr_stat-t.nii.gz
            ├── task-rest_method-roiToRoi_analysis-patientsVsControls_threshold-fwe_stat-t.nii.gz
            ├── task-rest_method-roiToRoi_analysis-patientsVsControls_clusterTable.tsv
            └── task-rest_method-roiToRoi_analysis-patientsVsControls_report.html
```

## Four Analysis Methods

### 1. Seed-to-Voxel
**Use case**: Examine connectivity between a specific brain region (seed) and all other voxels

**Config**:
```yaml
method: "seedToVoxel"
seeds_file: "seeds.tsv"  # Tab-separated: name, x, y, z
radius: 5.0              # Sphere radius in mm
```

**Output**: One NIfTI per seed with correlation values at each voxel

---

### 2. ROI-to-Voxel
**Use case**: Like seed-to-voxel but with arbitrary ROI masks instead of spheres

**Config**:
```yaml
method: "roiToVoxel"
roi_masks: ["/path/to/roi1.nii.gz", "/path/to/roi2.nii.gz"]
```

**Output**: One NIfTI per ROI with correlation values

---

### 3. Seed-to-Seed
**Use case**: Correlation matrix between multiple seeds

**Config**:
```yaml
method: "seedToSeed"
seeds_file: "seeds.tsv"
radius: 5.0
```

**Output**: Correlation matrix (numpy array) and visualization

---

### 4. ROI-to-ROI
**Use case**: Whole-brain parcellation-based connectivity matrix

**Config**:
```yaml
method: "roiToRoi"
atlas: "schaefer2018_100"
connectivity_kind: "correlation"  # or "covariance"
```

**Output**: NxN correlation matrix where N = number of atlas regions

## Common Workflows

### Workflow 1: Basic Resting-State Analysis

```bash
# 1. Participant-level
connectomix /data/bids /data/output participant \
  -c participant_config.yaml -v

# 2. Check HTML reports in /data/output/sub-*/

# 3. Group-level
connectomix /data/bids /data/output group \
  -c group_config.yaml -v

# 4. Check group results in /data/output/group/
```

### Workflow 2: Multiple Seeds Analysis

```bash
# Create seeds.tsv with DMN nodes
# Run participant-level
connectomix /data/bids /data/output participant \
  --task rest --method seedToVoxel --seeds_file dmn_seeds.tsv
```

### Workflow 3: Data-Driven Parcellation

```yaml
# In config.yaml
method: "roiToRoi"
atlas: "canica"
n_components: 20
```

## Troubleshooting

### Error: "Geometric consistency check failed"
**Solution**: Ensure all subjects have the same MNI space. Connectomix will automatically resample if geometries don't match.

### Error: "Confound not found in confounds file"
**Solution**: Check fMRIPrep's confounds TSV. Use `--denoising minimal` for basic motion parameters only.

### Error: "No functional files found"
**Solution**: Check BIDS entity filters. Use `-v` to see query details.

### Slow permutation testing
**Solution**: Reduce `n_permutations` (e.g., 5000) or increase `n_jobs` for parallelization.

## Tips and Best Practices

1. **Start small**: Test with 1-2 subjects before full dataset
2. **Use verbose mode** (`-v`) when debugging
3. **Check HTML reports** for quality assurance
4. **Geometric consistency**: Let Connectomix handle resampling automatically
5. **Denoising**: Start with `csfwm_6p`, adjust based on your data quality
6. **Group analysis**: Ensure participants.tsv has all required covariates
7. **Permutations**: 10000 is good for publication, 5000 for exploration
8. **Smoothing**: 6-8mm FWHM is typical for group analysis

## Configuration Parameters Reference

### Participant-Level Core Parameters
- `subject`: List of subject IDs (without "sub-" prefix)
- `tasks`: List of task names
- `sessions`: List of session IDs
- `runs`: List of run IDs
- `spaces`: List of MNI spaces
- `method`: "seedToVoxel" | "roiToVoxel" | "seedToSeed" | "roiToRoi"

### Denoising Parameters
- `confounds`: List of confound column names from fMRIPrep
- `high_pass`: High-pass filter cutoff in Hz (default: 0.01)
- `low_pass`: Low-pass filter cutoff in Hz (default: 0.08)
- `ica_aroma`: Use ICA-AROMA denoising (bool)

### Method-Specific Parameters
- `seeds_file`: Path to seeds TSV (for seed-based methods)
- `radius`: Seed sphere radius in mm (default: 5.0)
- `roi_masks`: List of ROI NIfTI paths (for ROI-to-voxel)
- `atlas`: Atlas name (for ROI-to-ROI)
- `connectivity_kind`: "correlation" | "covariance"

### CanICA Parameters
- `n_components`: Number of ICA components (default: 20)
- `canica_threshold`: Threshold for RegionExtractor (default: 1.0)

### Group-Level Core Parameters
- `analysis_name`: Name for this analysis
- `smoothing`: FWHM in mm for spatial smoothing
- `covariates`: List of column names from participants.tsv
- `add_intercept`: Add intercept to design matrix (bool)
- `contrast`: Contrast specification (string or list)

### Statistical Parameters
- `uncorrected_alpha`: Alpha for uncorrected threshold (default: 0.001)
- `fdr_alpha`: Alpha for FDR correction (default: 0.05)
- `fwe_alpha`: Alpha for FWE correction (default: 0.05)
- `two_sided_test`: Two-sided vs one-sided test (bool)
- `thresholding_strategies`: List of strategies to apply
- `n_permutations`: Number of permutations for FWE (default: 10000)
- `n_jobs`: Number of parallel jobs (default: 1)

## Version and Help

```bash
# Check version
connectomix --version

# Get help
connectomix --help
connectomix participant --help
connectomix group --help
```

## Additional Resources

- **STATUS.md** - Current implementation status
- **ROADMAP.md** - Development priorities
- **CLAUDE.md** - Coding guidelines (for developers)
- **README.md** - Comprehensive documentation (coming soon)

---

For more detailed information, see the full documentation in README.md.
