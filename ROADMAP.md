# Connectomix Development Roadmap

This document outlines the remaining work to complete the Connectomix v3.0.0 rewrite.

## Current Status

**Completed**: ~65% (23/33 files, ~2935/6000 lines)

**Core infrastructure is complete**:
- ✅ Configuration system
- ✅ Utilities (logging, validation, matrix operations)
- ✅ BIDS I/O
- ✅ Preprocessing (resampling, denoising, CanICA)
- ✅ All four connectivity methods
- ✅ CLI and entry point

**See STATUS.md for detailed completion status.**

---

## Remaining Implementation Priorities

### Priority 1: Statistics Module (~600 lines)

Location: `connectomix/statistics/`

#### `glm.py` - Second-Level GLM (~200 lines)
**Purpose**: Fit group-level general linear models and compute contrasts

Key functions:
- `build_design_matrix(participants_df, covariates, add_intercept)` → DataFrame
  - Create design matrix from participants.tsv
  - Handle categorical and continuous covariates
  - Add intercept column if requested
  
- `fit_second_level_model(maps, design_matrix, smoothing)` → SecondLevelModel
  - Use nilearn.glm.SecondLevelModel
  - Apply spatial smoothing
  - Fit GLM across subjects
  
- `compute_contrast(model, contrast_def)` → nib.Nifti1Image
  - Compute t-statistic maps
  - Support simple contrasts ("covariate", "intercept")
  - Support complex contrasts ([1, -1, 0] for group differences)

**Dependencies**: nilearn.glm.SecondLevelModel, pandas, statsmodels

---

#### `permutation.py` - Permutation Testing (~150 lines)
**Purpose**: Family-wise error (FWE) correction via non-parametric permutation testing

Key function:
- `run_permutation_test(maps, design_matrix, contrast, n_permutations, n_jobs)` → dict
  - Generate null distribution via label permutation
  - Compute max statistic for each permutation
  - Return: {'null_distribution': array, 'threshold': float, 'p_values': nifti}
  
**Dependencies**: nilearn.glm, numpy, joblib for parallelization

---

#### `thresholding.py` - Multiple Comparison Correction (~150 lines)
**Purpose**: Apply various thresholding strategies to statistical maps

Key functions:
- `apply_uncorrected_threshold(stat_map, alpha, two_sided)` → nib.Nifti1Image
  - Simple t/z threshold
  
- `apply_fdr_threshold(stat_map, alpha, two_sided)` → nib.Nifti1Image
  - False Discovery Rate correction (Benjamini-Hochberg)
  
- `apply_fwe_threshold(stat_map, null_distribution, alpha)` → nib.Nifti1Image
  - Use permutation null distribution
  - Threshold at (1-alpha) quantile
  
- `apply_cluster_threshold(stat_map, cluster_alpha, extent_threshold)` → nib.Nifti1Image
  - Cluster-forming threshold with extent

**Dependencies**: nilearn.glm, scipy.stats, nilearn.image

---

#### `clustering.py` - Cluster Analysis (~100 lines)
**Purpose**: Extract cluster information and add anatomical labels

Key functions:
- `get_cluster_table(thresholded_map, stat_map, min_cluster_size)` → DataFrame
  - Use nilearn.reporting.get_clusters_table
  - Columns: Cluster ID, Size (voxels), Peak Coordinates, Peak Value
  
- `add_anatomical_labels(cluster_table, atlas_name)` → DataFrame
  - Add "Region" column with anatomical labels
  - Support AAL, Harvard-Oxford atlases

**Dependencies**: nilearn.reporting, nilearn.datasets

---

### Priority 2: Pipeline Orchestration (~800 lines)

Location: `connectomix/core/`

#### `participant.py` - Participant Pipeline (~500 lines)
**Purpose**: Orchestrate entire participant-level analysis

Main function: `run_participant_pipeline(bids_dir, output_dir, config, logger)`

Pipeline steps:
1. Validate BIDS directory and derivatives
2. Create BIDS layout
3. Query fMRIPrep files for selected participants
4. Check geometric consistency (ALL subjects, not just selected)
5. Resample to reference if needed
6. Denoise functional images
7. Generate CanICA atlas if method='roiToRoi' and atlas='canica'
8. Compute connectivity (dispatch to appropriate method)
9. Save outputs with BIDS-compliant names and JSON sidecars
10. Generate HTML report for each participant

**Key features**:
- Progress tracking with timer and logging
- Skip already processed subjects (check output files)
- Detailed error reporting per subject
- Configuration backup to output directory

---

#### `group.py` - Group Pipeline (~300 lines)
**Purpose**: Orchestrate group-level statistical analysis

Main function: `run_group_pipeline(bids_dir, output_dir, config, logger)`

Pipeline steps:
1. Validate BIDS directory and participant-level outputs
2. Validate geometric consistency across subjects
3. Query participant-level connectivity maps
4. Load participants.tsv and build design matrix
5. Smooth maps if config.smoothing > 0
6. Fit second-level GLM
7. Compute contrast
8. Run permutation testing (if 'fwe' in thresholding_strategies)
9. Apply thresholding strategies (uncorrected, FDR, FWE)
10. Extract cluster tables for thresholded maps
11. Save all outputs (stat maps, thresholded maps, cluster tables)
12. Generate HTML report

**Key features**:
- Validate all subjects have same geometry
- Ensure all subjects have required participant-level outputs
- Save design matrix as TSV with sidecar
- Comprehensive logging of statistical results

---

### Priority 3: Visualization and Reporting (~500 lines)

Location: `connectomix/utils/`

#### `visualization.py` - Plotting Functions (~200 lines)
**Purpose**: Generate plots for reports

Key functions:
- `plot_design_matrix(design_matrix, output_path)` → Path
  - Heatmap with subject rows, covariate columns
  - Use seaborn for aesthetics
  
- `plot_connectivity_matrix(matrix, labels, output_path)` → Path
  - Correlation/covariance matrix heatmap
  - Optional: hierarchical clustering reordering
  
- `plot_stat_map(stat_map, threshold, output_path, title)` → Path
  - Glass brain plot using nilearn.plotting.plot_glass_brain
  - Stat map overlay using nilearn.plotting.plot_stat_map
  
- `plot_seeds(seeds_coords, background_img, output_path)` → Path
  - Visualize seed locations on anatomical template

**Dependencies**: matplotlib, seaborn, nilearn.plotting

---

#### `reports.py` - HTML Report Generation (~300 lines)
**Purpose**: Generate comprehensive HTML reports

Class: `HTMLReportGenerator`

Methods:
- `__init__(title, output_path)`
- `add_section(heading, content)` - Add text/HTML section
- `add_parameters(config_dict)` - Add parameter table
- `add_image(image_path, caption)` - Embed image
- `add_connectivity_matrix_plot(matrix, labels)` - Plot and embed matrix
- `add_stat_map_plot(stat_map, threshold)` - Plot and embed brain map
- `add_qa_metrics(metrics_dict)` - Add quality assurance table
- `add_cluster_table(cluster_df)` - Add cluster results table
- `generate()` - Write HTML file

**Report sections**:
- **Participant report**: Parameters, QA metrics (tSNR, noise reduction), connectivity matrix/maps, seed locations
- **Group report**: Parameters, design matrix, contrast definition, statistical results, thresholded maps, cluster tables

**Dependencies**: jinja2 or simple string formatting, base64 for image embedding

---

### Priority 4: Atlas Management (~200 lines)

Location: `connectomix/data/`

#### `atlases.py` - Atlas Loading (~200 lines)
**Purpose**: Centralized atlas management

Key components:
- `ATLAS_REGISTRY` - Dict mapping atlas names to nilearn dataset functions
  - "aal" → nilearn.datasets.fetch_atlas_aal
  - "schaefer2018_100" → nilearn.datasets.fetch_atlas_schaefer_2018 (n_rois=100)
  - "schaefer2018_200" → nilearn.datasets.fetch_atlas_schaefer_2018 (n_rois=200)
  - "harvardoxford" → nilearn.datasets.fetch_atlas_harvard_oxford

Functions:
- `load_atlas(atlas_name)` → Tuple[nib.Nifti1Image, List[str]]
  - Fetch atlas using nilearn
  - Return (atlas_image, region_labels)
  
- `list_available_atlases()` → List[str]
  - Return all available atlas names
  
- `validate_atlas_name(atlas_name)` → bool
  - Check if atlas exists in registry

**Dependencies**: nilearn.datasets

---

### Priority 5: Documentation (~1000 lines)

#### Comprehensive README.md (~800 lines)

**Structure**:
1. **Introduction**
   - What is Connectomix?
   - Key features
   - Citation

2. **Installation**
   - Requirements (Python 3.8+)
   - pip install instructions
   - Development installation

3. **Quick Start**
   - Minimal working example
   - Expected outputs

4. **Detailed Usage**
   - CLI overview
   - Configuration files (JSON/YAML)
   - BIDS dataset requirements
   - fMRIPrep prerequisites

5. **Analysis Methods**
   - Seed-to-Voxel (with example config)
   - ROI-to-Voxel (with example config)
   - Seed-to-Seed (with example config)
   - ROI-to-ROI (with example config)

6. **Preprocessing and Denoising**
   - Geometric consistency checking
   - Resampling
   - Confound regression
   - Temporal filtering
   - Predefined strategies

7. **Group-Level Analysis**
   - Design matrix creation
   - Contrasts
   - Multiple comparison correction
   - Cluster analysis

8. **Outputs and Interpretation**
   - Directory structure
   - Output file naming (BIDS)
   - JSON sidecars
   - HTML reports

9. **Advanced Topics**
   - Custom atlases
   - CanICA data-driven parcellation
   - Parallelization

10. **Troubleshooting**
    - Common errors and solutions
    - BIDS validation
    - Geometry mismatches

11. **Citations**
    - Nilearn
    - fMRIPrep
    - BIDS
    - Relevant papers

12. **Appendices**
    - Configuration reference (all parameters)
    - Atlas descriptions
    - BIDS entities used

---

#### Example Configuration Files (~200 lines)

Create in `examples/` directory:

1. **`participant_seed_to_voxel.yaml`**
   - Single seed (PCC)
   - Basic denoising
   - Comments explaining each parameter

2. **`participant_roi_to_roi.yaml`**
   - Schaefer 100 atlas
   - Moderate denoising
   - Comments on atlas options

3. **`participant_multiple_seeds.yaml`**
   - Multiple seeds (DMN nodes)
   - Advanced denoising
   - Quality thresholds

4. **`group_two_groups.yaml`**
   - Categorical covariate (group)
   - Contrast: group1 - group2
   - All thresholding strategies

5. **`group_correlation_with_age.yaml`**
   - Continuous covariate (age)
   - Intercept + age contrast
   - FWE permutation testing

Each example should:
- Be fully functional
- Include inline comments
- Show common use cases
- Demonstrate best practices

---

## Implementation Guidelines

### General Principles
1. **Follow existing patterns**: All completed modules provide templates
2. **Type hints everywhere**: 100% coverage as in existing code
3. **Google-style docstrings**: Complete with Args, Returns, Raises sections
4. **Error handling**: Use custom exceptions with actionable messages
5. **Logging**: Use existing logger with structured sections
6. **Testing**: Manually test each function before moving on

### Coding Standards (from CLAUDE.md)
- Use `pathlib.Path` for all file paths
- Use dataclasses for configuration
- Use context managers where appropriate
- Validate inputs early
- Log important steps at INFO level
- Log details at DEBUG level

### Dependencies
All required packages are already in `requirements.txt`:
- nibabel >=5.2.0
- nilearn >=0.10.3
- numpy >=1.24.0
- pandas >=2.0.0
- pybids >=0.16.4
- PyYAML >=6.0
- scipy >=1.10.0
- statsmodels >=0.14.0
- matplotlib >=3.7.0
- seaborn >=0.12.0

### Integration Points

**Statistics → Pipelines**:
- `core/group.py` imports from `statistics/`
- Uses `fit_second_level_model()`, `run_permutation_test()`, `apply_*_threshold()`, `get_cluster_table()`

**Visualization → Reporting**:
- `utils/reports.py` imports from `utils/visualization.py`
- Uses plotting functions to generate embedded images

**Atlases → Connectivity**:
- `connectivity/roi_to_roi.py` and `connectivity/roi_to_voxel.py` import from `data/atlases.py`
- Use `load_atlas()` for ROI definitions

**Pipelines → Everything**:
- `core/participant.py` orchestrates: config, io, preprocessing, connectivity, reporting
- `core/group.py` orchestrates: config, io, statistics, visualization, reporting

---

## Estimated Timeline

- **Days 1-2**: Statistics module (4 files)
- **Days 3-5**: Pipeline orchestration (2 files)
- **Days 6-7**: Visualization and reporting (2 files)
- **Day 8**: Atlas management (1 file)
- **Days 9-12**: Documentation (README + examples)
- **Days 13-14**: Testing, refinement, polish

**Total**: ~2 weeks for completion

---

## Testing Strategy (Post-Implementation)

### Manual Testing Checklist
- [ ] Participant-level seed-to-voxel analysis runs successfully
- [ ] Participant-level ROI-to-ROI analysis runs successfully
- [ ] Group-level analysis with two groups runs successfully
- [ ] Group-level analysis with continuous covariate runs successfully
- [ ] HTML reports generated correctly
- [ ] All thresholding strategies work
- [ ] CanICA atlas generation works
- [ ] Configuration validation catches errors
- [ ] BIDS compliance in all outputs

### Test Data
Use small synthetic or open-source datasets:
- fMRIPrep output from OpenNeuro
- Create minimal test dataset with 3-5 subjects
- Test with different BIDS entity combinations

---

## Next Immediate Step

**Start with `statistics/glm.py`** - Build design matrices and fit second-level models.

See STATUS.md for what's already complete and can be used as reference.
