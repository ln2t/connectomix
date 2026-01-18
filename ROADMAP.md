# Connectomix Development Roadmap

**Last Updated**: January 18, 2026  
**Version**: 3.0.0

## Current Status

**Connectomix v3.0.0 is feature-complete (~98%).**

All core functionality is implemented and working:
- ✅ Configuration system with validation
- ✅ BIDS I/O and file discovery
- ✅ Preprocessing (resampling, denoising, temporal censoring)
- ✅ All four connectivity methods
- ✅ Four connectivity measures (correlation, covariance, partial correlation, precision)
- ✅ Statistics (GLM, permutation testing, thresholding, clustering)
- ✅ Pipeline orchestration (participant and group levels)
- ✅ HTML report generation with visualizations
- ✅ Atlas management
- ✅ CLI with comprehensive options
- ✅ README documentation with connectivity measures guide

See **STATUS.md** for detailed module breakdown.

---

## Remaining Work

### Priority 1: Example Configuration Files
**Location**: `examples/`

Create ready-to-use configuration files demonstrating common use cases:

1. **`participant_roi_to_roi.yaml`** - Basic ROI-to-ROI with Schaefer atlas
2. **`participant_seed_to_voxel.yaml`** - Seed-based analysis with DMN seeds
3. **`participant_task_conditions.yaml`** - Task fMRI with condition selection
4. **`group_two_groups.yaml`** - Patient vs control comparison
5. **`group_correlation.yaml`** - Continuous covariate (e.g., age)

### Priority 2: Documentation Polish

1. **License** - Add license information to README
2. **Citation** - Add citation format
3. **Changelog** - Document version history

### Priority 3: Testing (Future)

1. **Unit tests** for core functions
2. **Integration tests** with small test dataset
3. **CI/CD pipeline** for automated testing

---

## Future Enhancements (v3.1+)

### Potential Features

1. **Surface-based connectivity**
   - Support for FreeSurfer/fMRIPrep surface outputs
   - fsaverage space connectivity

2. **Dynamic connectivity**
   - Sliding window analysis
   - Phase-based connectivity

3. **Graph theory metrics**
   - Network measures (degree, clustering, modularity)
   - Hub detection

4. **Machine learning integration**
   - Feature extraction for classification
   - Connectivity-based prediction

5. **Interactive visualization**
   - Web-based connectivity explorer
   - 3D brain rendering

6. **Parallel processing**
   - Multi-subject parallelization
   - GPU acceleration for large datasets

---

## Design Decisions

### Why Temporal Censoring?

Task fMRI connectivity requires analyzing specific experimental conditions. The temporal censoring system allows:
- Computing connectivity for each condition separately
- Removing high-motion timepoints to improve data quality
- Dropping dummy scans for better signal stability

### Why Check ALL Subjects for Geometry?

Group-level analysis requires all subjects to have identical spatial geometry. By checking ALL subjects upfront (not just selected ones), we prevent failures during group analysis that would waste hours of computation.

### Why HTML Reports?

Self-contained HTML reports with embedded figures are:
- Easily shareable without external dependencies
- Viewable in any web browser
- Include both visualizations and downloadable data

---

## Contributing

See **CLAUDE.md** for coding guidelines:
- Type hints on all functions
- Google-style docstrings
- pathlib.Path for file paths
- Custom exceptions for errors
- Comprehensive logging

---

## Version History

### v3.0.0 (January 2026)
- Complete rewrite of Connectomix
- Four connectivity methods (seed-to-voxel, ROI-to-voxel, seed-to-seed, ROI-to-ROI)
- **Four connectivity measures** for ROI-to-ROI: correlation, covariance, partial correlation, precision
- Time series extraction and saving (*.npy files)
- Participant and group-level pipelines
- Temporal censoring for task fMRI
- Motion scrubbing (FD-based)
- Comprehensive HTML reports with:
  - Connectivity matrices with theoretical explanations
  - Connectome glass brain visualizations
  - Denoising QA histograms (before/after comparison)
  - Downloadable figures
- BIDS-compliant I/O
- 12 built-in atlases

---

*For the latest implementation status, see STATUS.md.*
