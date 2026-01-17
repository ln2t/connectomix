# Documentation Index

This directory contains streamlined documentation for the Connectomix v3.0.0 rewrite project.

## Documentation Files

### 📋 **STATUS.md** - Implementation Status
Current progress report showing:
- What's been completed (23 files, ~2935 lines, ~65%)
- What remains to be done (~3100 lines, ~35%)
- Module-by-module breakdown with file counts and line counts
- Code quality metrics and key features

**Use this to**: Understand what's already working and what's left to implement.

---

### 🗺️ **ROADMAP.md** - Development Plan
Detailed implementation roadmap for remaining work:
- Statistics module (GLM, permutation testing, thresholding, clustering)
- Pipeline orchestration (participant and group pipelines)
- Visualization and reporting (plots and HTML generation)
- Atlas management
- Documentation (README and examples)

Each section includes:
- Purpose and key functions
- Implementation estimates (~line counts)
- Dependencies and integration points
- Testing strategy

**Use this to**: Plan and execute the remaining development work.

---

### ⚡ **QUICKSTART.md** - Usage Reference
Quick reference guide with:
- Command-line examples for common use cases
- Configuration file templates for all analysis types
- Parameter reference table
- Available atlases and denoising strategies
- Output structure
- Troubleshooting tips

**Use this to**: Get started quickly with Connectomix once pipelines are complete.

---

### 🔧 **CLAUDE.md** - Coding Guidelines
Comprehensive coding standards and implementation patterns:
- Project organization strategy
- Python best practices (type hints, dataclasses, pathlib)
- Module-specific implementation templates
- Common patterns and pitfalls to avoid
- Testing strategy (future)

**Use this to**: Maintain consistent code style and quality during development.

---

## Documentation Structure Rationale

This streamlined structure eliminates duplication from the original 6 documentation files:

**Removed files** (content consolidated):
- `SESSION_SUMMARY.md` → merged into **STATUS.md**
- `PROGRESS_REPORT.md` → merged into **STATUS.md**
- `PROJECT_SUMMARY.md` → split between **STATUS.md** and **ROADMAP.md**
- `IMPLEMENTATION_PLAN.md` → cleaned and updated as **ROADMAP.md**
- `QUICK_REFERENCE.md` → reorganized as **QUICKSTART.md**
- `NEXT_STEPS.md` → merged into **ROADMAP.md**

**Result**: 4 focused, non-redundant documentation files with clear purposes.

---

## Quick Navigation

**I want to...**

- **See what's been built** → Read [STATUS.md](STATUS.md)
- **Continue development** → Read [ROADMAP.md](ROADMAP.md)
- **Learn how to use Connectomix** → Read [QUICKSTART.md](QUICKSTART.md)
- **Understand coding standards** → Read [CLAUDE.md](CLAUDE.md)

---

## Project Overview

**Connectomix v3.0.0** is a complete rewrite of the functional connectivity analysis tool.

### Key Features
- BIDS-compliant with fMRIPrep integration
- Four connectivity methods: seed-to-voxel, ROI-to-voxel, seed-to-seed, ROI-to-ROI
- Participant-level and group-level analysis pipelines
- Flexible preprocessing with predefined denoising strategies
- Statistical inference with multiple comparison correction
- HTML reports with quality assurance metrics

### Technology Stack
- Python 3.8+
- Nilearn for neuroimaging
- PyBIDS for BIDS compliance
- Nibabel for NIfTI I/O
- NumPy, Pandas, SciPy for data processing

### Current Status
**~65% complete** - Core infrastructure is production-ready. Remaining work focuses on statistics, pipeline integration, and reporting.

---

For the comprehensive user guide and installation instructions, see **README.md** (coming soon).
