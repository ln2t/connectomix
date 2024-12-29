#!/usr/bin/env python3
"""BIDS app to compute connectomes from fmri data preprocessed with FMRIPrep

Author: Antonin Rovai

Created: August 2022
"""

# TODO list:
# - add more unittests functions
# - create more test datasets for group-level analysis, in particular featuring:
# --- Independent samples testing DONE
# --- Paired samples testing: inter-session OR inter-task OR inter-run comparison
# --- Regression: covariate and confounds removal
# - include plot of null distribution of max stat in report
# - roi-to-voxel analyzes
# - cluster-based inference (mass or size)

# Restructure the config file as follows:
# method: seed-based or roi-to-roi or ICA or ReHo or ...

from connectomix.tools.tools import main
if __name__ == "__main__":
    main()
