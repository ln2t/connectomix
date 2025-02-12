import matplotlib.pyplot as plt
import importlib.resources

from connectomix.core.core import participant_level_pipeline, group_level_pipeline
from connectomix.tests.paths import bids_dir, fmriprep_dir, output_dir

example_seeds_for_seedToVoxel = str(importlib.resources.files("connectomix.tests.data").joinpath("example_seeds_for_seedToVoxel.tsv"))

method = "seedToVoxel"

# Participant
participant_level_pipeline(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": method,
                                   "seeds_file": example_seeds_for_seedToVoxel})

plt.close('all')

# Group
## One-sample t-test
group_level_pipeline(bids_dir,
                     output_dir,
                     config={"method": method,
                             "seeds_file": example_seeds_for_seedToVoxel,
                             "analysis_name": "testMeanEffect",
                             "contrast": "intercept"})

plt.close('all')

## Two-sample t-test, unpaired
group_level_pipeline(bids_dir,
                     output_dir,
                     config={"method": method,
                             "seeds_file": example_seeds_for_seedToVoxel,
                             "analysis_name": "testControlVersusPatients",
                             "covariates" : "group",
                             "contrast": "control-patient",
                             "add_intercept": False})

plt.close('all')

## Regression
group_level_pipeline(bids_dir,
                     output_dir,
                     config={"method": method,
                             "seeds_file": example_seeds_for_seedToVoxel,
                             "analysis_name": "testEffectOfAge",
                             "covariates" : "age",
                             "contrast": "age"})

plt.close('all')