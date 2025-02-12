import matplotlib.pyplot as plt
import importlib.resources

from connectomix.core.core import participant_level_pipeline, group_level_pipeline
from connectomix.tests.paths import bids_dir, fmriprep_dir, output_dir

precuneus_L_mask = str(importlib.resources.files("connectomix.tests.data").joinpath("AAL_Precuneus_L.nii.gz"))
precentral_R_mask = str(importlib.resources.files("connectomix.tests.data").joinpath("AAL_Precentral_R.nii.gz"))

method = "roiToVoxel"
roi_masks = {"precentralR": precentral_R_mask,
             "precuneusL": precuneus_L_mask}

# Participant
participant_level_pipeline(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": method,
                                   "roi_masks": roi_masks})

plt.close('all')

# Group
## One-sample t-test
group_level_pipeline(bids_dir,
                     output_dir,
                     config={"method": method,
                             "roi_masks": roi_masks,
                             "analysis_name": "testMeanEffect",
                             "contrast": "intercept"})

plt.close('all')

## Two-sample t-test, unpaired
group_level_pipeline(bids_dir,
                     output_dir,
                     config={"method": method,
                             "roi_masks": roi_masks,
                             "analysis_name": "testControlVersusPatients",
                             "covariates" : "group",
                             "contrast": "control-patient",
                             "add_intercept": False})

plt.close('all')

## Regression
group_level_pipeline(bids_dir,
                     output_dir,
                     config={"method": method,
                             "roi_masks": roi_masks,
                             "analysis_name": "testEffectOfAge",
                             "covariates" : "age",
                             "contrast": "age"})

plt.close('all')