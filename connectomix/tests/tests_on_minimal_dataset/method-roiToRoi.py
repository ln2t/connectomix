import matplotlib.pyplot as plt
import importlib.resources

from connectomix.utils.modes import participant_level_analysis, group_level_analysis
from connectomix.tests.paths import bids_dir, fmriprep_dir, output_dir

method = "roiToRoi"

# Participant
## atlas-based
for atlas in ["aal", "schaeffer100", "harvardoxford"]:
    participant_level_analysis(bids_dir,
                               output_dir,
                               derivatives={"fmriprep": fmriprep_dir},
                               config={"method": method,
                                       "atlas": atlas})

## canica
participant_level_analysis(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": method,
                                   "canica": True})

# # Group
# ## One-sample t-test
# group_level_analysis(bids_dir,
#                      output_dir,
#                      config={"method": "roiToVoxel",
#                              "roi_masks": {"precentralR": precentral_R_mask,
#                                            "precuneusL": precuneus_L_mask},
#                              "analysis_name": "testMeanEffect",
#                              "contrast": "intercept"})
#
# ## Two-sample t-test, unpaired
# group_level_analysis(bids_dir,
#                      output_dir,
#                      config={"method": "roiToVoxel",
#                              "roi_masks": {"precentralR": precentral_R_mask,
#                                            "precuneusL": precuneus_L_mask},
#                              "analysis_name": "testControlVersusPatients",
#                              "covariates" : "group",
#                              "contrast": "control-patient",
#                              "add_intercept": False})
#
# ## Regression
# group_level_analysis(bids_dir,
#                      output_dir,
#                      config={"method": "roiToVoxel",
#                              "roi_masks": {"precentralR": precentral_R_mask,
#                                            "precuneusL": precuneus_L_mask},
#                              "analysis_name": "testEffectOfAge",
#                              "covariates" : "age",
#                              "contrast": "age"})

plt.close('all')