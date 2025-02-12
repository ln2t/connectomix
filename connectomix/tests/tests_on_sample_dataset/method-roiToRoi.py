import matplotlib.pyplot as plt

from connectomix.core.core import participant_level_pipeline, group_level_pipeline
from connectomix.tests.paths import bids_dir, fmriprep_dir, output_dir

method = "roiToRoi"

# Participant
## atlas-based
for atlas in ["aal", "schaeffer100", "harvardoxford"]:
    participant_level_pipeline(bids_dir,
                               output_dir,
                               derivatives={"fmriprep": fmriprep_dir},
                               config={"method": method,
                                       "atlas": atlas})

## canica
participant_level_pipeline(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": method,
                                   "canica": True})

# Group
## One-sample t-test
group_level_pipeline(bids_dir,
                     output_dir,
                     config={"method": method,
                             "atlas": "aal",
                             "analysis_name": "testMeanEffect",
                             "contrast": "intercept",
                             "fdr_alpha": 0.5,
                             "fwe_alpha":  0.5,
                             "n_permutations": 10,
                             "thresholding_strategies": ["uncorrected",
                                                         "fdr",
                                                         "fwe"]})

plt.close('all')

## One-sample t-test, canica
group_level_pipeline(bids_dir,
                     output_dir,
                     config={"method": method,
                             "canica": True,
                             "analysis_name": "testMeanEffect",
                             "contrast": "intercept",
                             "fdr_alpha": 0.5,
                             "fwe_alpha":  0.5,
                             "n_permutations": 10,
                             "thresholding_strategies": ["uncorrected",
                                                         "fdr",
                                                         "fwe"]})

plt.close('all')