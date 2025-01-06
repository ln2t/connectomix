import matplotlib.pyplot as plt
import importlib.resources

from connectomix.utils.modes import participant_level_analysis, group_level_analysis
from connectomix.tests.paths import bids_dir, fmriprep_dir, output_dir

example_seeds_for_seedToSeed = str(importlib.resources.files("connectomix.tests.seeds").joinpath("example_seeds_for_seedToSeed.tsv"))
example_seeds_for_seedToVoxel = str(importlib.resources.files("connectomix.tests.seeds").joinpath("example_seeds_for_seedToVoxel.tsv"))

precuneus_L_mask = str(importlib.resources.files("connectomix.tests.seeds").joinpath("AAL_Precuneus_L.nii.gz"))
precentral_R_mask = str(importlib.resources.files("connectomix.tests.seeds").joinpath("AAL_Precentral_R.nii.gz"))

## seedToVoxel
participant_level_analysis(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": "seedToVoxel",
                               "seeds_file": example_seeds_for_seedToVoxel})

## roiToVoxel

participant_level_analysis(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": "roiToVoxel",
                               "roi_masks": {"precentralR": precentral_R_mask,
                                                 "precuneusL": precuneus_L_mask}})

## seedToSeed
participant_level_analysis(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": "seedToSeed",
                                   "seeds_file": example_seeds_for_seedToSeed})
plt.close('all')

## roiToRoi, atlas-based - aal

participant_level_analysis(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": "aal"})
plt.close('all')

## roiToRoi, atlas-based - schaeffer100

participant_level_analysis(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": "schaeffer100"})
plt.close('all')

## roiToRoi, atlas-based - harvardoxford

participant_level_analysis(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": "harvardoxford"})
plt.close('all')

# Group analysis - mean effect
group_level_analysis(bids_dir,
                     output_dir,
                     config={"analysis_label": "meanEffect",
                             "group1_subjects": ["001", "004", "037", "042"],
                             "roi_masks": {"precentralR": precentral_R_mask,
                                           "precuneusL": precuneus_L_mask}})

# Group analysis - covariate (effect of age)
group_level_analysis(bids_dir,
                     output_dir,
                     config={"analysis_label": "effetOfAge",
                             "group1_subjects": ["001", "004", "037", "042"],
                             "group_confounds": "age",
                             "group_contrast": "age",
                             "roi_masks": {"precentralR": precentral_R_mask,
                                           "precuneusL": precuneus_L_mask}})

# Group analysis, independent samples comparison (controls versus patients)
group_level_analysis(bids_dir,
                     output_dir,
                     config={"analysis_label": "controlVsPatient",
                             "group_confounds": "group",
                             "group_contrast": "control-patient",
                             "roi_masks": {"precentralR": precentral_R_mask,
                                           "precuneusL": precuneus_L_mask}})

## roiToRoi, seed-based

# Participant analysis
participant_level_analysis(bids_dir,
                           output_dir,
                           derivatives={"fmriprep": fmriprep_dir},
                           config={"method": "seeds",
                                   "seeds_file": example_seeds_for_roiToRoi})
plt.close('all')

# Group analysis - controls versus patients
group_level_analysis(bids_dir,
                     output_dir,
                     config={"method": "seeds",
                             "seeds_file": example_seeds_for_roiToRoi})

# ------ Real-world example - Hilarious Mosquito dataset

bids_dir = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/rawdata"
derivatives_dir = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/derivatives"
derivatives = {}
derivatives['fmriprep'] = os.path.join(derivatives_dir, 'fmriprep_v23.1.3')
output_dir = os.path.join(derivatives_dir, 'connectomix-roiToVoxel-dev')

## roiToVoxel analysis, seed-based

# Participant analysis
config = {}
config['seeds_file'] = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix/seeds/example_seeds.tsv"
participant_level_analysis(bids_dir, output_dir, derivatives, config)

# Group analysis - mean effect
config = {}
config['seeds_file'] = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix/seeds/example_seeds.tsv"
config["analysis_label"] = "meanEffect"
group_level_analysis(bids_dir, output_dir, config)

# Group analysis - covariate (effect of age)
# config = {}
# config['seeds_file'] = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix/seeds/example_seeds.tsv"
# config["analysis_label"] = "effetOfAge"
# config["group_confounds"] = "age"
# config["group_contrast"] = "age"
# group_level_analysis(bids_dir, output_dir, config)

# Group analysis, independent samples comparison (controls versus patients)
config = {}
config['seeds_file'] = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix/seeds/example_seeds.tsv"
config["group_confounds"] = "group"
config["group_contrast"] = "patient-control"
config["analysis_label"] = "PatientVsControls"
config["sessions"] = ["1"]
# config["n_permutations"] = 5000
group_level_analysis(bids_dir, output_dir, config)
