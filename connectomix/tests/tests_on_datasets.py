from connectomix.tools.tools import *

import os

# ds = "ds005699"

# ----- Test: single subject -----
ds = "ds005625"
bids_dir = os.path.join('/rawdata', ds)
derivatives_dir = os.path.join('/derivatives', ds)
derivatives = {}
derivatives['fmriprep'] = os.path.join(derivatives_dir, 'fmriprep')
output_dir = os.path.join(derivatives_dir, 'connectomix-roiToVoxel-dev')

## roiToVoxel, seed-based (default)

config = {}
config['seeds_file'] = '/code/ds005625/connectomix/seeds/example_seeds.tsv'
participant_level_analysis(bids_dir, output_dir, derivatives, config)

## roiToVoxel, roi-based (from mask)

config = {}
config['roi_masks'] = {"cerebInferiorPosteriorLobe": "/home/arovai/mycloud/MNI_inferior_posterior_lobe.nii.gz",
                       "cerebAnteriorLobe": "/home/arovai/mycloud/MNI_anterior_lobe.nii.gz"}
participant_level_analysis(bids_dir, output_dir, derivatives, config)

## roiToRoi, seed-based

config = {}
config["method"] = "seeds"
config['seeds_file'] = '/code/ds005625/connectomix/seeds/brain_and_cerebellum_seeds.tsv'
participant_level_analysis(bids_dir, output_dir, derivatives, config)

## roiToRoi, atlas-based - aal

config = {}
config["method"] = "aal"
participant_level_analysis(bids_dir, output_dir, derivatives, config)

## roiToRoi, atlas-based - schaeffer100

config = {}
config["method"] = "schaeffer100"
participant_level_analysis(bids_dir, output_dir, derivatives, config)

## roiToRoi, atlas-based - harvardoxford

config = {}
config["method"] = "harvardoxford"
participant_level_analysis(bids_dir, output_dir, derivatives, config)

# ------ Set-up for ds005699 - 4 subjects, two groups, one task (no session or run)
ds = "ds005699"
bids_dir = os.path.join('/rawdata', ds)
derivatives_dir = os.path.join('/derivatives', ds)
derivatives = {}
derivatives['fmriprep'] = os.path.join(derivatives_dir, 'fmriprep')
output_dir = os.path.join(derivatives_dir, 'connectomix-roiToVoxel-dev')

## roiToVoxel analysis, seed-based

# Participant analysis
config = {}
config['seeds_file'] = '/code/ds005699/connectomix/seeds/example_seeds.tsv'
participant_level_analysis(bids_dir, output_dir, derivatives, config)

# Group analysis - mean effect
config = {}
config['seeds_file'] = '/code/ds005699/connectomix/seeds/example_seeds.tsv'
config["analysis_label"] = "meanEffect"
group_level_analysis(bids_dir, output_dir, config)

# Group analysis - covariate (effect of age)
config = {}
config['seeds_file'] = '/code/ds005699/connectomix/seeds/example_seeds.tsv'
config["analysis_label"] = "effetOfAge"
config["group_confounds"] = "age"
config["group_contrast"] = "age"
group_level_analysis(bids_dir, output_dir, config)

# Group analysis, independent samples comparison (controls versus patients)
config = {}
config['seeds_file'] = '/code/ds005699/connectomix/seeds/example_seeds.tsv'
config["analysis_label"] = "controlVsPatient"
config["group_confounds"] = "group"
config["group_contrast"] = "control-patient"
group_level_analysis(bids_dir, output_dir, config)

## roiToVoxel, roi-based (from mask)

# Participant analysis
config = {}
config['roi_masks'] = {"cerebInferiorPosteriorLobe": "/home/arovai/mycloud/MNI_inferior_posterior_lobe.nii.gz",
                       "cerebAnteriorLobe": "/home/arovai/mycloud/MNI_anterior_lobe.nii.gz"}
participant_level_analysis(bids_dir, output_dir, derivatives, config)

# Group analysis - mean effect
config = {}
config["analysis_label"] = "meanEffect"
config['roi_masks'] = {"cerebInferiorPosteriorLobe": "/home/arovai/mycloud/MNI_inferior_posterior_lobe.nii.gz",
                       "cerebAnteriorLobe": "/home/arovai/mycloud/MNI_anterior_lobe.nii.gz"}
group_level_analysis(bids_dir, output_dir, config)

# Group analysis - covariate (effect of age)
config = {}
config['roi_masks'] = {"cerebInferiorPosteriorLobe": "/home/arovai/mycloud/MNI_inferior_posterior_lobe.nii.gz",
                       "cerebAnteriorLobe": "/home/arovai/mycloud/MNI_anterior_lobe.nii.gz"}
config["analysis_label"] = "effetOfAge"
config["group_confounds"] = "age"
config["group_contrast"] = "age"
group_level_analysis(bids_dir, output_dir, config)

# Group analysis, independent samples comparison (controls versus patients)
config = {}
config['roi_masks'] = {"cerebInferiorPosteriorLobe": "/home/arovai/mycloud/MNI_inferior_posterior_lobe.nii.gz",
                       "cerebAnteriorLobe": "/home/arovai/mycloud/MNI_anterior_lobe.nii.gz"}
config["analysis_label"] = "controlVsPatient"
config["group_confounds"] = "group"
config["group_contrast"] = "control-patient"
group_level_analysis(bids_dir, output_dir, config)

## roiToRoi, seed-based

# Participant analysis
config = {}
config["method"] = "seeds"
config['seeds_file'] = '/code/ds005699/connectomix/seeds/brain_and_cerebellum_seeds.tsv'
participant_level_analysis(bids_dir, output_dir, derivatives, config)

# Group analysis - controls versus patients
config = {}
config["method"] = "seeds"
config['seeds_file'] = '/code/ds005699/connectomix/seeds/brain_and_cerebellum_seeds.tsv'
config["analysis_type"] = "independent"
group_level_analysis(bids_dir, output_dir, config)

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
