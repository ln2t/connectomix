# ------ Real-world example - Hilarious Mosquito dataset

bids_dir = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/rawdata"
derivatives_dir = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/derivatives"
derivatives = {}
derivatives['fmriprep'] = os.path.join(derivatives_dir, 'fmriprep_v23.1.3')
output_dir = os.path.join(derivatives_dir, 'connectomix-roiToVoxel-dev')

## roiToVoxel analysis, seed-based

# Participant analysis
config = {}
config['seeds_file'] = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix/data/example_seeds.tsv"
participant_level_analysis(bids_dir, output_dir, derivatives, config)

# Group analysis - mean effect
config = {}
config['seeds_file'] = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix/data/example_seeds.tsv"
config["analysis_label"] = "meanEffect"
group_level_analysis(bids_dir, output_dir, config)

# Group analysis - covariate (effect of age)
# config = {}
# config['seeds_file'] = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix/data/example_seeds.tsv"
# config["analysis_label"] = "effetOfAge"
# config["group_confounds"] = "age"
# config["group_contrast"] = "age"
# group_level_analysis(bids_dir, output_dir, config)

# Group analysis, independent samples comparison (controls versus patients)
config = {}
config['seeds_file'] = "/data/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix/data/example_seeds.tsv"
config["group_confounds"] = "group"
config["group_contrast"] = "patient-control"
config["analysis_label"] = "PatientVsControls"
config["sessions"] = ["1"]
# config["n_permutations"] = 5000
group_level_analysis(bids_dir, output_dir, config)
