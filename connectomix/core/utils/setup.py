import warnings

from bids import BIDSLayout


def setup_config(layout, config, level):
    """
    Set the configuration fields to their default values if not explicitly specified in the input config.

    Parameters
    ----------
    config : dict
        Input configuration. Can be completely empty (config = {}).
    layout : BIDSLayout
        BIDS layout object of the dataset.
    level : str
        "participant" or "group"

    Returns
    -------
    config : dict
        A complete configuration.

    """
    from connectomix.core.utils.loaders import load_config
    config = load_config(config)
    config = setup_config_bids(config, layout, level)
    config = setup_config_analysis(config, level)

    if level == "participant":
        config = setup_config_preprocessing(config)
    elif level == "group":
        config = setup_config_stats(config)

    return config


def setup_config_bids(config, layout, level):
    from connectomix.core.utils.tools import config_helper

    derivatives_to_parse = ""
    if level == "participant":
        derivatives_to_parse = "fMRIPrep"
    elif level == "group":
        derivatives_to_parse = "connectomix"

    derivatives_layout = layout.derivatives[derivatives_to_parse]

    config["subject"] = config_helper(config,
                                      "subject",
                                      derivatives_layout.get_subjects())
    config["tasks"] = config_helper(config,
                                    "tasks",
                                    derivatives_layout.get_tasks())
    config["runs"] = config_helper(config,
                                   "runs",
                                   derivatives_layout.get_runs())
    config["sessions"] = config_helper(config,
                                       "sessions",
                                       derivatives_layout.get_sessions())
    config["spaces"] = config_helper(config,
                                     "spaces",
                                     derivatives_layout.get_spaces())

    if 'MNI152NLin2009cAsym' in config["spaces"]:
        # First default to 'MNI152NLin2009cAsym'
        config["spaces"] = ['MNI152NLin2009cAsym']
    elif 'MNI152NLin6Asym' in config["spaces"]:
        # Second default to 'MNI152NLin6Asym' (useful when using ica-aroma denoising)
        config["spaces"] = ['MNI152NLin6Asym']

    return config


def setup_config_preprocessing(config):
    from connectomix.core.utils.tools import config_helper

    # Reference functional file for resampling
    config["reference_functional_file"] = config_helper(config,
                                                        "reference_functional_file",
                                                        "first_functional_file")

    # High-pass filter for data denoising - Default value from Ciric et al 2017
    config["high_pass"] = config_helper(config,
                                        "high_pass",
                                        0.01)

    # Low-pass filter for data denoising - Default value from Ciric et al 2017
    config["low_pass"] = config_helper(config,
                                       "low_pass",
                                       0.08)

    # List of default signal confounds for denoising
    default_confounds = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf_wm']

    # Confounds for denoising
    config["confounds"] = config_helper(config,
                                        "confound_columns",
                                        default_confounds)

    # ICA-AROMA denoising
    config["ica_aroma"] = config_helper(config,
                                        "ica_aroma",
                                        False,
                                        [True, False])

    # For ICA-AROMA, default to space 'MNI152NLin6Asym'
    if config["ica_aroma"]:
        print("Defaulting to space MNI152NLin6Asym for ICA-AROMA denoising (overriding spaces from config file")
        config["spaces"] = ['MNI152NLin6Asym']
    elif "MNI152NLin6Asym" in config["spaces"]:
        warnings.warn(
            "Space 'MNI152NLin6Asym' was found in the list of spaces and ica_aroma was disabled. To avoid name conflicts, we force you to use ica_aroma with MNI152NLin6Asym. For now, 'MNI152NLin6Asym' will be removed from the list of spaces.")
        config["spaces"] = [space for space in config["spaces"] if space != 'MNI152NLin6Asym']
    return config


def setup_config_analysis(config, level):
    from connectomix.core.utils.tools import config_helper

    config["method"] = config_helper(config,
                                     "method",
                                     "seedToVoxel",
                                     ["seedToVoxel", "roiToVoxel", "roiToRoi", "seedToSeed"])

    allowed_connectivity_kinds = ["correlation", "covariance", "partial correlation", "precision"]
    allowed_atlases = ["schaeffer100", "aal", "harvardoxford", "canica"]
    match config["method"]:
        case "seedToVoxel":
            config["seeds_file"] = config_helper(config,
                                                 "seeds_file",
                                                 None)
            config["radius"] = config_helper(config,
                                             "radius",
                                             5)
        case "roiToVoxel":
            config["roi_masks"] = config_helper(config,
                                                "roi_masks",
                                                None)
        case "seedToSeed":
            config["seeds_file"] = config_helper(config,
                                                 "seeds_file",
                                                 None)
            config["radius"] = config_helper(config,
                                             "radius",
                                             5)
            config["connectivity_kind"] = config_helper(config,
                                                         "connectivity_kind",
                                                         "correlation",
                                                         allowed_connectivity_kinds)
            config["custom_seeds_name"] = config_helper(config,
                                                        "custom_seeds_name",
                                                        None)
        case "roiToRoi":
            config["atlas"] = config_helper(config,
                                            "atlas",
                                            "schaeffer100",
                                            allowed_atlases)
            config["connectivity_kind"] = config_helper(config,
                                                         "connectivity_kind",
                                                         "correlation",
                                                         allowed_connectivity_kinds)
            if config["atlas"] == "canica" or config.get("canica", False):
                # see nilearn for the meaning of these options
                config["canica_threshold"] = config_helper(config,
                                                           'canica_threshold',
                                                           0.5)
                config["canica_min_region_size"] = config_helper(config,
                                                                 'canica_min_region_size',
                                                                 50)
                config["canica"] = True
                config["atlas"] = "canica"
    if level == "group":
        config["smoothing"] = config_helper(config,
                                            "smoothing",
                                            8)  # In mm, smoothing for the cmaps

        config["analysis_name"] = config_helper(config,
                                                "analysis_name",
                                                "customName")

        config["paired_tests"] = config_helper(config,
                                               "paired_tests",
                                               False,
                                               [True, False])

        config["covariates"] = config_helper(config,
                                             "covariates",
                                             [])  # TODO: write a function that fetches the col names of participants.tsv

        config["add_intercept"] = config_helper(config,
                                            "add_intercept",
                                            True,
                                            [True, False])

        config["contrast"] = config_helper(config,
                                          "contrast",
                                          "intercept")

        # Roi-to-roi specific parameters
        #
        # config["group1_subjects"] = config_helper(config,
        #                                           "group1_subjects",
        #                                           None)
        # config["group2_subjects"] = config_helper(config,
        #                                           "group2_subjects",
        #                                           None)
        #
        # if config["analysis_type"] == 'independent' and config["group1_subjects"] is None:
        #     from connectomix.core.tools import guess_groups
        #     guessed_groups = guess_groups(layout)
        #     if len(guessed_groups) == 2:
        #         group1_name = list(guessed_groups.keys())[0]
        #         group2_name = list(guessed_groups.keys())[1]
        #         warnings.warn(
        #             f"Group have been guessed. Assuming group 1 is {group1_name} and group 2 is {group2_name}")
        #         config["group1_subjects"] = list(guessed_groups.values())[0]
        #         config["group2_subjects"] = list(guessed_groups.values())[1]
        #         config[
        #             "analysis_label"] = f"{group1_name}VersuS{group2_name}"  # This overwrites the above generic name to ensure people don't get confused with the automatic selection of subjects
        #         warnings.warn(f"Setting analysis label to {config['analysis_label']}")
        #         config["group1_name"] = group1_name
        #         config["group2_name"] = group2_name
        #     else:
        #         config["group1_subjects"] = config.get("subjects", layout.derivatives[
        #             "connectomix"].get_subjects())  # this is used only through the --helper tool (or the autonomous mode)
        #         warnings.warn("Could not detect two groups, putting all subjects into first group.")

    return config


def setup_config_stats(config):
    from connectomix.core.utils.tools import config_helper

    # Stats and permutations
    config["uncorrected_alpha"] = config_helper(config,
                                                "uncorrected_alpha",
                                                0.001)

    config["fdr_alpha"] = config_helper(config,
                                        "fdr_alpha",
                                        0.05)

    config["fwe_alpha"] = float(config_helper(config,
                                              "fwe_alpha",
                                              0.05))

    config["n_permutations"] = config_helper(config,
                                             "n_permutations",
                                             20)

    config["two_sided_test"] = config_helper(config,
                                             "two_sided_test",
                                             True,
                                             [True, False])

    config["n_jobs"] = config_helper(config,
                                     "n_jobs",
                                     2)

    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        # p-value for cluster forming threshold
        config["cluster_forming_alpha"] = config_helper(config,
                                                        "cluster_forming_alpha",
                                                        0.01)

    return config

# def create_participant_level_default_config_file(bids_dir, output_dir, fmriprep_dir):
#     """
#     Create default configuration file in YAML format for default parameters, at participant level.
#     The configuration file is saved at 'derivatives_dir/config/default_participant_level_config.yaml'
#
#     Parameters
#     ----------
#     bids_dir : str or Path
#         Path to BIDS directory.
#     output_dir : str or Path
#         Path to derivatives.
#     fmriprep_dir : str or Path
#         Path to fMRIPrep derivatives.
#
#     Returns
#     -------
#     None.
#
#     """
#
#     # Print some stuff for the primate using this function
#     print("Generating default configuration file for default parameters, please wait while the dataset is explored...")
#
#     # Create derivative directory
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     # Create the dataset_description.json file
#     from connectomix.core.makers import create_dataset_description
#     create_dataset_description(output_dir)
#
#     # Create a BIDSLayout to parse the BIDS dataset
#     layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir, output_dir])
#
#     # Load all default values in config file
#     config = setup_config({}, layout, "participant")
#
#     # Prepare the YAML content with comments
#     yaml_content_with_comments = f"""\
# # Connectomix Configuration File
# # This file is generated automatically. Please modify the parameters as needed.
# # Full documentation is located at github.com/ln2t/connectomix
# # Important note: more parameters can be tuned than those shown here, this is only a starting point.
#
# # List of subjects
# subjects: {config.get("subjects")}
#
# # List of tasks
# tasks: {config.get("tasks")}
#
# # List of runs
# runs: {config.get("runs")}
#
# # List of sessions
# sessions: {config.get("sessions")}
#
# # List of output spaces
# spaces: {config.get("spaces")}
#
# # Confounding variables to include when extracting timeseries. Choose from confounds computed from fMRIPrep.
# confound_columns: {config.get("confound_columns")}
#
# # Use ICA-AROMA denoised data or not. If set to True, fMRIPrep output must be further processed using fMRIPost-AROMA (see https://github.com/nipreps/fmripost-aroma)
# ica_aroma: {config.get("ica_aroma")}
#
# # Kind of connectivity measure to compute
# connectivity_kind: {config.get("connectivity_kinds")}  # Choose from covariance, correlation, partial correlation or precision. This option is passed to nilearn.connectome.ConnectivityMeasure.
#
# # Method to define regions of interests to compute connectivity
# method: {config.get("method")} # Method to determine ROIs to compute variance. Uses the Schaeffer 2018 with 100 rois by default. More options are described in the documentation.
#
# # Other parameters
# high_pass: {config.get("high_pass")} # High-pass filtering, in Hz, applied to BOLD data. Low (<0.008 Hz) values does minimal changes to the signal, while high (>0.01) values improves sensitivity.
# low_pass: {config.get("low_pass")} # Low-pass filtering, in Hz, applied to BOLD data. High (>0.1 Hz) values does minimal changes to the signal, while low (< 0.08 Hz)values improves specificity.
# seeds_file: {config["seeds_file"]} # Path to seed file for seed-based ROIs
# radius: {config["radius"]} # Radius, in mm, to create the spheres at the coordinates of the seeds for seed-based ROIs
#     """
#
#     # Build filenames for each output
#     yaml_file = Path(output_dir) / 'config' / 'default_participant_level_config.yaml'
#
#     from connectomix.core.makers import ensure_directory
#     ensure_directory(yaml_file)
#
#     # Save the YAML content with comments
#     with open(yaml_file, 'w') as yaml_out:
#         yaml_out.write(yaml_content_with_comments)
#
#     print(f"Default YAML configuration file saved at {yaml_file}. Go to github.com/ln2t/connectomix for more details.")
#     print("See also below for the output:")
#     print(yaml_content_with_comments)

