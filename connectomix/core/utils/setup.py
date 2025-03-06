import warnings

from bids import BIDSLayout


def setup_config(layout, config, level, cli_options=None):
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
    config = setup_config_bids(config, layout, level, cli_options)
    config = setup_config_analysis(config, level)

    if level == "participant":
        config = setup_config_preprocessing(config)
    elif level == "group":
        config = setup_config_stats(config)

    return config


def setup_config_bids(config, layout, level, cli_options=None):
    from connectomix.core.utils.tools import config_helper

    derivatives_to_parse = ""
    if level == "participant":
        derivatives_to_parse = "fMRIPrep"
    elif level == "group":
        derivatives_to_parse = "connectomix"

    derivatives_layout = layout.derivatives.get_pipeline(derivatives_to_parse)

    if cli_options and cli_options.get("participant_label", None) is not None:
        config["subject"] = cli_options["participant_label"]

    if cli_options and cli_options.get("session", None) is not None:
        config["sessions"] = cli_options["session"]

    if cli_options and cli_options.get("task", None) is not None:
        config["tasks"] = cli_options["task"]

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
                                        "confounds",
                                        default_confounds)

    config["overwrite_denoised_files"] = config_helper(config,
                                                       "overwrite_denoised_files",
                                                       True,
                                                       [True, False])

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
                                     "roiToRoi",
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
    return config


def setup_config_stats(config):
    from connectomix.core.utils.tools import config_helper

    config["two_sided_test"] = config_helper(config,
                                             "two_sided_test",
                                             True,
                                             [True, False])

    config["thresholding_strategies"] = config_helper(config,
                                                      "thresholding_strategies",
                                                      ["uncorrected", "fdr", "fwe"])

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

    config["n_jobs"] = config_helper(config,
                                     "n_jobs",
                                     1)

    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        config["cluster_forming_alpha"] = str(config_helper(config,
                                                        "cluster_forming_alpha",
                                                        0.01))
    elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
        config["cluster_forming_alpha"] = None

    return config