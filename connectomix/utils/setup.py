import warnings
from pathlib import Path
from bids import BIDSLayout

from connectomix.utils.tools import guess_groups, create_dataset_description, ensure_directory

# CONFIG
# Function to manage default group-level options
def set_unspecified_participant_level_options_to_default(config, layout):
    """
    Set the configuration fields to their default values if not explicitly specified in the input config.

    Parameters
    ----------
    config : dict
        Input configuration. Can be completely empty (config = {}).
    layout : BIDSLayout
        BIDS layout object of the dataset.

    Returns
    -------
    config : dict
        A complete configuration.

    """
    # BIDS stuff
    config["subject"] = config.get("subject",
                                   layout.derivatives['fMRIPrep'].get_subjects())  # Subjects to include in the analysis
    config["tasks"] = config.get("tasks",
                                 layout.derivatives['fMRIPrep'].get_tasks())  # Tasks to include in the analysis
    config["runs"] = config.get("runs", layout.derivatives['fMRIPrep'].get_runs())  # Runs to include in the analysis
    config["sessions"] = config.get("sessions", layout.derivatives[
        'fMRIPrep'].get_sessions())  # Sessions to include in the analysis
    config["spaces"] = config.get("spaces",
                                  layout.derivatives['fMRIPrep'].get_spaces())  # Spaces to include in the analysis

    if 'MNI152NLin2009cAsym' in config.get("spaces"):
        config["spaces"] = ['MNI152NLin2009cAsym']  # First default to 'MNI152NLin2009cAsym'
    elif 'MNI152NLin6Asym' in config.get("spaces"):
        config["spaces"] = [
            'MNI152NLin6Asym']  # Second default to 'MNI152NLin6Asym' (useful when using ica-aroma denoising)

    # Analysis parameters
    config["method"] = config.get("method",
                                  "roiToVoxel")  # The method to define connectome, e.g. from a valid atlas name or "roiToVoxel"
    config["seeds_file"] = config.get("seeds_file",
                                      None)  # Path to file with seed coordinates for seed-based and roi-to-voxel
    config["radius"] = config.get("radius", 5)  # Radius of the sphere, in mm, for the seeds
    config["supported_atlases"] = ["schaeffer100", "aal",
                                   "harvardoxford"]  # This is not a user parameters but is used only internally

    # Preprocessing parameters
    config["reference_functional_file"] = config.get("reference_functional_file",
                                                     "first_functional_file")  # Reference functional file for resampling
    config["high_pass"] = config.get("high_pass",
                                     0.01)  # High-pass filter for data denoising - Default value from Ciric et al 2017
    config["low_pass"] = config.get("low_pass",
                                    0.08)  # Low-pass filter for data denoising - Default value from Ciric et al 2017
    default_confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                'csf_wm']  # List of default signal confounds for denoising
    config["confound_columns"] = config.get("confound_columns",
                                            default_confound_columns)  # Signal confounds for denoising
    config["ica_aroma"] = config.get("ica_aroma", False)  # ICA-AROMA denoising

    if config["ica_aroma"]:  # For ICA-AROMA, default to space 'MNI152NLin6Asym'
        print("Defaulting to space MNI152NLin6Asym for ICA-AROMA denoising (overriding spaces from config file")
        config["spaces"] = ['MNI152NLin6Asym']
    elif "MNI152NLin6Asym" in config["spaces"]:
        warnings.warn(
            "Space 'MNI152NLin6Asym' was found in the list of spaces and ica_aroma was disabled. To avoid name conflicts, we force you to use ica_aroma with MNI152NLin6Asym. For now, 'MNI152NLin6Asym' will be removed from the list of spaces.")
        config["spaces"] = [space for space in config["spaces"] if space != 'MNI152NLin6Asym']

    # Roi-to-roi specific parameters
    config["connectivity_kinds"] = config.get("connectivity_kinds", [
        "correlation"])  # The kind of connectivity measure, unused for roi-to-voxel
    # .. canica options
    config["canica_threshold"] = config.get('canica_threshold', 0.5)
    config["canica_min_region_size"] = config.get('canica_min_region_size',
                                                  50)  # Extract also regions from the canica components for connectivity analysis

    # Roi-to-voxel specific parameters
    config["roi_masks"] = config.get("roi_masks", None)  # List of path to mask for roi-to-voxel

    # Consistency checks
    # TODO: add more consistency checks!
    if config["method"] == "roiToVoxel" and (config["seeds_file"] is not None and config["roi_masks"] is not None):
        raise ValueError(
            "Config fields 'seeds_file' and 'roi_masks' cannot both be defined when performing 'roiToVoxel' analyzes")

    # List-ify connectivity_kinds in case it was not set to a list in config by user
    if not isinstance(config["connectivity_kinds"], list):
        config["connectivity_kinds"] = [config["connectivity_kinds"]]

    return config


# Function to manage default group-level options
def set_unspecified_group_level_options_to_default(config, layout):
    """
    Set the configuration fields to their default values if not explicitly specified in the input config.

    Parameters
    ----------
    config : dict
        Input configuration. Can be completely empty (config = {}).
    layout : BIDSLayout
        BIDS layout object of the dataset.

    Returns
    -------
    config : dict
        A complete configuration.

    """

    # BIDS stuff
    config["subject"] = config.get("subject", layout.derivatives[
        "connectomix"].get_subjects())  # Subjects to include in the analysis
    config["tasks"] = config.get("tasks",
                                 "restingstate" if "restingstate" in layout.derivatives["connectomix"].get_tasks() else
                                 layout.derivatives["connectomix"].get_tasks())
    config["runs"] = config.get("runs", layout.derivatives["connectomix"].get_runs())
    config["sessions"] = config.get("sessions", layout.derivatives["connectomix"].get_sessions())
    config["spaces"] = config.get("spaces", "MNI152NLin2009cAsym" if "MNI152NLin2009cAsym" in layout.derivatives[
        "connectomix"].get_spaces() else layout.derivatives["connectomix"].get_spaces())

    # Participant-level parameters
    config["method"] = config.get("method", "roiToVoxel")
    config["seeds_file"] = config.get("seeds_file",
                                      None)  # Path to file with seed coordinates for seed-based and roi-to-voxel
    config["radius"] = config.get("radius", 5)
    config["roi_masks"] = config.get("roi_masks", None)  # List of path to mask for roi-to-voxel

    # Analysis parameters
    config["analysis_type"] = config.get("analysis_type",
                                         "independent")  # Options: 'independent' or 'paired' or 'regression'
    config["analysis_label"] = config.get("analysis_label", "CUSTOMNAME")
    config["smoothing"] = config.get("smoothing", 8)  # In mm, smoothing for the cmaps
    config["group_confounds"] = config.get("group_confounds", [])
    config["group_contrast"] = config.get("group_contrast", "intercept")

    # Stats and permutations
    config["uncorrected_alpha"] = config.get("uncorrected_alpha", 0.001)
    config["fdr_alpha"] = config.get("fdr_alpha", 0.05)
    config["fwe_alpha"] = float(config.get("fwe_alpha", 0.05))
    config["n_permutations"] = config.get("n_permutations", 20)

    # Toi-to-voxel parameters
    config["cluster_forming_alpha"] = config.get("cluster_forming_alpha",
                                                 0.01)  # p-value for cluster forming threshold in roiToVoxel analysiss

    # Roi-to-roi specific parameters
    config["connectivity_kinds"] = config.get("connectivity_kinds", ["correlation"])
    config["group1_subjects"] = config.get("group1_subjects", None)
    config["group2_subjects"] = config.get("group2_subjects", None)

    if config["analysis_type"] == 'independent' and config["group1_subjects"] is None:
        guessed_groups = guess_groups(layout)
        if len(guessed_groups) == 2:
            group1_name = list(guessed_groups.keys())[0]
            group2_name = list(guessed_groups.keys())[1]
            warnings.warn(f"Group have been guessed. Assuming group 1 is {group1_name} and group 2 is {group2_name}")
            config["group1_subjects"] = list(guessed_groups.values())[0]
            config["group2_subjects"] = list(guessed_groups.values())[1]
            config[
                "analysis_label"] = f"{group1_name}VersuS{group2_name}"  # This overwrites the above generic name to ensure people don't get confused with the automatic selection of subjects
            warnings.warn(f"Setting analysis label to {config['analysis_label']}")
            config["group1_name"] = group1_name
            config["group2_name"] = group2_name
        else:
            config["group1_subjects"] = config.get("subjects", layout.derivatives[
                "connectomix"].get_subjects())  # this is used only through the --helper tool (or the autonomous mode)
            warnings.warn("Could not detect two groups, putting all subjects into first group.")

    # List-ify connectivity_kinds in case it was not set to a list in config by user
    if not isinstance(config["connectivity_kinds"], list):
        config["connectivity_kinds"] = [config["connectivity_kinds"]]

    # analysis_options = {}

    # # If "analysis_options" is not set by user, then try to create it
    # if "analysis_options" not in config.keys():
    #     if config["analysis_type"] == 'regression':
    #         analysis_options["subjects_to_regress"] = layout.derivatives["connectomix"].get_subjects()
    #         analysis_options["covariate"] = "COVARIATENAME"
    #         analysis_options["confounds"] = []
    #     else:

    #         if config["analysis_type"] == 'independent':

    #             guessed_groups = guess_groups(layout)
    #             if len(guessed_groups) == 2:
    #                 group1_name = list(guessed_groups.keys())[0]
    #                 group2_name = list(guessed_groups.keys())[1]
    #                 warnings.warn(f"Group have been guessed. Assuming group 1 is {group1_name} and group 2 is {group2_name}")
    #                 analysis_options["group1_subjects"] = list(guessed_groups.values())[0]
    #                 analysis_options["group2_subjects"] = list(guessed_groups.values())[1]
    #                 config["analysis_label"] = f"{group1_name}VersuS{group2_name}"  # This overwrites the above generic name to ensure people don't get confused with the automatic selection of subjects
    #                 warnings.warn(f"Setting analysis label to {config['analysis_label']}")
    #                 analysis_options["group1_name"] = group1_name
    #                 analysis_options["group2_name"] = group2_name
    #             else:
    #                 config["group1_subjects"] = config.get("subjects", layout.derivatives["connectomix"].get_subjects())  # this is used only through the --helper tool (or the autonomous mode)
    #                 warnings.warn("Could not detect two groups, putting all subjects into first group.")

    #         elif config["analysis_type"] == 'paired':
    #             analysis_options["subjects"] = layout.derivatives["connectomix"].get_subjects()
    #             analysis_options["sample1_entities"] = dict(tasks=config.get("tasks"),
    #                                                         sessions=config.get("sessions"),
    #                                                         runs=config.get("runs"))
    #             analysis_options["sample2_entities"] = analysis_options["sample1_entities"]  # This does not make sense, but is done only to help the user to fine-tune the config file manually.

    # TODO: add more checks that config has all required information to work
    # Set the analysis_options field
    # Todo: enable confounds to be optional in the config file. Currently it does not work if analysis_options are set in config file as it does not take the default value in the following line. But we want to be able to leave the confounds field empty in the config file.
    # config = config.get("analysis_options", analysis_options)

    # if config.get("method") == 'seeds' or config.get("method") == 'roiToVoxel':
    #     config["radius"] = config.get("radius", 5)

    return config


def create_participant_level_default_config_file(bids_dir, output_dir, fmriprep_dir):
    """
    Create default configuration file in YAML format for default parameters, at participant level.
    The configuration file is saved at 'derivatives_dir/config/default_participant_level_config.yaml'

    Parameters
    ----------
    bids_dir : str or Path
        Path to BIDS directory.
    output_dir : str or Path
        Path to derivatives.
    fmriprep_dir : str or Path
        Path to fMRIPrep derivatives.

    Returns
    -------
    None.

    """

    # Print some stuff for the primate using this function
    print("Generating default configuration file for default parameters, please wait while the dataset is explored...")

    # Create derivative directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the dataset_description.json file
    create_dataset_description(output_dir)

    # Create a BIDSLayout to parse the BIDS dataset
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir, output_dir])

    # Load all default values in config file
    config = set_unspecified_participant_level_options_to_default({}, layout)

    # Prepare the YAML content with comments
    yaml_content_with_comments = f"""\
# Connectomix Configuration File
# This file is generated automatically. Please modify the parameters as needed.
# Full documentation is located at github.com/ln2t/connectomix
# Important note: more parameters can be tuned than those shown here, this is only a starting point.

# List of subjects
subjects: {config.get("subjects")}

# List of tasks
tasks: {config.get("tasks")}

# List of runs
runs: {config.get("runs")}

# List of sessions
sessions: {config.get("sessions")}

# List of output spaces
spaces: {config.get("spaces")}

# Confounding variables to include when extracting timeseries. Choose from confounds computed from fMRIPrep.
confound_columns: {config.get("confound_columns")}

# Use ICA-AROMA denoised data or not. If set to True, fMRIPrep output must be further processed using fMRIPost-AROMA (see https://github.com/nipreps/fmripost-aroma)
ica_aroma: {config.get("ica_aroma")}

# Kind of connectivity measure to compute
connectivity_kind: {config.get("connectivity_kinds")}  # Choose from covariance, correlation, partial correlation or precision. This option is passed to nilearn.connectome.ConnectivityMeasure.

# Method to define regions of interests to compute connectivity
method: {config.get("method")} # Method to determine ROIs to compute variance. Uses the Schaeffer 2018 with 100 rois by default. More options are described in the documentation.

# Other parameters
high_pass: {config.get("high_pass")} # High-pass filtering, in Hz, applied to BOLD data. Low (<0.008 Hz) values does minimal changes to the signal, while high (>0.01) values improves sensitivity.
low_pass: {config.get("low_pass")} # Low-pass filtering, in Hz, applied to BOLD data. High (>0.1 Hz) values does minimal changes to the signal, while low (< 0.08 Hz)values improves specificity.
seeds_file: {config["seeds_file"]} # Path to seed file for seed-based ROIs
radius: {config["radius"]} # Radius, in mm, to create the spheres at the coordinates of the seeds for seed-based ROIs
    """

    # Build filenames for each output
    yaml_file = Path(output_dir) / 'config' / 'default_participant_level_config.yaml'

    ensure_directory(yaml_file)

    # Save the YAML content with comments
    with open(yaml_file, 'w') as yaml_out:
        yaml_out.write(yaml_content_with_comments)

    print(f"Default YAML configuration file saved at {yaml_file}. Go to github.com/ln2t/connectomix for more details.")
    print("See also below for the output:")
    print(yaml_content_with_comments)


# Helper function to create default configuration file based on what the dataset contains at group level
def create_group_level_default_config_file(bids_dir, output_dir):
    """
    Create default configuration file in YAML format for default parameters, at group level.
    Configuration file is saved at 'derivatives/config/default_group_level_config.yaml'.

    Parameters
    ----------
    bids_dir : str or Path
        Path to BIDS directory.
    derivatives_dir : str or Path
        Path to derivatives.

    Returns
    -------
    None.

    """

    # Create a BIDSLayout to parse the BIDS dataset
    layout = BIDSLayout(bids_dir, derivatives=[output_dir])

    # Print some stuff for the primate using this function
    print("Generating default configuration file for default parameters, please wait while the dataset is explored...")

    # Load default configuration for several types of analysis
    independent_config = set_unspecified_group_level_options_to_default(dict(analysis_type='independent'), layout)
    paired_config = set_unspecified_group_level_options_to_default(dict(analysis_type='paired'), layout)
    regression_config = set_unspecified_group_level_options_to_default(dict(analysis_type='regression'), layout)

    # The default configuration is 'independet' but we also use the other ones in comments to help the user
    config = independent_config

    # Prepare the YAML content with comments
    yaml_content_with_comments = f"""\
# Connectomix Configuration File
# This file is generated automatically. Please modify the parameters as needed.
# Full documentation is located at github.com/ln2t/connectomix
# All parameters are set to their plausible or default value

analysis_label: {config.get("analysis_label")}  # Custom name for the analysis, e.g. ControlVersuSPatients, PreTreatmentVersuSPostTreatment, or EffectOfIQWithoutAge

# Analysis type
analysis_type: {config.get("analysis_type")}  # Choose from independent, paired, or regression. If regression is selected, provide also one covariate and optionnaly a list of confounds in analysis_options.

# Statistical alpha-level thresholds
uncorrected_alpha: {config.get("uncorrected_alpha")}  # Without multiple-comparison correction
fdr_alpha: {config.get("fdr_alpha")}  # Used in the BH-FDR multiple-comparison correction method
fwe_alpha: {config.get("fwe_alpha")}  # Used in the Family-Wise Error multiple-comparison correction method (maximum and minimum t-statistic distributions estimated from permutations of the data).

# Number of permutations to estimate the null distributions
n_permutations: {config.get("n_permutations")}  # Can be kept to a low value for testing purposes (e.g. 20). If increased, computational time goes up. Reliable results are expected for very large value, e.g. 10000.

# Selected task
tasks: {config.get("tasks")}

# Selected run
runs: {config.get("runs")}

# Selected session
sessions: {config.get("sessions")}

# Selected space
spaces: {config.get("spaces")}

# Groups to compare: names and subjects
group1_name: {config.get("group1_name")}
group2_name: {config.get("group2_name")}
group1_subjects: {config.get("group1_subjects")}
group2_subjects: {config.get("group2_subjects")}
# Paired analysis specifications
# subjects : {paired_config["subjects"]}  # Subjects to include in the paired analysis
# sample1_entities :  # These entities altogether must match exaclty two scans be subject
    # tasks: {paired_config["sample1_entities"]["tasks"]}
    # sessions: {paired_config["sample1_entities"]["sessions"]}
    # runs: {paired_config["sample1_entities"]["runs"]}
# sample2_entities : 
    # tasks: {paired_config["sample2_entities"]["tasks"]}
    # sessions: {paired_config["sample2_entities"]["sessions"]}
    # runs: {paired_config["sample2_entities"]["runs"]}
# Regression parameters
# subjects_to_regress: {regression_config.get("subjects_to_regress")}  # Subjects to include in the regression analysis
# covariate: {regression_config.get("covariate")}  # Covariate for analysis type 'regression'
# confounds: {regression_config.get("confounds")}  # Confounds for analysis type 'regression' (optionnal)

# Kind of connectivity used at participant-level
connectivity_kind: {config.get("connectivity_kinds")}

# Method used at participant-level
method: {config.get("method")}
    """

    # Build filenames for each output
    yaml_file = Path(output_dir) / 'config' / 'default_group_level_config.yaml'

    ensure_directory(yaml_file)

    # Save the YAML content with comments
    with open(yaml_file, 'w') as yaml_out:
        yaml_out.write(yaml_content_with_comments)

    print(f"Default YAML configuration file saved at {yaml_file}. Go to github.com/ln2t/connectomix for more details.")
    print("See also below for the output:")
    print(yaml_content_with_comments)

