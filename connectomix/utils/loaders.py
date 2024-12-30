import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from nilearn import datasets
from nilearn.plotting import find_parcellation_cut_coords

from connectomix.utils.tools import apply_nonbids_filter


# LOADERS
# Helper function to fetch atlas maps, labels and coords
def get_atlas_data(atlas_name, get_cut_coords=False):
    """
    A wrapper function for nilearn.datasets atlas-fetching utils.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to fetch. Choose from 'schaeffer100', 'aal' or 'harvardoxford'.
    get_cut_coords : bool, optional
        If true, cut coords for the regions of the atlas will be computed. The default is False, as this is typically time-consuming.

    Returns
    -------
    maps : Nifti1Image
        The atlas maps.
    labels : list of strings.
        Labels of the atlas regions.
    coords : list of list of three integers
        The coordinates of the regions, in the same order as 'labels'.

    """

    if atlas_name == "schaeffer100":
        warnings.warn("Using Schaefer 2018 atlas with 100 rois")
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
        maps = atlas["maps"]
        coords = find_parcellation_cut_coords(labels_img=maps) if get_cut_coords else []
        labels = atlas["labels"]
    elif atlas_name == "aal":
        warnings.warn("Using AAL atlas")
        atlas = datasets.fetch_atlas_aal()
        maps = atlas["maps"]
        coords = find_parcellation_cut_coords(labels_img=atlas['maps']) if get_cut_coords else []
        labels = atlas["labels"]
    elif atlas_name == "harvardoxford":
        warnings.warn("Using Harvard-Oxford atlas (cort-maxprob-thr25-1mm)")
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
        maps = atlas["maps"]
        coords = find_parcellation_cut_coords(labels_img=atlas['maps']) if get_cut_coords else []
        labels = atlas["labels"]
        labels = labels[1:]  # Needed as first entry is 'background'
    else:
        raise ValueError(f"Requested atlas {atlas_name} is not supported. Check spelling or documentation.")
    return maps, labels, coords


# Helper function to read the repetition time (TR) from a JSON file
def get_repetition_time(json_file):
    """
    Extract repetition time from BOLD sidecar json file.

    Parameters
    ----------
    json_file : str or Path
        Path to BOLD sidecar file.

    Returns
    -------
    float
        Repetition time, in seconds.

    """
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    return metadata.get('RepetitionTime', None)


# Helper function to collect participant-level matrices
def retrieve_connectivity_matrices_from_particpant_level(subjects, layout, entities, method):
    """
    Tool to retrieve the paths to the connectivity matices computed at participant-level.

    Parameters
    ----------
    subjects : list
        List of participant ID to consider.
    layout : BIDSLayout
        The usual BIDS class for the dataset.
    entities : dict
        Entities used to filter BIDSLayout.get() call.
    method : str
        Name of method to select the appropriate files.

    Raises
    ------
    FileNotFoundError
        No connectivity matrix is found, probably an error in the entities.
    ValueError
        Too many connectivity matrices are found, probably an error in the entities.

    Returns
    -------
    group_dict : dict
        A dictionary with keys = subjects and values = path to the unique connectivity matrix to the subject.

    """
    group_dict = {}
    for subject in subjects:
        conn_files = layout.derivatives["connectomix"].get(subject=subject,
                                                           suffix="matrix",
                                                           **entities,
                                                           return_type='filename',
                                                           invalid_filters='allow',
                                                           extension='.npy')
        # Refine selection with non-BIDS entity filtering
        conn_files = apply_nonbids_filter("method", method, conn_files)
        if len(conn_files) == 0:
            raise FileNotFoundError(
                f"Connectivity matrix for subject {subject} not found, are you sure you ran the participant-level pipeline?")
        elif len(conn_files) == 1:
            group_dict[subject] = np.load(conn_files[0])  # Load the match
        else:
            raise ValueError(
                f"There are multiple matches for subject {subject}, review your configuration. Matches are {conn_files}")
    return group_dict


def get_maps_from_participant_level(subjects, layout, entities, method):
    """
    Tool to retrieve the paths to the effect maps computed at participant-level (this is for the roiToVoxel method)

    Parameters
    ----------
    subjects : list
        List of participant ID to consider.
    layout : BIDSLayout
        The usual BIDS class for the dataset.
    entities : dict
        Entities used to filter BIDSLayout.get() call.
    method : str
        Name of method to select the appropriate files.

    Raises
    ------
    FileNotFoundError
        No map is found, probably an error in the entities.
    ValueError
        Too many maps are found, probably an error in the entities.

    Returns
    -------
    group_dict : dict
        A dictionary with keys = subjects and values = path to the unique effect map to the subject.

    """
    group_dict = {}

    local_entities = entities.copy()

    if 'seed' in local_entities.keys():
        seed = local_entities['seed']
        local_entities.pop('seed')
    else:
        seed = None

    if 'desc' in local_entities.keys():
        local_entities.pop('desc')

    for subject in subjects:
        map_files = layout.derivatives["connectomix"].get(subject=subject,
                                                          suffix="effectSize",
                                                          **local_entities,
                                                          return_type='filename',
                                                          invalid_filters='allow',
                                                          extension='.nii.gz')
        # Refine selection with non-BIDS entity filtering
        map_files = apply_nonbids_filter("method", method, map_files)
        map_files = map_files if seed is None else apply_nonbids_filter("seed", seed, map_files)
        if len(map_files) == 0:
            raise FileNotFoundError(
                f"Maps for subject {subject} not found, are you sure you ran the participant-level pipeline?")
        elif len(map_files) == 1:
            group_dict[subject] = map_files[0]
        else:
            raise ValueError(
                f"There are multiple matches for subject {subject}, review your configuration. Matches are {map_files}")
    return group_dict


# Fucntion to get participant-level data for paired analysis
def retrieve_connectivity_matrices_for_paired_samples(layout, entities, config):
    """
    returns: A dict with key equal to each subject and whose value is a length-2 list with the loaded connectivity matrices
    """
    subjects = config["subjects"]

    # Extract sample-defining entities - some manual operation is required here as BIDSLayout uses singular words (e.g. 'run' unstead of 'runs')
    sample1_entities = entities.copy()
    sample1_entities['task'] = config["sample1_entities"]['tasks']
    sample1_entities['session'] = config["sample1_entities"]['sessions']
    sample1_entities['run'] = config["sample1_entities"]['runs']

    sample2_entities = entities.copy()
    sample2_entities['task'] = config["sample2_entities"]['tasks']
    sample2_entities['session'] = config["sample2_entities"]['sessions']
    sample2_entities['run'] = config["sample2_entities"]['runs']

    method = config["method"]

    sample1_dict = retrieve_connectivity_matrices_from_particpant_level(subjects, layout, sample1_entities, method)
    sample2_dict = retrieve_connectivity_matrices_from_particpant_level(subjects, layout, sample2_entities, method)

    # Let's make a consistency check, just to make sure we have what we think we have
    # This is probably not necessary though
    for subject in sample1_dict.keys():
        if subject not in sample2_dict.keys():
            raise KeyError(
                f"Second sample does not contain requested subject {subject}, something is wrong. Maybe a bug?")
    for subject in sample2_dict.keys():
        if subject not in sample1_dict.keys():
            raise KeyError(
                f"First sample does not contain requested subject {subject}, something is wrong. Maybe a bug?")

    # Unify the data in one dict
    paired_samples = {}
    for subject in subjects:
        paired_samples[subject] = [sample1_dict[subject], sample2_dict[subject]]

    return paired_samples


# Tool to extract covariate and confounds from participants.tsv in the same order as in given subject list
def retrieve_info_from_participant_table(layout, subjects, covariate, confounds=None):
    """
    Tool to extract data of interest from the participants.tsv file of the dataset.

    Parameters
    ----------
    layout : BIDSLayout
        Usual BIDS class for the dataset.
    subjects : list
        Subjects for which the data must be extracted.
    covariate : str
        Column name of participants.tsv which is to be extracted.
    confounds : list, optional
        List of strings, each of which corresponding to a column name of participants.tsv, and to be loaded as confounds. The default is None.

    Raises
    ------
    ValueError
        Name of covariate or confound does not exist in the columns of participants.tsv.

    Returns
    -------
    DataFrame
        Table to specified subjects and covariate value, optionally also with selected confounds.

    """
    # Load participants.tsv
    participants_file = layout.get(return_type='filename', extension='tsv', scope='raw')[0]
    participants_df = pd.read_csv(participants_file, sep='\t')

    # Ensure the measure exists in participants.tsv
    if covariate not in participants_df.columns:
        raise ValueError(f"The covariate '{covariate}' is not found in 'participants.tsv'.")

    # Check if confounds exist in the participants.tsv file
    if confounds:
        for confound in confounds:
            if confound not in participants_df.columns:
                raise ValueError(f"The confound '{confound}' is not found in 'participants.tsv'.")

    # List of columns to extract (measure and confounds)
    columns_to_extract = [covariate]
    if confounds:
        columns_to_extract.extend(confounds)

    # Create an empty DataFrame to store the extracted data
    extracted_data = pd.DataFrame()

    # Extract the measure and confounds for each subject in subjects_list
    for subject in subjects:
        # Find the row corresponding to the subject
        subject_row = participants_df.loc[participants_df['participant_id'] == 'sub-' + subject, columns_to_extract]

        # Check if the subject exists in the participants.tsv file
        if subject_row.empty:
            raise ValueError(f"Subject '{subject}' is not found in 'participants.tsv'.")

        # Append the subject's data to the extracted DataFrame
        extracted_data = pd.concat([extracted_data, subject_row])

    # Reset the index of the resulting DataFrame
    extracted_data.reset_index(drop=True, inplace=True)

    return extracted_data  # This is a DataFrame


# Helper function to select confounds
def select_confounds(confounds_file, config):
    """
    Extract confounds selected for denoising from fMRIPrep confounds file.

    Parameters
    ----------
    confounds_file : str or Path
        Path to fMRIPrep confounds file.
    config : dict
        Configuration dict.

    Raises
    ------
    ValueError
        If requested confound is not found in the columns of fMRIPrep confounds file.

    Returns
    -------
    selected_confounds : DataFrame
        Confounds to regression from fMRI signal.

    """
    confounds = pd.read_csv(confounds_file, delimiter='\t')

    # First check selected confound columns are valid names
    for confound_column in config.get("confound_columns"):
        if not confound_column in confounds.columns:
            raise ValueError(f"Confounds column {confound_column} is not a valid confound name.")

    # If aroma denoising is used, make sure confounds do not contain motion parameters and warn user
    if config["ica_aroma"]:
        motion_parameters = ["trans_x", "trans_x_derivative1", "trans_x_derivative1_power2", "trans_x_power2",
                             "trans_y", "trans_y_derivative1", "trans_y_derivative1_power2", "trans_y_power2",
                             "trans_z", "trans_z_derivative1", "trans_z_power2", "trans_z_derivative1_power2",
                             "rot_x", "rot_x_derivative1", "rot_x_derivative1_power2", "rot_x_power2",
                             "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2",
                             "rot_z", "rot_z_derivative1", "rot_z_power2", "rot_z_derivative1_power2"]
        for motion_parameter in motion_parameters:
            if motion_parameter in config["confound_columns"]:
                config["confound_columns"].remove(motion_parameter)
                warnings.warn(
                    f"Motion parameter {motion_parameter} is detected in the confounds list, but you have selected aroma-denoising, which already deals with motion paramters. Removing {motion_parameter} from the confounds list.")

    # Select the confounds
    selected_confounds = confounds[config.get("confound_columns")]

    # Deal with NaN in confound values
    # Todo: implement better method to deal with NaN's. Those are always present when taking derivatives of confounds and nilearn trows an error. Maybe a bug in nilearn? Open an issue?
    # warnings.warn("If NaNs are present in the confounds, they are replaced by zero to ensure compatibility with nilearn. This is potentially very wrong.")
    # selected_confounds = selected_confounds.fillna(0)
    return selected_confounds


# Helper function to load the configuration file
def load_config(config):
    """
    Load configuration either from dict or config file.

    Parameters
    ----------
    config : dict, str or Path
        If dict, a configuration dict. If str or Path, path to the configuration file to load.

    Raises
    ------
    FileNotFoundError
        If file to load configuration is not found.
    TypeError
        If type of config is not dict, str or Path.

    Returns
    -------
    dict
        Configuration dict.

    """

    if isinstance(config, dict):
        return config
    else:
        if isinstance(config, (str, Path)):
            config = Path(config)
            if not config.exists():
                raise FileNotFoundError(f"File not found: {config}")

            # Detect file extension
            file_extension = config.suffix.lower()

            # Load JSON file
            if file_extension == ".json":
                with open(config, 'r') as file:
                    return json.load(file)

            # Load YAML file
            elif file_extension in [".yaml", ".yml"]:
                with open(config, 'r') as file:
                    return yaml.safe_load(file)
            else:
                raise TypeError(
                    f"Wrong configuration data {config}. Must provide either path (to .json or .yaml or .yml) or dict.")


def load_seed_file(seeds_file):
    # Read seed labels and coordinates from file
    if os.path.isfile(seeds_file):
        with open(seeds_file) as seeds_file:
            tsv_file = csv.reader(seeds_file, delimiter="\t")
            labels = []
            coords = []
            for line in tsv_file:
                labels.append(line[0])
                coords.append(np.array(line[1:4], dtype=int))
    else:
        raise FileNotFoundError(f"Seeds file {seeds_file} not found")

    # Remove spaces, dashes and underscores from labels
    labels = [label.replace('_', '').replace(' ', '').replace('-', '') for label in labels]

    # Verify that we still have a unique label for each seed
    if not len(labels) == len(set(labels)):
        raise ValueError(
            f"All labels loaded from {seeds_file} are not unique after removing spaces, dashes or underscores. Please correct this in your seeds file.")

    return coords, labels
