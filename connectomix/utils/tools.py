import os
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.image import index_img
from bids import BIDSLayout
from pathlib import Path
from datetime import datetime
import warnings

from connectomix.utils.loaders import load_config
from connectomix.utils.makers import save_copy_of_config
from connectomix.utils.setup import set_unspecified_participant_level_options_to_default, \
    set_unspecified_group_level_options_to_default

def setup_layout(bids_dir, output_dir, derivatives=dict()):
    # Create derivative directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the dataset_description.json file
    create_dataset_description(output_dir)

    # Create a BIDSLayout to parse the BIDS dataset and index also the derivatives
    return BIDSLayout(bids_dir, derivatives=[*list(derivatives.values()), output_dir])


def setup_config(layout, config, level):
    config = load_config(config)

    # Set unspecified config options to default values
    if level == "participant":
        config = set_unspecified_participant_level_options_to_default(config, layout)
    elif level == "group":
        config = set_unspecified_group_level_options_to_default(config, layout)

    # Get the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save a copy of the config file to the config directory
    config_filename = Path(
        layout.derivatives["connectomix"].root) / "config" / "backups" / f"participant_level_config_{timestamp}.json"
    save_copy_of_config(config, config_filename)
    print(f"Configuration file saved to {config_filename}")
    return config


def get_mask(layout, entities):
    entites_for_mask = entities.copy()
    entites_for_mask["desc"] = "brain"
    entites_for_mask["suffix"] = "mask"
    mask_img = layout.derivatives["fMRIPrep"].get(**entites_for_mask)
    if len(mask_img) == 1:
        mask_img = mask_img[0]
    elif len(mask_img) == 0:
        print(entites_for_mask)
        raise ValueError(f"Mask img for entities {entities} not found.")
    else:
        raise ValueError(f"More that one mask for entitites {entities} found: {mask_img}.")
    return mask_img


def get_bids_entities_from_config(config):
    """
    Extract BIDS entities from config file.

    Parameters
    ----------
    config : dict

    Returns
    -------
    dict
        Fields: subject, task, run, session and space. Each field may be str- or list of str- valued

    """
    subject = config.get("subject")
    task = config.get("tasks")
    run = config.get("runs")
    session = config.get("sessions")
    space = config.get("spaces")
    return dict(subject=subject, task=task, run=run, session=session, space=space)


def get_files_for_analysis(layout, config):
    """
    Get functional, json and confound files from layout.derivatives,
    according to parameters in config.

    Parameters
    ----------
    layout : BIDSLayout
    config : dict

    Returns
    -------
    func_files : list
    json_files : list
    confound_files : list
    """
    # Get subjects, task, session, run and space from config file

    entities = get_bids_entities_from_config(config)

    # Select the functional, confound and metadata files
    func_files = layout.derivatives["fMRIPost-AROMA" if config["ica_aroma"] else "fMRIPrep"].get(
        suffix="bold",
        extension="nii.gz",
        return_type="filename",
        desc="nonaggrDenoised" if config["ica_aroma"] else "preproc",
        **entities
    )
    json_files = layout.derivatives["fMRIPrep"].get(
        suffix="bold",
        extension="json",
        return_type="filename",
        desc="preproc",
        **entities
    )

    entities.pop("space")
    confound_files = layout.derivatives["fMRIPrep"].get(
        suffix="timeseries",
        extension="tsv",
        return_type="filename",
        **entities
    )

    # TODO: add warning when some requested subjects don't have matching func files
    if not func_files:
        raise FileNotFoundError("No functional files found")
    if not confound_files:
        raise FileNotFoundError("No confound files found")
    if len(func_files) != len(confound_files):
        raise ValueError(
            f"Mismatched number of files: func_files {len(func_files)} and confound_files {len(confound_files)}")
    if len(func_files) != len(json_files):
        raise ValueError(f"Mismatched number of files: func_files {len(func_files)} and json_files {len(json_files)}")

    return func_files, json_files, confound_files


def setup_and_check_connectivity_kinds(config):
    # Set up connectivity measures
    connectivity_kinds = config["connectivity_kinds"]
    if isinstance(connectivity_kinds, str):
        connectivity_kinds = [connectivity_kinds]
    elif not isinstance(connectivity_kinds, list):
        raise ValueError(
            f"The connectivity_kinds value must either be a string or a list. You provided {connectivity_kinds}.")
    return connectivity_kinds


# Custom non-valid entity filter
def apply_nonbids_filter(entity, value, files):
    """
    Filter paths according to any type of entity, even if not allowed by BIDS.

    Parameters
    ----------
    entity : str
        The name of the entity to filter on (can be anything).
    value : str
        Entity value to filter.
    files : list
        List of paths to filters.

    Returns
    -------
    filtered_files : list
        List of paths after filtering is applied.

    """
    filtered_files = []
    if not entity == "suffix":
        entity = f"{entity}-"
    for file in files:
        if f"{entity}{value}" in os.path.basename(file).split("_"):
            filtered_files.append(file)
    return filtered_files


# Function to compare affines of images, with some tolerance
def check_affines_match(imgs):
    """
    Check if the affines of a list of Niimg objects (or file paths to .nii or .nii.gz) match.

    Parameters:
    - imgs: list of niimgs or paths

    Returns:
    - True if all affines match, False otherwise.
    """
    reference_img = nib.load(imgs[0]) if isinstance(imgs[0], (str, Path)) else imgs[0]
    reference_affine = reference_img.affine

    for img in imgs[1:]:
        img = nib.load(img) if isinstance(img, (str, Path)) else img
        if not np.allclose(img.affine, reference_affine):
            return False
    return True


# Group size verification tool
def check_group_has_several_members(group_subjects):
    """
    A basic tool to check if provided group of subjects actually contain more than one element.

    Parameters
    ----------
    group_subjects : list
        List of subjects.

    Raises
    ------
    ValueError
        Wrong size for the group list.

    Returns
    -------
    None.

    """
    if len(group_subjects) == 0:
        raise ValueError("One group has no member, please review your configuration file.")
    elif len(group_subjects) == 1:
        raise ValueError(
            "Detecting a group with only one member, this is not yet supported. If this is not what you intended to do, review your configuration file.")


# Try to guess groups in the dataset
def guess_groups(layout):
    """
    Reads the participants.tsv file, checks for a "group" column, and returns lists of participants for each group.

    Parameters:
    - layout

    Returns:
    - groups_dict: A dictionary with group names as keys and lists of participant IDs as values.

    Raises:
    - Warning: If there are not exactly two groups.
    """

    # Path to the participants.tsv file
    participants_file = Path(layout.get(extension="tsv", scope="raw", return_type="filename")[0])

    # Read the participants.tsv file
    participants_df = pd.read_csv(participants_file, sep="\t")

    groups_dict = {}

    # Check if the "group" column exists
    if "group" in participants_df.columns:
        # Create lists of participants for each group
        groups_dict = {}
        unique_groups = participants_df["group"].unique()

        # We also need the list of participants that have been processed at participant-level
        processed_participants = layout.derivatives["connectomix"].get_subjects()

        for group_value in unique_groups:
            # Get the list of participant IDs for the current group
            participants_in_group = participants_df.loc[
                participants_df["group"] == group_value, 'participant_id'].tolist()

            # Remove the 'sub-' prefix:
            participants_in_group = [subject.replace('sub-', '') for subject in participants_in_group]

            # Refine selection to keep only participants already processed at participant-level
            groups_dict[group_value] = list(set(processed_participants) & set(participants_in_group))
        # Raise a warning if there are not exactly two groups
        if len(groups_dict) != 2:
            warnings.warn(f"Expected exactly two groups, but found {len(groups_dict)} groups.")
    else:
        warnings.warn("No group column ground in the participants.tsv file, cannot guess any grouping.")

    return groups_dict


# Tool to parse the various derivatives passed to CLI
def parse_derivatives(derivatives_list):
    """Convert list of 'key=value' items into a dictionary."""
    derivatives_dict = {}
    if derivatives_list:
        for item in derivatives_list:
            if '=' in item:
                key, value = item.split('=', 1)
                derivatives_dict[key] = value
            else:
                raise argparse.ArgumentTypeError(
                    f"Invalid format for -d/--derivatives: '{item}' must be in 'key=value' format.")
    return derivatives_dict


# Tool to remove the entity defining the pairs to compare
def remove_pair_making_entity(entities):
    """
    When performing paired tests, only one type of entity can be a list with 2 values (those are used to form pairs).
    This is the "pair making entity". This function sets this entity to None.

    Parameters
    ----------
    entities : dict
        Entities to be used to form pairs in paired test.

    Returns
    -------
    unique_entities : dict
        Same as entities, with one entity set to None if it was a list of length > 1 in the input.

    """
    # Note that this function has no effect on entities in the case of independent samples comparison or during regression analysis
    unique_entities = entities.copy()

    task = entities['task']
    run = entities['run']
    session = entities['session']

    if isinstance(task, list):
        if len(task) > 1:
            unique_entities['task'] = None
    if isinstance(run, list):
        if len(run) > 1:
            unique_entities['run'] = None
    if isinstance(session, list):
        if len(session) > 1:
            unique_entities['session'] = None

    return unique_entities


def convert_4D_to_3D(imgs):
    """
    Convert list of 4D (or 3D) images into list of 3D images, when the fourth dimension contains only one image.

    Parameters
    ----------
    imgs : list
        List of Niimg or str of Path

    Returns
    -------
    imgs_3D : list
    """
    imgs_3D = []
    for img in imgs:
        img = nib.load(img) if isinstance(img, (str, Path)) else img
        if len(img.shape) == 4:
            if img.shape[3] == 1:
                imgs_3D.append(index_img(img, 0))
            else:
                raise ValueError("More that one image in fourth dimension, cannot convert 4D image to 3D")
    return imgs_3D


def create_dataset_description():
    return None


def ensure_directory():
    return None