import json

import nibabel as nib
import argparse
import os
from nibabel import Nifti1Image
from nilearn.image import load_img, resample_img, index_img, clean_img
from nilearn.reporting import get_clusters_table
from bids import BIDSLayout
from pathlib import Path

import numpy as np


def config_helper(config, key, default, choose_from=None):
    if key in config.keys():
        value = config[key]
    else:
        custom_print(f"Setting config field \"{key}\" to default value \"{default}\"")
        value = default

    if choose_from:
        if not value in choose_from:
            raise ValueError(f"Unsupported value {value} for config field {key}. Supported values are {choose_from}.")
    return value


def print_subject(layout, func_file):
    entities = layout.parse_file_entities(func_file)
    custom_print(f"Processing subject {entities['subject']}")


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


def img_is_not_empty(img):
    """
    Check if a NIfTI image has at least one non-zero voxel.
    """
    # Get the data array
    if type(img) == Nifti1Image:
        data = img.get_fdata()
    else:
        data = img

    # Check if there is at least one non-zero voxel
    return np.any(data != 0)


def resample_to_reference(layout, func_files, config):
    """
    Resamples files to reference, and save the result to a BIDS compliant location.
    Skips resampling if file already exists.

    Parameters
    ----------
    layout : BIDSLayout
        Usual BIDS class for the dataset.
    func_files : list
        Paths to func files to resample.
    reference_img : str or Nifti1Image
        Rerefence image to which all the others will be resampled to.

    Returns
    -------
    resampled_files : list
        Paths to the resampled files.

    """

    # Choose the first functional file as the reference for alignment
    if config["reference_functional_file"] == "first_functional_file":
        config["reference_functional_file"] = func_files[0]
    else:
        config["reference_functional_file"] = Path(layout.root) / config["reference_functional_file"]
    reference_img = load_img(config["reference_functional_file"])

    resampled_files = []
    for func_file in func_files:
        # Build BIDS-compliant filename for resampled data
        entities = layout.derivatives.get_pipeline("connectomix").parse_file_entities(func_file)
        resampled_path = layout.derivatives.get_pipeline("connectomix").build_path(entities,
                                                                      path_patterns=[
                                                                          'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_desc-resampled.nii.gz'],
                                                                      validate=False)

        make_parent_dir(resampled_path)
        resampled_files.append(str(resampled_path))

        # Resample to reference if file do not exist
        if not os.path.isfile(resampled_path):
            img = load_img(func_file)
            # We round the affine as sometimes there are mismatch (small numerical errors?) in fMRIPrep's output
            img = Nifti1Image(img.get_fdata(), affine=np.round(img.affine, 2), header=img.header)
            from connectomix.core.utils.tools import check_affines_match
            if check_affines_match([img, reference_img]):
                resampled_img = img
            else:
                custom_print("Doing some resampling, please wait...")
                resampled_img = resample_img(img, target_affine=reference_img.affine,
                                             target_shape=reference_img.shape[:3],
                                             interpolation='nearest',
                                             force_resample=True)

            resampled_img.to_filename(resampled_path)
        else:
            custom_print(f"Functional file {os.path.basename(resampled_path)} already exist, skipping resampling.")
    return resampled_files


def camel_case_list_of_strings(strings):
    strings = [element.lower().capitalize() for element in strings]
    strings[0] = strings[0].lower()
    return strings


def make_parent_dir(file_path):
    """
    Ensure that the directory for a given file path exists.
    If it does not exist, create it.

    Args:
    file_path (str): The full path to the file, including the filename.

    Example:
    ensure_directory("/path/to/my/directory/filename.txt")
    """
    Path(file_path).parents[0].mkdir(exist_ok=True, parents=True)


def denoise(layout, resampled_files, confound_files, json_files, config):
    """
    Tool to denoise fmri files based on confounds specified in config.

    Parameters
    ----------
    layout : BIDSLayout
    resampled_files : str or Path
    confound_files: str or Path
    json_files: str or Path
    config : dict

    Returns
    -------
    denoised_files : list
    """

    # Denoise the data
    denoised_paths = []
    for (func_file, confound_file, json_file) in zip(resampled_files, confound_files, json_files):
        custom_print(f"Denoising file {os.path.basename(func_file)}")
        entities = layout.parse_file_entities(func_file)
        denoised_path = func_file if config['ica_aroma'] else layout.derivatives.get_pipeline("connectomix").build_path(entities,
                                                                                                           path_patterns=[
                                                                                                               'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_denoised.nii.gz'],
                                                                                                           validate=False)
        denoised_paths.append(denoised_path)

        if not Path(denoised_path).exists() or (Path(denoised_path).exists() and config["overwrite_denoised_files"]):
            make_parent_dir(denoised_path)

            from connectomix.core.utils.loaders import load_confounds
            confounds = load_confounds(str(confound_file), config)

            # Set filter options based on the config file
            high_pass = config["high_pass"]
            low_pass = config["low_pass"]

            from connectomix.core.utils.loaders import load_repetition_time
            clean_img(func_file,
                      low_pass=low_pass,
                      high_pass=high_pass,
                      t_r=load_repetition_time(json_file),
                      confounds=confounds).to_filename(denoised_path)

            denoised_json_path = denoised_path.replace(".nii.gz", ".json")
            denoised_json_dict = {"confounds": config.get("confounds")}
            with open(denoised_json_path, "w") as f:
                json.dump(denoised_json_dict, f, indent=4)

        else:
            custom_print(f"Denoised data {os.path.basename(denoised_path)} already exists, skipping.")
    return denoised_paths


def find_labels_and_coords(config):
    labels = []
    coords = []
    if config["method"] == "seedToVoxel" or config["method"] == "seedToSeed":
        from connectomix.core.utils.loaders import load_seed_file
        coords, labels = load_seed_file(config["seeds_file"])
    elif config["method"] == "roiToVoxel" or config["method"] == "roiToRoi":
        if config["method"] == "roiToVoxel":
            labels = list(config["roi_masks"].keys())
            coords = [None for _ in labels]
        elif config["method"] == "roiToRoi" and not config.get("canica", False):
            from connectomix.core.utils.loaders import load_atlas_data
            _, labels, coords = load_atlas_data(config["atlas"], get_cut_coords=True)
        if config.get("canica", False):
            labels = None
            coords = None

    return labels, coords


def locate_clusters_on_atlas(cluster_table, config):
    # TODO write this function
    pass


def get_cluster_tables(significant_data, config):
    cluster_tables = {}
    for thresholding_strategy in significant_data.keys():
        if significant_data[thresholding_strategy] is not None:
            cluster_table = get_clusters_table(significant_data[thresholding_strategy],
                                               stat_threshold=0,
                                               two_sided=config["two_sided_test"])
            if cluster_table.empty:
                custom_print("No signigificant cluster found.")
            else:
                custom_print(f"Found {len(cluster_table)} cluster(s) and subcluster(s) for {thresholding_strategy}")
                if "atlas" in config.keys():
                    cluster_table = locate_clusters_on_atlas(cluster_table, config)
                else:
                    custom_print("No atlas provided, skipping cluster location identification")
        else:
            cluster_table = None

        cluster_tables[thresholding_strategy] = cluster_table
    return cluster_tables


def custom_print(*args, **kwargs):
    # ANSI escape codes for colors
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

    # Print with the specified color
    print(f"{CYAN}", end="")
    print(*args, **kwargs)
    print(f"{RESET}", end="")


def setup_terminal_colors():
    import warnings
    import traceback
    import sys



    # ANSI escape codes for colors
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    def custom_warning_format(message, category, filename, lineno, line=None):
        # Define the color for the warning message
        return f"{YELLOW}{filename}:{lineno}: {category.__name__}: {message}{RESET}\n"

    # Set the custom warning formatter
    warnings.formatwarning = custom_warning_format

    # ANSI escape codes for colors
    RED = '\033[91m'
    RESET = '\033[0m'

    def custom_exception_handler(exc_type, exc_value, exc_traceback):
        # Format the exception traceback with color
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"{RED}{tb_str}{RESET}", end="")

    # Set the custom exception handler
    sys.excepthook = custom_exception_handler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connectomix: Functional Connectivity from fMRIPrep outputs using BIDS structure")

    # Define the autonomous flag
    parser.add_argument("--autonomous", action="store_true",
                        help="Run the script in autonomous mode, guessing paths and settings.")

    # Define the run flag
    parser.add_argument("--run", action="store_true", help="Run the analysis based on what the autonomous mode found.")

    # Define positional arguments for bids_dir, derivatives_dir, and analysis_level
    parser.add_argument("bids_dir", nargs="?", type=str, help="BIDS root directory containing the dataset.")
    parser.add_argument("output_dir", nargs="?", type=str, help="Directory where to store the outputs.")
    parser.add_argument("analysis_level", nargs="?", choices=["participant", "group"],
                        help="Analysis level: either 'participant' or 'group'.")

    # Define optional arguments that apply to both analysis levels
    parser.add_argument("-d", "--derivatives", nargs="+",
                        help="Specify pre-computed derivatives as 'key=value' pairs (e.g., -d fmriprep=/path/to/fmriprep fmripost-aroma=/path/to/fmripost-aroma).")
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file.")
    parser.add_argument("-p", "--participant_label", type=str, help="Participant label to process (e.g., '01').")
    parser.add_argument("-s", "--session", type=str, help="Session to process (e.g., '1').")
    parser.add_argument("-t", "--task", type=str, help="Task to process (e.g., 'restingstate').")
    parser.add_argument("--denoising", type=str, help="A predefined denoising strategy (e.g. 'gs_csfwm_12p')")
    parser.add_argument("--helper", help="Helper function to write default configuration files.", action="store_true")

    return parser.parse_args()