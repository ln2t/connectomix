#!/usr/bin/env python3
"""BIDS app to compute connectomes from fmri data preprocessed with FMRIPrep

Author: Antonin Rovai

Created: August 2022
"""
import os
import argparse
import json
import pandas as pd
import numpy as np
import shutil
import warnings
from nilearn.image import resample_img, load_img
from nilearn.plotting import plot_matrix
from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.decomposition import CanICA
from nilearn.regions import RegionExtractor
from bids import BIDSLayout
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Define the version number
__version__ = "1.0.0"

# Helper function to load the configuration file
def load_config(config):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    elif isinstance(config, dict):
        pass
    else:
        raise TypeError(f"Wrong configuration data {config}. Must provide either path to json or dict.")
    return config

# Helper function to select confounds
# Todo: allow for more flexible choices from config
def select_confounds(confounds_file):
    confounds = pd.read_csv(confounds_file, delimiter='\t')
    selected_confounds = confounds[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'global_signal']]
    return selected_confounds

# Helper function to read the repetition time (TR) from a JSON file
def get_repetition_time(json_file):
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    return metadata.get('RepetitionTime', None)

# Helper function to generate a dataset_description.json file
def create_dataset_description(output_dir):
    description = {
        "Name": "connectomix",
        "BIDSVersion": "1.6.0",
        "PipelineDescription": {
            "Name": "connectomix",
            "Version": __version__,
            "CodeURL": "https://github.com/ln2t/connectomix"
        }
    }
    with open(output_dir / "dataset_description.json", 'w') as f:
        json.dump(description, f, indent=4)

# Helper function to resample all functional images to a reference image
def resample_to_reference(func_files, reference_img, preprocessing_dir):
    resampled_files = []
    for func_file in func_files:
        #todo: build filename with "resampled" at a place where it makes more sense for BIDS parsing.
        resampled_path = preprocessing_dir / f"{Path(func_file).name}"
        resampled_files.append(str(resampled_path))
        if not os.path.isfile(resampled_path):
            img = load_img(func_file)
            #todo: check if resampling is necessary by comparing affine and grid. If not, just copy the file.
            resampled_img = resample_img(img, target_affine=reference_img.affine, target_shape=reference_img.shape[:3], interpolation='nearest')
            resampled_img.to_filename(resampled_path)
        else:
            print(f"Functional file {resampled_path} already exist, skipping resampling.")
    return resampled_files

# Extract time series based on specified method
def extract_timeseries(func_file, confounds_file, t_r, method, method_options):
    confounds = select_confounds(confounds_file)

    # Set filter options based on the config file
    high_pass = method_options.get('high_pass', None)
    low_pass = method_options.get('low_pass', None)

    # Atlas-based extraction
    if method == 'atlas':
        atlas_img = method_options['atlas_path']
        masker = NiftiLabelsMasker(
            atlas_img=atlas_img,
            standardize=True,
            detrend=True,
            high_pass=high_pass,
            low_pass=low_pass,
            t_r=t_r
        )
        timeseries = masker.fit_transform(func_file, confounds=confounds.values)
        # Todo: find labels for atlas
        labels = None
    
    # ROI-based extraction
    elif method == 'roi':
        
        # Read seed labels and coordinates from file
        if os.path.isfile(method_options['seeds_file']):
            with open(method_options['seeds_file']) as seed_file:
                tsv_file = csv.reader(seed_file, delimiter="\t")
                labels = []
                coords = []
                for line in tsv_file:
                    labels.append(line[0])
                    coords.append(np.array(line[1:4], dtype=int))
        else:
            raise FileNotFoundError(f"Seeds file {method_options['seeds_file']} not found")
            
        radius = method_options['radius']
        masker = NiftiSpheresMasker(
            seeds=coords,
            radius=float(radius),
            standardize=True,
            detrend=True,
            high_pass=high_pass,
            low_pass=low_pass,
            t_r=t_r
        )
        timeseries = masker.fit_transform(func_file, confounds=confounds.values)

    # ICA-based extraction
    elif method == 'ica':
        extractor = method_options["extractor"]
        extractor.high_pass = high_pass
        extractor.low_pass  = low_pass
        extractor.t_r = t_r
        timeseries = extractor.transform(func_file, confounds=confounds.values)
        labels = None
    else:
        raise ValueError(f"Unknown method: {method}")

    return timeseries, labels

# Compute CanICA component images
# Todo: add file with paths to func files used to compute ICA, generate hash, use hash to name both component IMG and text file.
def compute_canica_components(func_filenames, output_dir, options):
    # Build path to save canICA components
    canica_filename = output_dir / "canica_components.nii.gz"
        
    # If has not yet been computed, compute canICA components
    if not os.path.isfile(canica_filename):
        # Todo: ensure the options in CanICA are adapted
        canica = CanICA(
            n_components=20,
            memory="nilearn_cache",
            memory_level=2,
            verbose=10,
            mask_strategy="whole-brain-template",
            random_state=0,
            standardize="zscore_sample",
            n_jobs=2,
        )
        canica.fit(func_filenames)
        
        # Save image to output filename
        canica.components_img_.to_filename(canica_filename)
    else:
        print(f"ICA component file {canica_filename} already exist, skipping computation.")
    extractor = RegionExtractor(
        canica_filename,
        threshold=options.get('threshold', 0.5),
        standardize=True,
        detrend=True,
        min_region_size=options.get('min_region_size', 50)
    )
    extractor.fit()
    print(f"Number of ICA-based components extracted: {extractor.regions_img_.shape[-1]}")
    return canica_filename, extractor

# Function to copy config to path
def save_copy_of_config(config, path):
    # If config is a str, assume it is a path and copy
    if isinstance(config, str):
        shutil.copy(config, path)
    # Otherwise, it is a dict and must be dumped to path
    elif isinstance(config, dict):
        with open(path, "w") as fp:
            json.dump(config, fp)
    return None

# Function to create directory in which path is located
def ensure_directory(file_path):
    """
    Ensure that the directory for a given file path exists.
    If it does not exist, create it.
    
    Args:
    file_path (str): The full path to the file, including the filename.

    Example:
    ensure_directory("/path/to/my/directory/filename.txt")
    """
    # Extract the directory path from the given file path
    directory = os.path.dirname(file_path)

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")
 
# Main function to run connectomix
def main(bids_dir, derivatives_dir, fmriprep_dir, config):
    # Print version information
    print(f"Running connectomix version {__version__}")

    # Prepare output directory
    output_dir = Path(derivatives_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the config file to the output directory
# Todo: add date and time to copy of config file to avoid clash
    save_copy_of_config(config, output_dir / "config.json")
    print(f"Configuration file saved to {output_dir / 'config.json'}")
    
    # Create the dataset_description.json file
    create_dataset_description(output_dir)

    # Load the configuration file
    config = load_config(config)
    
    # Create a BIDSLayout to parse the BIDS dataset
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir, output_dir])
    
    # Get task, session, run and space from config file    
    task = config.get("task", layout.derivatives['fMRIPrep'].get_tasks())
    run = config.get("run", layout.derivatives['fMRIPrep'].get_runs())
    session = config.get("session", layout.derivatives['fMRIPrep'].get_sessions())
    space = config.get("space", layout.derivatives['fMRIPrep'].get_spaces())   

    # Select the functional, confound and metadata files
    func_files = layout.derivatives['fMRIPrep'].get(
        suffix='bold',
        extension='nii.gz',
        return_type='filename',
        space=space,
        desc='preproc',
        task=task,
        run=run,
        session=session,
    )
    confound_files = layout.derivatives['fMRIPrep'].get(
        suffix='timeseries',
        extension='tsv',
        return_type='filename',
        task=task,
        run=run,
        session=session
    )
    json_files = layout.derivatives['fMRIPrep'].get(
        suffix='bold',
        extension='json',
        return_type='filename',
        space=space,
        desc='preproc',
        task=task,
        run=run,
        session=session
    )

    if not func_files:
        raise FileNotFoundError("No functional files found")
    if not confound_files:
        raise FileNotFoundError("No confound files found")
    if len(func_files) != len(confound_files):
        raise ValueError(f"Mismatched number of files: func_files {len(func_files)} and confound_files {len(confound_files)}")

    # Todo: add more consistency checks

    print(f"Found {len(func_files)} functional files")

    # Choose the first functional file as the reference for alignment
# Todo: add possibility to specify path to ref in config file
    reference_func_file = load_img(func_files[0])

    # Create a preprocessing directory to store aligned files
    preprocessing_dir = output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    # Resample all functional files to the reference image
    resampled_files = resample_to_reference(func_files, reference_func_file, preprocessing_dir)
    print("All functional files resampled to match the reference image.")

    # Set up connectivity measures
    connectivity_types = config['connectivity_measure']
    if isinstance(connectivity_types, str):
        connectivity_types = [connectivity_types]
        # Todo: raise error if not list

    # Compute CanICA components if necessary and store it in the methods options
    if config['method'] == 'ica':
        # Create a canICA directory to store component images
        canica_dir = output_dir / "canica"
        canica_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute CanICA and export path and extractor in options to be passed to compute time series
        config['method_options']['components'], config['method_options']['extractor'] = compute_canica_components(resampled_files,
                                                                           canica_dir,
                                                                           config["method_options"])

    # Iterate through each functional file
    for (func_file, confound_file, json_file) in zip(resampled_files, confound_files, json_files):
        # Print status
        print(f"Processing file {func_file}")
        
        
        method = config['method']
        method_options = config['method_options']
        
        

        # Generate the BIDS-compliant filename for the timeseries and save
        entities = layout.parse_file_entities(func_file)
        timeseries_path = layout.derivatives['connectomix'].build_path(entities,
                                                  path_patterns=['sub-{subject}/[ses-{ses}/]sub-{subject}_[ses-{ses}_][run-{run}_]task-{task}_space-{space}_method-%s_timeseries.npy' % method],
                                                  validate=False)
        ensure_directory(timeseries_path)
        
        # Extract timeseries
        if os.path.isfile(timeseries_path):
            print(f"Timeseries for {func_file} already exists, skipping computation.")
            timeseries = np.load(timeseries_path)
        else:
            timeseries, labels = extract_timeseries(str(func_file),
                                            str(confound_file),
                                            get_repetition_time(json_file),
                                            method,
                                            method_options)
            np.save(timeseries_path, timeseries)
        
        # Iterate over each connectivity type
        for connectivity_type in connectivity_types:
            print(f"Computing connectivity: {connectivity_type}")
            # Compute connectivity
            connectivity_measure = ConnectivityMeasure(kind=connectivity_type)
            conn_matrix = connectivity_measure.fit_transform([timeseries])[0]
            
            # Mask out the major diagonal
            np.fill_diagonal(conn_matrix, 0)
        
            # Generate the BIDS-compliant filename for the connectivity matrix and save
            # Todo: create a JSON file with component IMG hash and also path to file.
            # Todo: add session and run as optional entities in the path_patterms for each of the outputs. DONE in three places BUT must be tested!
            conn_matrix_path = layout.derivatives['connectomix'].build_path(entities,
                                                      path_patterns=['sub-{subject}/[ses-{ses}/]sub-{subject}_[ses-{ses}_][run-{run}_]task-{task}_space-{space}_method-%s_desc-%s_matrix.npy' % (method, connectivity_type)],
                                                      validate=False)
            ensure_directory(conn_matrix_path)
            np.save(conn_matrix_path, conn_matrix)
            
            # Generate the BIDS-compliant filename for the figure, generate the figure and save
            conn_matrix_plot_path = layout.derivatives['connectomix'].build_path(entities,
                                                      path_patterns=['sub-{subject}/[ses-{ses}/]sub-{subject}_[ses-{ses}_][run-{run}_]task-{task}_space-{space}_method-%s_desc-%s_matrix.svg' % (method, connectivity_type)],
                                                      validate=False)
            ensure_directory(conn_matrix_plot_path)
            plt.figure(figsize=(10, 10))
            #plot_matrix(conn_matrix, labels=masker.labels_, colorbar=True)
            plot_matrix(conn_matrix, labels=labels, colorbar=True)
            plt.savefig(conn_matrix_plot_path)
            plt.close()

# Group-level analysis
def group_level_analysis(bids_dir, derivatives_dir, fmriprep_dir, config):
    # Print version information
    print(f"Running connectomix (Group-level) version {__version__}")

    # Placeholder logic for group-level analysis
    print("Group-level analysis is currently under development.")
    
# if __name__ == '__main__':
#     # Argument parser for command-line inputs
#     parser = argparse.ArgumentParser(description="Connectomix: Functional Connectivity from fMRIPrep outputs")
#     parser.add_argument('bids_dir', type=str, help='BIDS root directory containing the dataset.')
#     parser.add_argument('derivatives_dir', type=str, help='Directory where to store the outputs.')
#     parser.add_argument('participant', type=str, help='Participant label to process (e.g., "sub-01").')
#     parser.add_argument('--config', type=str, help='Path to the configuration file.', required=True)

#     args = parser.parse_args()

#     # Run the main function
#     main(args.bids_dir, args.derivatives_dir, args.participant, args.config, args.fmriprep_dir)

# with open("/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/code/CTL_vs_FDA_config_test.json", "r") as json_file:
#     data = json.load(json_file)

# bids_dir = data["bids_dir"]
# fmriprep_dir = data["fmriprep_dir"]
# canica_dir = data["canica_dir"]
# connectomes_dir = data["connectomes_dir"]
# connectomix_dir = data["connectomix_dir"]
# confounds_dir = data["confounds_dir"]
# group1_regex = data["group1_regex"]
# group2_regex = data["group2_regex"]
# group1_label = data["group1_label"]
# group2_label = data["group2_label"]
# seeds_file = data["seeds_file"]

bids_dir = "/data/ds005418"
fmriprep_dir = "/data/ds005418/derivatives/fmriprep_v21.0.4"
connectomix_dir = "/data/ds005418/derivatives/connectomix_dev"

bids_dir = "/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/rawdata"
fmriprep_dir = "/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/derivatives/fmriprep_v23.1.3"
connectomix_dir = "/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/derivatives/connectomix_dev"

# config_file = "/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix_config_test.json"

# config = "/data/ds005418/code/connectomix/config.json"

config = {}
config["method"] = "ica"
config["method_options"] = {}
config["connectivity_measure"] = "correlation"
config["session"] = "1"
config["task"] = "restingstate"
config["space"] = "MNI152NLin2009cAsym"

config = {}
config["method"] = "roi"
config["method_options"] = {}
config["method_options"]["seeds_file"] = "/home/arovai/git/arovai/connectomix/connectomix/seeds/brain_and_cerebellum_seeds.tsv"
config["method_options"]["radius"] = "5"
config["connectivity_measure"] = "correlation"
config["session"] = "1"
config["task"] = "restingstate"
config["space"] = "MNI152NLin2009cAsym"

main(bids_dir, connectomix_dir, fmriprep_dir, config)
