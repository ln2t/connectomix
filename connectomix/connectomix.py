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
from nilearn import datasets
from bids import BIDSLayout
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, permutation_test
from statsmodels.stats.multitest import multipletests
import hashlib

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
def select_confounds(confounds_file, config):
    confounds = pd.read_csv(confounds_file, delimiter='\t')
    default_confound_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'global_signal']
    selected_confounds = confounds[config.get("confound_columns", default_confound_columns)]
    # Todo: check that provided columns exist in confounds
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
def resample_to_reference(layout, func_files, reference_img):
    resampled_files = []
    for func_file in func_files:
        # Build BIDS-compliant filename for resampled data
        entities = layout.derivatives["connectomix"].parse_file_entities(func_file)
        resampled_path = layout.derivatives["connectomix"].build_path(entities,
                          path_patterns=['sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_desc-resampled.nii.gz'],
                          validate=False)
        
        ensure_directory(resampled_path)
        resampled_files.append(str(resampled_path))
        
        # Resample to reference if file does not exists
        if not os.path.isfile(resampled_path):
            img = load_img(func_file)
            #todo: check if resampling is necessary by comparing affine and grid. If not, just copy the file.
            resampled_img = resample_img(img, target_affine=reference_img.affine, target_shape=reference_img.shape[:3], interpolation='nearest')
            resampled_img.to_filename(resampled_path)
        else:
            print(f"Functional file {resampled_path} already exist, skipping resampling.")
    return resampled_files

# Extract time series based on specified method
def extract_timeseries(func_file, confounds_file, t_r, config):
    confounds = select_confounds(confounds_file, config)
    
    method = config['method']
    method_options = config['method_options']

    # Set filter options based on the config file
    high_pass = method_options.get('high_pass', None)
    low_pass = method_options.get('low_pass', None)

    # Atlas-based extraction
    if method == 'atlas':
        # Load the default atlas and inform user
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
        warnings.warn("Using Harvard-Oxford atlas cort-maxprob-thr25-1mm.")
        labels = atlas["labels"]
        
        # Define masker object and proceed with timeseries computation
        masker = NiftiLabelsMasker(
            labels_img=atlas["filename"],
            labels=labels,
            standardize=True,
            detrend=True,
            high_pass=high_pass,
            low_pass=low_pass,
            t_r=t_r
        )
        timeseries = masker.fit_transform(func_file, confounds=confounds.values)        
        # Drop the first entry which is always the Background label
        labels = labels[1:]
    
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
 
# Main function to run connectomix
def main(bids_dir, derivatives_dir, fmriprep_dir, config):
    # Print version information
    print(f"Running connectomix version {__version__}")

    # Prepare output directory
    output_dir = Path(derivatives_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the config file to the output directory
    # Todo: add date and time to copy of config file to avoid clash
    if not "method_options" in config.keys():
        config["method_options"] = {}
    save_copy_of_config(config, output_dir / "config.json")
    print(f"Configuration file saved to {output_dir / 'config.json'}")
    
    # Create the dataset_description.json file
    create_dataset_description(output_dir)

    # Load the configuration file
    config = load_config(config)
    
    # Create a BIDSLayout to parse the BIDS dataset
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir, output_dir])
    
    # Get subjects, task, session, run and space from config file    
    subjects = config.get("subjects", layout.derivatives['fMRIPrep'].get_subjects())
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
        subject=subjects,
        task=task,
        run=run,
        session=session,
    )
    confound_files = layout.derivatives['fMRIPrep'].get(
        suffix='timeseries',
        extension='tsv',
        return_type='filename',
        subject=subjects,
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
        subject=subjects,
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
    reference_func_file = load_img(config.get("reference_functional_file", func_files[0]))

    # Resample all functional files to the reference image
    resampled_files = resample_to_reference(layout, func_files, reference_func_file)
    print("All functional files resampled to match the reference image.")

    # Set up connectivity measures
    connectivity_types = config['connectivity_measure']
    if isinstance(connectivity_types, str):
        connectivity_types = [connectivity_types]
    elif not isinstance(connectivity_types, list):
        raise ValueError(f"The connectivity_types value must either be a string or a list. You provided {connectivity_types}.")

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
               
        
        # Generate the BIDS-compliant filename for the timeseries and save
        entities = layout.parse_file_entities(func_file)
        timeseries_path = layout.derivatives['connectomix'].build_path(entities,
                                                  path_patterns=['sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-%s_timeseries.npy' % config['method']],
                                                  validate=False)
        ensure_directory(timeseries_path)
        
        # Extract timeseries
        timeseries, labels = extract_timeseries(str(func_file),
                                        str(confound_file),
                                        get_repetition_time(json_file),
                                        config)
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
            conn_matrix_path = layout.derivatives['connectomix'].build_path(entities,
                                                      path_patterns=['sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-%s_desc-%s_matrix.npy' % (config["method"], connectivity_type)],
                                                      validate=False)
            ensure_directory(conn_matrix_path)
            np.save(conn_matrix_path, conn_matrix)
            
            # Generate the BIDS-compliant filename for the figure, generate the figure and save
            conn_matrix_plot_path = layout.derivatives['connectomix'].build_path(entities,
                                                      path_patterns=['sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-%s_desc-%s_matrix.svg' % (config["method"], connectivity_type)],
                                                      validate=False)
            ensure_directory(conn_matrix_plot_path)
            plt.figure(figsize=(10, 10))
            plot_matrix(conn_matrix, labels=labels, colorbar=True)
            plt.savefig(conn_matrix_plot_path)
            plt.close()

# Custom non-valid entity filter
def apply_nonbids_filter(entity, value, files):
    filtered_files = []
    if not entity == "suffix":
        entity = f"{entity}-"
    for file in files:
        if f"{entity}{value}" in os.path.basename(file).split("_"):
            filtered_files.append(file)
    return filtered_files

# Permutation testing with stat max thresholding
# Todo: completely review this function. It is basically a placeholder.
def generate_permuted_null_distributions(group1_data, group2_data, config, layout, entities):
    """
    Perform a two-sided permutation test to determine positive and negative thresholds separately.
    Returns separate maximum and minimum thresholds for positive and negative t-values.
    """   
    # Extract values from config
    n_permutations = config.get("n_permutations", 10000)
    
    # Load pre-existing permuted data, if any
    perm_files = apply_nonbids_filter("comparison",
                         config["comparison_label"],
                         layout.derivatives["connectomix"].get(extension=".npy",
                                                               suffix="permutations",
                                                               return_type='filename'))
    
    perm_null_distributions = []
    for perm_file in perm_files:
        perm_data = np.load(perm_file)
        if len(perm_null_distributions) == 0:
            perm_null_distributions = perm_data
        else:
            perm_null_distributions = np.append(perm_null_distributions, perm_data , axis=0)
            
    # Run permutation testing by chunks and save permuted data
    n_resamples = 10  # number of permutations per chunk
    while len(perm_null_distributions) < n_permutations:
        # Run permutation chunk
        print(f"Running a chunk of permutations... (goal is {n_permutations} permutations)")
        perm_test = permutation_test((group1_data, group2_data),
                                                      stat_func,
                                                      vectorized=False,
                                                      n_resamples=n_resamples)
        
        perm_test.null_distribution
        
        # Build a temporary file before generatin hash
        temp_fn = layout.derivatives["connectomix"].build_path({**entities,
                                                          "comparison_label": config["comparison_label"],
                                                          "method": config["method"],
                                                          },
                                                     path_patterns=["group/permutations/{comparison_label}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_comparison-{comparison_label}_tmp.npy"],
                                                     validate=False)
        
        ensure_directory(temp_fn)
        
        # Clean any temporary file, in case previous run was abruptly interrupted
        if os.path.isfile(temp_fn):
            os.remove(temp_fn)
        
        # Add result to list of permuted data
        if len(perm_null_distributions) == 0:
            perm_null_distributions = perm_test.null_distribution
        else:
            perm_null_distributions = np.append(perm_null_distributions, perm_test.null_distribution, axis=0)
        
        # Save to temporary file
        np.save(temp_fn, perm_test.null_distribution)
        # Generate hash to ensure we save each permutations in a separate file
        h = hashlib.md5(open(temp_fn, 'rb').read()).hexdigest()    
        # Rename temporary file to final filename
        final_fn = layout.derivatives["connectomix"].build_path({**entities,
                                                          "comparison_label": config["comparison_label"],
                                                          "method": config["method"],
                                                          "hash": h
                                                          },
                                                     path_patterns=["group/permutations/{comparison_label}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_comparison-{comparison_label}_hash-{hash}_permutations.npy"],
                                                     validate=False)
        ensure_directory(final_fn)
        os.rename(temp_fn, final_fn)
    
    # Extract max and min stat distributions
    null_max_distribution = np.nanmax(perm_null_distributions, axis=(1, 2))
    null_min_distribution = np.nanmin(perm_null_distributions, axis=(1, 2))
        
    return null_max_distribution, null_min_distribution

# Define a function to compute the difference in connectivity between the two groups
# Todo: adapt this for paired tests
def stat_func(x, y, axis=0):
    from scipy.stats import ttest_ind
    # Compute the t-statistic between two independent groups
    t_stat, _ = ttest_ind(x, y)
    return t_stat

# Group-level analysis
def group_level_analysis(bids_dir, derivatives_dir, fmriprep_dir, config):
    # Print version information
    print(f"Running connectomix (Group-level) version {__version__}")

    # Load config
    config = load_config(config)

    # Create a BIDSLayout to handle files and outputs
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir, derivatives_dir])
    
    # Load group specifications from config
    group1_subjects = config['group1_subjects']
    group2_subjects = config['group2_subjects']
    
    # Retrieve connectivity type and other configuration parameters
    connectivity_type = config['connectivity_measure']
    method = config['method']
    task = config.get("task", "restingstate")
    run = config.get("run", None)
    session = config.get("session", None)
    space = config.get("space", "MNI152NLin2009cAsym")
    analysis_type = config.get("analysis_type", "independent")  # Options: 'independent' or 'paired'
    
    entities = {
        "task": task,
        "space": space,
        "session": session,
        "run": run,
        "desc": connectivity_type,
        "extension": ".npy"
    }
    
    # Retrieve the connectivity matrices for group 1 and group 2 using BIDSLayout
    group1_matrices = []
    for subject in group1_subjects:
        conn_files = layout.derivatives["connectomix"].get(subject=subject,
                                                           suffix="matrix",
                                                           **entities,
                                                           return_type='filename',
                                                           invalid_filters='allow')
        # Refine selection with non-BIDS entity filtering
        conn_files = apply_nonbids_filter("method", method, conn_files)
        if len(conn_files) == 0:
            raise FileNotFoundError(f"Connectivity matrix for subject {subject} not found, are you sure you ran the participant-level pipeline?")
        elif len(conn_files) == 1:
            group1_matrices.append(np.load(conn_files[0]))  # Load the match
        else:
            raise ValueError(f"There are multiple matches for subject {subject}, review your configuration.")
            
    group2_matrices = []
    for subject in group2_subjects:
        conn_files = layout.derivatives["connectomix"].get(subject=subject,
                                                           suffix="matrix",
                                                           **entities,
                                                           return_type='filename',
                                                           invalid_filters='allow')
        # Refine selection with non-BIDS entity filtering
        conn_files = apply_nonbids_filter("method", method, conn_files)
        if len(conn_files) == 0:
            raise FileNotFoundError(f"Connectivity matrix for subject {subject} not found, are you sure you ran the participant-level pipeline?")
        elif len(conn_files) == 1:
            group2_matrices.append(np.load(conn_files[0]))  # Load the match
        else:
            raise ValueError(f"There are multiple matches for subject {subject}, review your configuration.")
    
    # Convert to 3D arrays: (subjects, nodes, nodes)
    # Todo: make sure this is actually necessary
    group1_data = np.stack(group1_matrices, axis=0)
    group2_data = np.stack(group2_matrices, axis=0)
    
    # Perform the appropriate group-level analysis
    if analysis_type == "independent":
        # Independent t-test between different subjects
        t_stats, p_values = ttest_ind(group1_data, group2_data, axis=0, equal_var=False)
    elif analysis_type == "paired":
        # Paired t-test within the same subjects
        if len(group1_subjects) != len(group2_subjects):
            raise ValueError("Paired t-test requires an equal number of subjects in both groups.")
        t_stats, p_values = ttest_rel(group1_data, group2_data, axis=0)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Threshold 1: Uncorrected p-value
    p_uncorr_threshold = config.get("uncorrected_threshold", 0.001)
    uncorr_mask = p_values < p_uncorr_threshold

    # Threshold 2: FDR correction
    fdr_threshold = config.get("fdr_threshold", 0.05)
    fdr_mask = multipletests(p_values.flatten(), alpha=fdr_threshold, method='fdr_bh')[0].reshape(p_values.shape)

    # Threshold 3: Permutation-based threshold
    n_permutations = config.get("n_permutations", 10000)
    null_max_distribution, null_min_distribution = generate_permuted_null_distributions(group1_data, group2_data, config, layout, entities)
    
    # Compute thresholds at desired significance
    alpha = float(config.get("fwe_alpha", "0.05"))
    t_max = np.percentile(null_max_distribution, (1 - alpha / 2) * 100)
    t_min = np.percentile(null_min_distribution, alpha / 2 * 100)
    
    perm_mask = (t_stats > permuted_max) | (t_stats < permuted_min)
    
    # Save thresholds to a BIDS-compliant JSON file
    thresholds = {
        "uncorrected_threshold": p_uncorr_threshold,
        "fdr_threshold": fdr_threshold,
        "permutation_thresholds": {
            "max_positive": permuted_max["positive"].tolist(),
            "min_negative": permuted_min["negative"].tolist(),
            "n_permutations": n_permutations
        }
    }
    threshold_file = layout.derivatives['connectomix'].build_path(entities, 
                                       path_patterns=['group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-%s_desc-%s_thresholds.json' % (method, connectivity_type)],
                                       validate=False)

    with open(threshold_file, 'w') as f:
        json.dump(thresholds, f, indent=4)
    
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
connectomix_dir = "/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/derivatives/connectomix_dev_wip"

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
config["confound_columns"] = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'global_signal']
# config["reference_functional_file"] = "/path/to/func/ref"
config["subjects"] = "CTL01"

config = {}
config["method"] = "atlas"
config["connectivity_measure"] = "correlation"
config["session"] = "1"
config["task"] = "restingstate"
config["space"] = "MNI152NLin2009cAsym"
config["confound_columns"] = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'global_signal']
# config["reference_functional_file"] = "/path/to/func/ref"
config["subjects"] = ["CTL01", "CTL02", "CTL03", "CTL04", "CTL05", "CTL06", "CTL07", "CTL08", "CTL10"]
config["subjects"] = ["CTL09"]

# main(bids_dir, connectomix_dir, fmriprep_dir, config)

config = {}
config["method"] = "atlas"
config["connectivity_measure"] = "correlation"
config["session"] = "1"
config["task"] = "restingstate"
config["space"] = "MNI152NLin2009cAsym"
config["group1_subjects"] = ["CTL01", "CTL02", "CTL03", "CTL04"]
config["group2_subjects"] = ["CTL05", "CTL06", "CTL07", "CTL08", "CTL09", "CTL10"]
config["n_permutations"] = 15
config["comparison_label"] = "CTL0102vsCTL0304TEST"

group_level_analysis(bids_dir, connectomix_dir, fmriprep_dir, config)
