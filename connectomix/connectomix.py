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
from pathlib import Path
from nireports.assembler.report import Report
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

# Helper function to create a summary table of regions or components
def create_summary_table(atlas_labels=None):
    if atlas_labels is not None:
        summary_df = pd.DataFrame({"Region Name": atlas_labels})
    else:
        summary_df = pd.DataFrame()
    return summary_df

# Helper function to generate a Nireports report using reportlets
def generate_report(report_filename, connectivity_results, config):
    report = Report(title=f"Connectomix Report: {report_filename.split('/')[-1]}",
                    out_dir='/'.join(report_filename.split('/')[:-1]),
                    run_uuid='dummyuid')

    # Add sections for each connectivity type
        
    for connectivity_type, result in connectivity_results.items():
        report.add_subsubheader(f"Connectivity Type: {connectivity_type}")
        report.add_image(result['conn_matrix_plot_path'], title=f"Connectivity Matrix ({connectivity_type})")

    # Save the final report
    report.save(report_filename)
    print(f"Report saved to {report_filename}")

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
    
    # ROI-based extraction
    elif method == 'roi':
        coords = method_options['roi_coords']
        radius = method_options['radius']
        masker = NiftiSpheresMasker(
            seeds=coords,
            radius=radius,
            standardize=True,
            detrend=True,
            high_pass=high_pass,
            low_pass=low_pass,
            t_r=t_r
        )
        timeseries = masker.fit_transform(func_file, confounds=confounds.values)

    # ICA-based extraction
    elif method == 'ica':
# Todo: update here the extractor features high pass, low pass, and tr
        timeseries = method_options["extractor"].transform(func_file, confounds=confounds.values)

    else:
        raise ValueError(f"Unknown method: {method}")

    return timeseries

# Compute CanICA component images
# Todo: add file with paths to func files used to compute ICA, generate hash, use hash to name both component IMG and text file.
def compute_canica_components(func_filenames, output_dir, t_r, options):
    # Build path to save canICA components
    canica_filename = output_dir / "canica_components.nii.gz"
    
# Todo: remove this two steps
    # Set filter options based on the config file
    high_pass = options.get('high_pass', None)
    low_pass = options.get('low_pass', None)
    
    # Print warning about assuming the same t_r for all files
    warnings.warn(f"Assuming all files have the same t_r of {t_r} seconds.")
    
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
# Todo: remove high pass, low pass, and tr from these calls or Region extractor.
    extractor = RegionExtractor(
        canica_filename,
        threshold=options.get('threshold', 0.5),
        standardize=True,
        detrend=True,
        min_region_size=options.get('min_region_size', 50),
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=t_r
    )
    extractor.fit()
    print(f"Number of ICA-based components extracted: {extractor.regions_img_.shape[-1]}")
    return canica_filename, extractor

# Helper function to copy config to path
def save_copy_of_config(config, path):
    # If config is a str, assume it is a path and copy
    if isinstance(config, str):
        shutil.copy(config, path)
    # Otherwise, it is a dict and must be dumped to path
    elif isinstance(config, dict):
        json.dump(config, path)
    return None
            

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
    
    # Get task, session, run and space from config file
    task = config.get("task", "restingstate")
    run = config.get("run", None)
    session = config.get("session", None)
    space = config.get("space", 'MNI152NLin2009cAsym')

    # Create a BIDSLayout to parse the BIDS dataset
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir, output_dir])

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
# Todo: adapt arguments here and in definition 
        config['method_options']['components'], config['method_options']['extractor'] = compute_canica_components(resampled_files,
                                                                           canica_dir,
                                                                           get_repetition_time(json_files[0]),
                                                                           config["method_options"])

    # Iterate through each functional file
    for (func_file, confound_file, json_file) in zip(resampled_files, confound_files, json_files):
# Todo: print status like 'processing func file'
        # Extract timeseries
        method = config['method']
        method_options = config['method_options']
        timeseries = extract_timeseries(str(func_file),
                                        str(confound_file),
                                        get_repetition_time(json_file),
                                        method,
                                        method_options)

        # Iterate over each connectivity type
        connectivity_results = {}
        for connectivity_type in connectivity_types:
            print(f"Computing connectivity: {connectivity_type}")
            # Compute connectivity
            connectivity_measure = ConnectivityMeasure(kind=connectivity_type)
            conn_matrix = connectivity_measure.fit_transform([timeseries])[0]
        
            # Generate the BIDS-compliant filename for the connectivity matrix and figure
            entities = layout.parse_file_entities(func_file)
# Todo: create a JSON file with component IMG hash and also path to file.
# Todo: add subfolder in path patterns, create corresponding directory before saving with a function ensure_path_is_in_existing_dir
            conn_matrix_path = layout.derivatives['connectomix'].build_path(entities,
                                                      path_patterns=['sub-{subject}_task-{task}_space-{space}_method-%s_desc-%s_matrix.npy' % (method, connectivity_type)],
                                                      validate=False)
            conn_matrix_plot_path = layout.derivatives['connectomix'].build_path(entities,
                                                      path_patterns=['sub-{subject}_task-{task}_space-{space}_method-%s_desc-%s_matrix.png' % (method, connectivity_type)],
                                                      validate=False)
            np.save(conn_matrix_path, conn_matrix)
            plt.figure(figsize=(10, 10))
            #plot_matrix(conn_matrix, labels=masker.labels_, colorbar=True)
            plot_matrix(conn_matrix, colorbar=True)
            plt.savefig(conn_matrix_plot_path)
            plt.close()
    
            # Store results for this connectivity type
            connectivity_results[connectivity_type] = {
                'timeseries': timeseries,
                'conn_matrix': conn_matrix
            }        

        # Generate a single report with sections for each atlas and subsections for each connectivity type
        report_filename = layout.derivatives['connectomix'].build_path(entities,
                                                  path_patterns=['sub-{subject}_task-{task}_space-{space}_method-%s_report.html' % method],
                                                  validate=False)
        generate_report(report_filename, connectivity_results, config)

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

# config_file = "/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix_config_test.json"

config = "/data/ds005418/code/connectomix/config.json"

main(bids_dir, connectomix_dir, fmriprep_dir, config)
