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
from nilearn.plotting import plot_matrix, plot_carpet
from nilearn.input_data import NiftiLabelsMasker
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
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Helper function to select confounds
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
            "CodeURL": "https://github.com/arovai/connectomix"
        }
    }
    with open(output_dir / "dataset_description.json", 'w') as f:
        json.dump(description, f, indent=4)

# Helper function to resample all functional images to a reference image
def resample_to_reference(func_files, reference_img, preprocessing_dir):
    resampled_files = []
    for func_file in func_files:
        #todo: build filename with "resampled" at a place where it makes more sense for BIDS parsing.
        resampled_path = preprocessing_dir / f"resampled_{Path(func_file).name}"
        resampled_files.append(str(resampled_path))
        if not os.path.isfile(resampled_path):
            img = load_img(func_file)
            #todo: check if resampling is necessary by comparing affine and grid. If not, just copy the file.
            resampled_img = resample_img(img, target_affine=reference_img.affine, target_shape=reference_img.shape[:3], interpolation='nearest')
            resampled_img.to_filename(resampled_path)
    return resampled_files

# Helper function to create a summary table of regions or components
def create_summary_table(atlas_labels=None):
    if atlas_labels is not None:
        summary_df = pd.DataFrame({"Region Name": atlas_labels})
    else:
        summary_df = pd.DataFrame()
    return summary_df

# Helper function to generate a Nireports report using reportlets
def generate_report(output_dir, report_filename, atlas_results, config):
    report = Report(title=f"Connectomix Report: {report_filename}")

    # Add sections for each atlas and connectivity type
    for atlas_name, connectivity_results in atlas_results.items():
        report.add_subheader(atlas_name)
        for connectivity_type, result in connectivity_results.items():
            report.add_subsubheader(f"Connectivity Type: {connectivity_type}")
            report.add_carpet(result['carpet_plot_path'], title=f"Time Series Carpet Plot ({atlas_name}, {connectivity_type})")
            report.add_image(result['conn_matrix_plot_path'], title=f"Connectivity Matrix ({atlas_name}, {connectivity_type})")
            report.add_dataframe(result['summary_table'], title=f"Region Information ({atlas_name}, {connectivity_type})")

    # Save the final report
    report_path = output_dir / f"{report_filename}_report.html"
    report.save(report_path)
    print(f"Report saved to {report_path}")

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

    # ICA-based extraction
    elif method == 'ica':
        extractor = RegionExtractor(
            method_options['component_images'],
            threshold=method_options.get('threshold', 0.5),
            standardize=True,
            detrend=True,
            min_region_size=method_options.get('min_region_size', 50),
            high_pass=high_pass,
            low_pass=low_pass,
            t_r=t_r
        )
        extractor.fit()
        masker = NiftiLabelsMasker(labels_img=extractor.regions_img_, standardize=True, detrend=True)
        print(f"Number of ICA-based components extracted: {extractor.regions_img_.shape[-1]}")

    else:
        raise ValueError(f"Unknown method: {method}")

    timeseries = masker.fit_transform(func_file, confounds=confounds.values)
    return timeseries

# Compute CanICA component images
def compute_canica_components(func_filenames, output_dir):
    # Build path to save canICA components
    canica_filename = output_dir / "canICA_components.nii.gz"
    
    # If has not yet been computed, compute canICA components
    if not os.path.isfile(canica_filename):
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
    return canica_filename

# Main function to run connectomix
def main(bids_dir, derivatives_dir, fmriprep_dir, config_file):
    # Print version information
    print(f"Running connectomix version {__version__}")

    # Prepare output directory
    output_dir = Path(derivatives_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the config file to the output directory
    shutil.copy(config_file, output_dir / "config.json")
    print(f"Configuration file saved to {output_dir / 'config.json'}")
    
    # Create the dataset_description.json file
    create_dataset_description(output_dir)

    # Load the configuration file
    config = load_config(config_file)

    # Create a BIDSLayout to parse the BIDS dataset
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir, output_dir])

    # Select the functional files and confound files for the given participant
    func_files = layout.derivatives['fMRIPrep'].get(
        suffix='bold',
        extension='nii.gz',
        return_type='filename',
        space='MNI152NLin2009cAsym',
        desc='preproc',
        session='1',
        task='restingstate',
        run=None
    )
    confound_files = layout.derivatives['fMRIPrep'].get(
        suffix='timeseries',
        extension='tsv',
        return_type='filename',
        session='1',
        task='restingstate',
        run=None
    )
    json_files = layout.derivatives['fMRIPrep'].get(
        suffix='bold',
        extension='json',
        return_type='filename',
        space='MNI152NLin2009cAsym',
        desc='preproc',
        session='1',
        task='restingstate',
        run=None
    )

    if not func_files:
        raise FileNotFoundError("No functional files found")
    if not confound_files:
        raise FileNotFoundError("No confound files found")
    if len(func_files) != len(confound_files):
        raise ValueError(f"Mismatched number of files: func_files {len(func_files)} and confound_files {len(confound_files)}")

    print(f"Found {len(func_files)} functional files")

    # Choose the first functional file as the reference for alignment
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

    # Compute CanICA components if necessary and store it in the methods options
    if config['method'] == 'ica':
        # Create a canICA directory to store component images
        canica_dir = output_dir / "canICA"
        canica_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute CanICA and export filename in options
        config['method_options']['component_images'] = compute_canica_components(resampled_files, canica_dir)

    # Iterate through each functional file
    for (func_file, confound_file, json_file) in zip(resampled_files, confound_files, json_files):
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
        
            # Generate the BIDS-compliant filename for the connectivity matrix figure
            entities = layout.parse_file_entities(func_file)
            conn_matrix_plot_path = layout['connectomix'].build_path(entities,
                                                      path_patterns=['sub-{subject}_task-{task}_space-{space}_method-%s_desc-%s_matrix.png' % (method, connectivity_type)],
                                                      validate=False)

            # Create output paths
            conn_matrix_plot_path = output_dir / f"sub-{participant}_atlas-{atlas_name}_connectivity-{connectivity_type}_matrix.png"
            plt.figure(figsize=(10, 10))
            plot_matrix(conn_matrix, labels=masker.labels_, colorbar=True)
            plt.savefig(conn_matrix_plot_path)
            plt.close()
    
            # Create carpet plot path
            carpet_plot_path = output_dir / f"sub-{participant}_atlas-{atlas_name}_connectivity-{connectivity_type}_carpet_plot.png"
            plot_carpet(timeseries, title=f"Carpet Plot ({atlas_name}, {connectivity_type})", figure=carpet_plot_path)
    
            # Create a summary table
            summary_table = create_summary_table(atlas_labels=masker.labels_)
    
            # Store results for this connectivity type
            connectivity_results[connectivity_type] = {
                'timeseries': timeseries,
                'conn_matrix': conn_matrix,
                'summary_table': summary_table,
                'conn_matrix_plot_path': conn_matrix_plot_path,
                'carpet_plot_path': carpet_plot_path
            }        

    # Iterate through each atlas and generate results for each connectivity type
    atlas_results = {}
    for atlas_path in config['method_options']['atlas_path']:
        # Extract atlas name from the filename
        atlas_name = Path(atlas_path).stem
        print(f"Processing atlas: {atlas_name}")

        # Extract timeseries using the current atlas
        masker = NiftiLabelsMasker(atlas_img=atlas_path, standardize=True)
        timeseries = masker.fit_transform(resampled_files, confounds=[select_confounds(c) for c in confound_files])

        

        # Store results for this atlas
        atlas_results[atlas_name] = connectivity_results

    # Generate a single report with sections for each atlas and subsections for each connectivity type
    report_filename = f"sub-{participant}_task-{layout.parse_file_entities(func_files[0]).get('task', 'unknown')}_space-MNI152_report"
    generate_report(output_dir, report_filename, atlas_results, config)

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


with open("/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/code/CTL_vs_FDA_config_test.json", "r") as json_file:
    data = json.load(json_file)

bids_dir = data["bids_dir"]
fmriprep_dir = data["fmriprep_dir"]
canica_dir = data["canica_dir"]
connectomes_dir = data["connectomes_dir"]
connectomix_dir = data["connectomix_dir"]
confounds_dir = data["confounds_dir"]
group1_regex = data["group1_regex"]
group2_regex = data["group2_regex"]
group1_label = data["group1_label"]
group2_label = data["group2_label"]
seeds_file = data["seeds_file"]

config_file = "/mnt/hdd_10Tb_internal/gin/datasets/2021-Hilarious_Mosquito-978d4dbc2f38/code/connectomix_config_test.json"

main(bids_dir, connectomix_dir, fmriprep_dir, config_file)
