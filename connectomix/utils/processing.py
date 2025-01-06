from nilearn.decomposition import CanICA
from nilearn.masking import apply_mask, unmask
from nilearn.regions import RegionExtractor
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference, make_second_level_design_matrix
from statsmodels.stats.multitest import multipletests
import os
import json
import pandas as pd
from nibabel import Nifti1Image
from nilearn.image import load_img, resample_img, resample_to_img, clean_img, index_img, math_img, binarize_img
from nilearn.plotting import plot_matrix, plot_connectome, find_parcellation_cut_coords, \
    find_probabilistic_atlas_cut_coords, plot_stat_map, plot_glass_brain, plot_design_matrix
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec, vec_to_sym_matrix
from bids import BIDSLayout
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, permutation_test
import warnings

from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker

import numpy as np

def img_is_not_empty(img):
    """
    Check if a NIfTI image has at least one non-zero voxel.
    """
    # Get the data array
    data = img.get_fdata()

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
    if config.get("reference_functional_file") == "first_functional_file":
        config["reference_functional_file"] = func_files[0]
    reference_img = load_img(config["reference_functional_file"])

    resampled_files = []
    for func_file in func_files:
        # Build BIDS-compliant filename for resampled data
        entities = layout.derivatives["connectomix"].parse_file_entities(func_file)
        resampled_path = layout.derivatives["connectomix"].build_path(entities,
                                                                      path_patterns=[
                                                                          'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_desc-resampled.nii.gz'],
                                                                      validate=False)

        from connectomix.utils.makers import ensure_directory
        ensure_directory(resampled_path)
        resampled_files.append(str(resampled_path))

        # Resample to reference if file do not exist
        if not os.path.isfile(resampled_path):
            img = load_img(func_file)
            # We round the affine as sometimes there are mismatch (small numerical errors?) in fMRIPrep's output
            img = Nifti1Image(img.get_fdata(), affine=np.round(img.affine, 2), header=img.header)
            from connectomix.utils.tools import check_affines_match
            if check_affines_match([img, reference_img]):
                resampled_img = img
            else:
                print("Doing some resampling, please wait...")
                resampled_img = resample_img(img, target_affine=reference_img.affine,
                                             target_shape=reference_img.shape[:3],
                                             interpolation='nearest',
                                             force_resample=True)

            resampled_img.to_filename(resampled_path)
        else:
            print(f"Functional file {os.path.basename(resampled_path)} already exist, skipping resampling.")
    return resampled_files

def compute_canica_components(layout, func_filenames, config):
    """
    Wrapper for nilearn.decomposition.CanICA. Computes group-level ICA components as well as extracts connected regions from the decomposition.

    Parameters
    ----------
    func_filenames : list
        List of path to func files from which to compute the components.
    layout : BIDSLayout
        Layout of the BIDS dataset, including relevant derivatives.

    Returns
    -------
    canica_filename : str
        Path to the savec canica components image.
    extractor : Extractor
        Extractor object from the nilearn package (and already fit to the data at hand).

    """

    from connectomix.utils.tools import get_bids_entities_from_config
    entities = get_bids_entities_from_config(config)
    entities.pop('subject')

    # Build path to save canICA components
    canica_filename = layout.derivatives["connectomix"].build_path(entities,
                                                                   path_patterns=[
                                                                       "canica/[ses-{session}_][run-{run}_]task-{task}_space-{space}_canicacomponents.nii.gz"],
                                                                   validate=False)
    canica_sidecar = layout.derivatives["connectomix"].build_path(entities,
                                                                  path_patterns=[
                                                                      "canica/[ses-{session}_][run-{run}_]task-{task}_space-{space}_canicacomponents.json"],
                                                                  validate=False)
    extracted_regions_filename = layout.derivatives["connectomix"].build_path(entities,
                                                                              path_patterns=[
                                                                                  "canica/[ses-{session}_][run-{run}_]task-{task}_space-{space}_extractedregions.nii.gz"],
                                                                              validate=False)
    extracted_regions_sidecar = layout.derivatives["connectomix"].build_path(entities,
                                                                             path_patterns=[
                                                                                 "canica/[ses-{session}_][run-{run}_]task-{task}_space-{space}_extractedregions.json"],
                                                                             validate=False)

    from connectomix.utils.makers import ensure_directory
    ensure_directory(canica_filename)
    ensure_directory(canica_sidecar)
    ensure_directory(extracted_regions_filename)
    ensure_directory(extracted_regions_sidecar)

    # Define canica parameters
    # Todo: ensure the options in CanICA are adapted
    canica_parameters = dict(n_components=20,
                             memory="nilearn_cache",
                             memory_level=2,
                             verbose=10,
                             mask_strategy="whole-brain-template",
                             random_state=0,
                             standardize="zscore_sample",
                             n_jobs=2)

    # Dump config to file for reproducibility
    with open(canica_sidecar, "w") as fp:
        json.dump({**canica_parameters, "func_filenames": func_filenames}, fp, indent=4)

    # If has not yet been computed, compute canICA components
    if not os.path.isfile(canica_filename):
        canica = CanICA(**canica_parameters)
        canica.fit(func_filenames)

        # Save image to output filename
        print(f"Saving canica components image to {canica_filename}")
        canica.components_img_.to_filename(canica_filename)
    else:
        print(f"ICA component file {os.path.basename(canica_filename)} already exist, skipping computation.")

    extractor_options = dict(threshold=config["canica_threshold"],
                             min_region_size=config["canica_min_region_size"],
                             standardize="zscore_sample",
                             detrend=True)

    # Dump config to file for reproducibility
    with open(extracted_regions_sidecar, "w") as fp:
        json.dump(extractor_options, fp, indent=4)

    # Extract regions from canica components
    extractor = RegionExtractor(
        canica_filename,
        **extractor_options
    )
    extractor.fit()

    print(f"Number of ICA-based components extracted: {extractor.regions_img_.shape[-1]}")

    print(f"Saving extracted ROIs to {extracted_regions_filename}")
    extractor.regions_img_.to_filename(extracted_regions_filename)

    config['components'] = canica_filename
    config['extractor'] = extractor

    return config

def denoise_fmri_data(layout, resampled_files, confound_files, json_files, config):
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
        print(f"Denoising file {func_file}")
        entities = layout.parse_file_entities(func_file)
        denoised_path = func_file if config['ica_aroma'] else layout.derivatives["connectomix"].build_path(entities,
                                                                                                           path_patterns=[
                                                                                                               'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_denoised.nii.gz'],
                                                                                                           validate=False)
        denoised_paths.append(denoised_path)

        if not Path(denoised_path).exists():
            from connectomix.utils.makers import ensure_directory
            ensure_directory(denoised_path)

            from connectomix.utils.loaders import select_confounds
            confounds = select_confounds(str(confound_file), config)

            # Set filter options based on the config file
            high_pass = config['high_pass']
            low_pass = config['low_pass']

            from connectomix.utils.loaders import get_repetition_time
            clean_img(func_file,
                      low_pass=low_pass,
                      high_pass=high_pass,
                      t_r=get_repetition_time(json_file),
                      confounds=confounds).to_filename(denoised_path)
        else:
            print(f"Denoised data {denoised_path} already exists, skipping.")
    return denoised_paths

def preprocessing(layout, config):
    # Select all files needed for analysis
    from connectomix.utils.tools import get_files_for_analysis
    func_files, json_files, confound_files = get_files_for_analysis(layout, config)
    print(f"Found {len(func_files)} functional files:")
    [print(os.path.basename(fn)) for fn in func_files]

    # Resample all functional files to the reference image
    resampled_files = resample_to_reference(layout, func_files, config)
    print("All functional files resampled to match the reference image.")

    denoised_files = denoise_fmri_data(layout, resampled_files, confound_files, json_files, config)
    print("Denoising finished.")

    return denoised_files, json_files

def extract_timeseries(func_file, config):
    """
    Extract timeseries from fMRI data on Regions-Of-Interests (ROIs).

    Parameters
    ----------
    func_file : str or Path
        Path to fMRI data.
    t_r : float
        Repetition Time.
    config : dict
        Configuration parameters.

    Raises
    ------
    FileNotFoundError
        When ROIs are defined using seeds, the seeds are read from a seeds file. This error is raised if the seeds file is not found.

    Returns
    -------
    timeseries : numpy.array
        The extracted time series. Shape is number of ROIs x number of timepoints.
    labels : list
        List of ROIs labels, in the same order as in timeseries.

    """

    method = config["method"]
    t_r = config["t_r"]

    if method == "seedToVoxel" or method == "seedToSeed":
        from connectomix.utils.loaders import load_seed_file
        coords, labels = load_seed_file(config["seeds_file"])

        radius = config["radius"]
        masker = NiftiSpheresMasker(
            seeds=coords,
            radius=float(radius),
            standardize="zscore_sample",
            detrend=False,
            high_pass=None,
            low_pass=None,
            t_r=t_r  # TODO: check if tr is necessary when filtering is not applied
        )
        timeseries = masker.fit_transform(func_file)
    elif method == "roiToVoxel" or method == "roiToRoi":
        if method == "roiToVoxel":
            labels = list(config["roi_masks"].keys())
            imgs = list(config["roi_masks"].values())
            for roi_path in imgs:
                if not os.path.isfile(roi_path):
                    raise FileNotFoundError(
                        f"No file found at provided path {roi_path} for roi_mask. Please review your configuration.")
        elif method == "roiToRoi" and not config["canica"]:
            from connectomix.utils.loaders import get_atlas_data
            imgs, labels, _ = get_atlas_data(method)
            imgs = [imgs]

        if config["canica"]:
            # ICA-based extraction
            extractor = config["extractor"]
            extractor.high_pass = None
            extractor.low_pass = None
            extractor.t_r = t_r
            timeseries = extractor.transform(func_file)
            labels = None
        else:
            timeseries = []
            for img in imgs:
                masker = NiftiLabelsMasker(
                    labels_img=img,
                    standardize="zscore_sample",
                    detrend=False,
                    high_pass=None,
                    low_pass=None,
                    t_r=t_r  # TODO: check if tr is necessary when filtering is not applied
                )
                timeseries.append(masker.fit_transform(func_file))

            timeseries = np.hstack(timeseries)

    # if method in config["supported_atlases"] or (method == "roiToVoxel" and config["roi_masks"] is not None):
    #     if method in config["supported_atlases"]:
    #         from connectomix.utils.loaders import get_atlas_data
    #         imgs, labels, _ = get_atlas_data(method)
    #         imgs = [imgs]
    #     else:
    #         labels = list(config["roi_masks"].keys())
    #         imgs = list(config["roi_masks"].values())
    #
    #         for roi_path in imgs:
    #             if not os.path.isfile(roi_path):
    #                 raise FileNotFoundError(
    #                     f"No file found at provided path {roi_path} for roi_mask. Please review your configuration.")
    #
    #     timeseries = []
    #     for img in imgs:
    #         masker = NiftiLabelsMasker(
    #             labels_img=img,
    #             standardize="zscore_sample",
    #             detrend=False,
    #             high_pass=None,
    #             low_pass=None,
    #             t_r=t_r  # TODO: check if tr is necessary when filtering is not applied
    #         )
    #         timeseries.append(masker.fit_transform(func_file))
    #     timeseries = np.hstack(timeseries)
    # if method == "ica":
    #     # ICA-based extraction
    #     extractor = config["extractor"]
    #     extractor.high_pass = None
    #     extractor.low_pass = None
    #     extractor.t_r = t_r
    #     timeseries = extractor.transform(func_file)
    #     labels = None

    return timeseries, labels

def add_new_entities(entities, label, config):

    match config["method"]:
        case "seedToVoxel":
            new_entity_key = "seed"
            new_entity_val = label
            suffix = "effectSize"
            entities["extension"] = "nii.gz"
        case "roiToVoxel":
            new_entity_key = "roi"
            new_entity_val = label
            suffix = "effectSize"
            entities["extension"] = "nii.gz"
        case "seedToSeed":
            new_entity_key = "seeds"
            new_entity_val = config["custom_seeds_name"]
            suffix = config["connectivity_kind"]
            entities["extension"] = "npy"
        case "roiToRoi":
            new_entity_key = "atlas"
            new_entity_val = config["atlas"]
            suffix = config["connectivity_kind"]
            entities["extension"] = "npy"
        case _:
            new_entity_key = None
            new_entity_val = None
            suffix = None

    entities["method"] = config["method"]
    entities["new_entity_key"] = new_entity_key
    entities["new_entity_val"] = new_entity_val
    entities["suffix"] = suffix

    return entities

def build_output_path(layout, entities, label, config):
    from connectomix.utils.makers import ensure_directory

    entities = add_new_entities(entities, label, config)
    pattern = "sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_{new_entity_key}-{new_entity_val}_{suffix}.{extension}"
    output_path = layout.derivatives["connectomix"].build_path(entities, path_patterns=[pattern], validate=False)
    ensure_directory(output_path)

    return output_path

def glm_analysis_participant_level(t_r, mask_img, func_file, timeseries):
    glm = FirstLevelModel(t_r=t_r,
                          mask_img=resample_to_img(mask_img,
                                                   func_file,
                                                   force_resample=True,
                                                   interpolation="nearest"),
                          high_pass=None,
                          standardize=True)
    frame_times = np.arange(len(timeseries)) * t_r
    design_matrix = make_first_level_design_matrix(frame_times=frame_times,
                                                   events=None,
                                                   hrf_model=None,
                                                   drift_model=None,
                                                   add_regs=timeseries)

    glm.fit(run_imgs=str(func_file),
            design_matrices=design_matrix)

    contrast_vector = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    return glm.compute_contrast(contrast_vector, output_type="effect_size")

def save_roi_to_voxel_map(layout, roi_to_voxel_img, entities, label, config):
    # Create plot of z-score map and save
    roi_to_voxel_plot_path = layout.derivatives["connectomix"].build_path(entities,
                                                                          path_patterns=[
                                                                              'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-%s_seed-%s_plot.svg' % (
                                                                              config["method"], label)],
                                                                          validate=False)
    from connectomix.utils.makers import ensure_directory
    ensure_directory(roi_to_voxel_plot_path)

    if config["seeds_file"] is not None:
        from connectomix.utils.loaders import load_seed_file
        coords, labels = load_seed_file(config["seeds_file"])
        coord = coords[labels.index(label)]

        roi_to_voxel_plot = plot_stat_map(roi_to_voxel_img,
                                          title=f"seed-to-voxel effect size for seed {label} (coords {coords})",
                                          cut_coords=coord)
        roi_to_voxel_plot.add_markers(marker_coords=[coord],
                                      marker_color="k",
                                      marker_size=2 * config["radius"])
    else:
        roi_to_voxel_plot = plot_stat_map(roi_to_voxel_img,
                                          title=f"roi-to-voxel effect size for roi {label}")

    roi_to_voxel_plot.savefig(roi_to_voxel_plot_path)

def roi_to_voxel_single_subject_analysis(layout, func_file, timeseries_list, labels, config):
    """
    Run roi-to-voxel analysis on denoised data. Save the outputs in BIDS derivative format.

    Parameters
    ----------
    layout : BIDSLayout
    func_files : list
    json_files : list
    config : dict

    Returns
    -------
    None.

    """
    entities = layout.parse_file_entities(func_file)
    from connectomix.utils.tools import get_mask
    mask_img = get_mask(layout, entities)

    for timeseries, label in zip(timeseries_list.T, labels):
        roi_to_voxel_img = glm_analysis_participant_level(config["t_r"], mask_img, func_file, timeseries.reshape(-1, 1))
        roi_to_voxel_path = build_output_path(layout, entities, label, config)
        roi_to_voxel_img.to_filename(roi_to_voxel_path)
        save_roi_to_voxel_map(layout, roi_to_voxel_img, entities, label, config)

def save_matrix_plot():
    # Placeholder function
    # TODO: implement this function
    # # Generate the BIDS-compliant filename for the figure, generate the figure and save
    # conn_matrix_plot_path = layout.derivatives["connectomix"].build_path(entities,
    #                                                                      path_patterns=[
    #                                                                          "sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-%s_desc-%s_matrix.svg" % (
    #                                                                          config["method"], connectivity_kind)],
    #                                                                      validate=False)
    # ensure_directory(conn_matrix_plot_path)
    # plt.figure(figsize=(10, 10))
    # plot_matrix(conn_matrix, labels=labels, colorbar=True)
    # plt.savefig(conn_matrix_plot_path)
    # plt.close()
    return None

def roi_to_roi_single_subject_analysis(layout, func_file, timeseries_list, config):
    """
    Run roi-to-roi analysis on denoised data. Save the outputs in BIDS derivative format.

    Parameters
    ----------
    layout : BIDSLayout
    func_files : list
    json_files : list
    config : dict

    Returns
    -------
    None.

    """
    entities = layout.parse_file_entities(func_file)
    connectivity_measure = ConnectivityMeasure(kind=config["connectivity_kind"])
    conn_matrix = connectivity_measure.fit_transform([timeseries_list])[0]
    np.fill_diagonal(conn_matrix, 0)
    conn_matrix_path = build_output_path(layout, entities, None, config)
    np.save(conn_matrix_path, conn_matrix)
    save_matrix_plot()

def single_subject_analysis(layout, func_file, config):

    timeseries_list, labels = extract_timeseries(str(func_file), config)

    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        roi_to_voxel_single_subject_analysis(layout, func_file, timeseries_list, labels, config)
    elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
        roi_to_roi_single_subject_analysis(layout, func_file, timeseries_list, config)

def make_second_level_input(layout, label, config):

    second_level_input = pd.DataFrame(columns=['subject_label', 'map_name', 'effects_map_path'])

    from connectomix.utils.tools import get_bids_entities_from_config
    entities = get_bids_entities_from_config(config)
    map_files = layout.derivatives["connectomix"].get(return_type='filename',
                                                      extension='.nii.gz',
                                                      **entities)
    from connectomix.utils.tools import apply_nonbids_filter
    map_files = apply_nonbids_filter("seed", label, map_files)
    map_files = apply_nonbids_filter("method", config["method"], map_files)

    # TODO: for paired analysis, compute difference of maps and save result to folder.
    if config['analysis_type'] == 'paired':
        raise ValueError('Paired analysis not yet supported')

    for file in map_files:
        file_entities = layout.parse_file_entities(file)
        second_level_input.loc[len(second_level_input)] = [f"sub-{file_entities['subject']}", label, file]

    return second_level_input

def get_group_level_confounds(layout, subjects_label, config):
    participants_file = layout.get(return_type="filename", extension="tsv", scope="raw")[0]
    participants_df = pd.read_csv(participants_file, sep='\t')

    if not isinstance(config["group_confounds"], list):
        config["group_confounds"] = [config["group_confounds"]]

    confounds = participants_df[["participant_id", *config["group_confounds"]]]
    confounds = confounds.rename(columns={"participant_id": "subject_label"}).copy()

    if "group" in config["group_confounds"]:
        group_labels = set(confounds["group"].values)

        for group in group_labels:
            confounds[group] = 0
            confounds.loc[confounds["group"] == group, group] = 1

        confounds.drop(columns=["group"], inplace=True)

    confounds.rename(columns={"participant_id": "subject_label"}, inplace=True)

    return None if len(confounds.columns) == 1 else confounds

def make_group_level_design_matrix(layout, second_level_input, label, config):
    subjects_label = list(second_level_input["subject_label"])
    confounds = get_group_level_confounds(layout, subjects_label, config)

    design_matrix = make_second_level_design_matrix(subjects_label, confounds=confounds)

    if "group" in config["group_confounds"]:
        design_matrix.drop(columns=["intercept"], inplace=True)

    from connectomix.utils.tools import get_bids_entities_from_config
    entities = get_bids_entities_from_config(config)
    entities.pop("subject")
    design_matrix_plot_path = layout.derivatives["connectomix"].build_path({**entities,
                                                                            "analysis_label": config["analysis_label"],
                                                                            "method": config["method"],
                                                                            "seed": label
                                                                            },
                                                                           path_patterns=[
                                                                               "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_seed-{seed}_analysis-{analysis_label}_designMatrix.svg"],
                                                                           validate=False)
    from connectomix.utils.makers import ensure_directory
    ensure_directory(design_matrix_plot_path)
    plot_design_matrix(design_matrix, output_file=design_matrix_plot_path)

    return design_matrix

def compute_group_level_contrast(layout, glm, label, config):
    from connectomix.utils.tools import get_bids_entities_from_config
    entities = get_bids_entities_from_config(config)
    entities.pop("subject")
    entities["seed"] = label
    contrast_label = config["group_contrast"]
    contrast_path = layout.derivatives["connectomix"].build_path({**entities,
                                                                  "analysis_label": config["analysis_label"],
                                                                  "method": config["method"],
                                                                  "seed": label
                                                                  },
                                                                 path_patterns=[
                                                                     "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_seed-{seed}_analysis-{analysis_label}_zScore.nii.gz"],
                                                                 validate=False)

    from connectomix.utils.makers import ensure_directory
    ensure_directory(contrast_path)
    print(f"Computing contrast label named \'{contrast_label}\'")
    glm.compute_contrast(contrast_label,
                         first_level_contrast=label,
                         output_type="z_score").to_filename(contrast_path)
    return contrast_path

def save_group_level_contrast_plots(layout, contrast_path, coord, label, config):
    # Create plot of contrast map and save
    from connectomix.utils.tools import get_bids_entities_from_config
    entities = get_bids_entities_from_config(config)
    entities.pop("subject")
    entities["seed"] = label
    contrast_plot_path = layout.derivatives["connectomix"].build_path({**entities,
                                                                       "analysis_label": config["analysis_label"],
                                                                       "method": config["method"],
                                                                       "seed": label
                                                                       },
                                                                      path_patterns=[
                                                                          "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_seed-{seed}_analysis-{analysis_label}_zScore.svg"],
                                                                      validate=False)
    from connectomix.utils.makers import ensure_directory
    ensure_directory(contrast_plot_path)
    contrast_plot = plot_stat_map(contrast_path,
                                  threshold=3.0,
                                  title=f"roi-to-voxel contrast for seed {label} (coords {coord})",
                                  cut_coords=coord)
    contrast_plot.add_markers(marker_coords=[coord],
                              marker_color="k",
                              marker_size=2 * config["radius"])
    contrast_plot.savefig(contrast_plot_path)

def compute_non_parametric_max_mass(layout, glm, label, config):
    from connectomix.utils.tools import get_bids_entities_from_config
    entities = get_bids_entities_from_config(config)
    entities.pop("subject")
    entities["seed"] = label
    np_logp_max_mass_path = layout.derivatives["connectomix"].build_path({**entities,
                                                                          "analysis_label": config["analysis_label"],
                                                                          "method": config["method"],
                                                                          "seed": label
                                                                          },
                                                                         path_patterns=[
                                                                             "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_seed-{seed}_analysis-{analysis_label}_logpMaxMass.nii.gz"],
                                                                         validate=False)

    from connectomix.utils.makers import ensure_directory
    ensure_directory(np_logp_max_mass_path)
    np_outputs = non_parametric_inference(glm.second_level_input_,
                                          design_matrix=glm.design_matrix_,
                                          second_level_contrast=config["group_contrast"],
                                          first_level_contrast=label,
                                          smoothing_fwhm=config["smoothing"],
                                          two_sided_test=True,  # TODO: put this in config file
                                          n_jobs=2,  # TODO: put this in config file
                                          threshold=float(config["cluster_forming_alpha"]),
                                          n_perm=config["n_permutations"])
    np_outputs["logp_max_mass"].to_filename(np_logp_max_mass_path)
    return np_logp_max_mass_path

def save_significant_contrast_maps(layout, contrast_path, np_logp_max_mass_path, label, config):
    for significance_level in ["uncorrected", "fdr", "fwe"]:
        alpha = float(config[f"{significance_level}_alpha"])
        from connectomix.utils.tools import get_bids_entities_from_config
        entities = get_bids_entities_from_config(config)
        entities.pop("subject")
        entities["seed"] = label
        thresholded_contrast_path = layout.derivatives["connectomix"].build_path({**entities,
                                                                                  "analysis_label": config[
                                                                                      "analysis_label"],
                                                                                  "method": config["method"],
                                                                                  "seed": label,
                                                                                  "significance_level": significance_level
                                                                                  },
                                                                                 path_patterns=[
                                                                                     "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_seed-{seed}_analysis-{analysis_label}_{significance_level}.nii.gz"],
                                                                                 validate=False)
        from connectomix.utils.makers import ensure_directory
        ensure_directory(thresholded_contrast_path)

        match significance_level:
            case "uncorrected":
                thresholded_img, _ = threshold_stats_img(contrast_path, alpha=alpha, height_control=None,
                                                         two_sided=True)
                thresholded_img.to_filename(thresholded_contrast_path)
            case "fdr":
                thresholded_img, _ = threshold_stats_img(contrast_path, alpha=alpha, height_control="fdr",
                                                         two_sided=True)
                thresholded_img.to_filename(thresholded_contrast_path)
            case "fwe":
                mask = math_img(f"img >= -np.log10({alpha})", img=np_logp_max_mass_path)
                mask = binarize_img(mask)

                if img_is_not_empty(mask):
                    masked_data = apply_mask(contrast_path, mask)
                    unmask(masked_data, mask).to_filename(thresholded_contrast_path)
                else:
                    warnings.warn(
                        f"For map {contrast_path}, no voxel survives FWE thresholding at alpha level {alpha}.")

def save_max_mass_plot(layout, np_logp_max_mass_path, label, coords, config):
    from connectomix.utils.tools import get_bids_entities_from_config
    entities = get_bids_entities_from_config(config)
    entities.pop("subject")
    entities["seed"] = label
    np_logp_max_mass_plot_path = layout.derivatives["connectomix"].build_path({**entities,
                                                                               "analysis_label": config[
                                                                                   "analysis_label"],
                                                                               "method": config["method"],
                                                                               "seed": label
                                                                               },
                                                                              path_patterns=[
                                                                                  "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_seed-{seed}_analysis-{analysis_label}_logpMaxMass.svg"],
                                                                              validate=False)
    from connectomix.utils.makers import ensure_directory
    ensure_directory(np_logp_max_mass_plot_path)
    plot_glass_brain(
        np_logp_max_mass_path,
        colorbar=True,
        cmap="autumn",
        vmax=2.69,  # this is hardcoded but that's not a problem as it is only for plots
        display_mode="z",
        plot_abs=False,
        cut_coords=coords,
        threshold=-np.log10(float(config["fwe_alpha"]))).savefig(np_logp_max_mass_plot_path)

def roi_to_voxel_group_analysis(layout, config):
    from connectomix.utils.tools import get_bids_entities_from_config
    entities = get_bids_entities_from_config(config)
    entities.pop("subject")

    if config["seeds_file"]:
        from connectomix.utils.loaders import load_seed_file
        coords, labels = load_seed_file(config["seeds_file"])
    elif config["roi_masks"]:
        labels = list(config["roi_masks"].keys())
        coords = None

    # TODO: add check at group level that config file should either have seeds_file OR roi_masks

    for label in labels:
        second_level_input = make_second_level_input(layout,
                                                     label,
                                                     config)  # get all first-level maps. In paired case, compute differences. Resamples and save all in folder.
        design_matrix = make_group_level_design_matrix(layout, second_level_input, label, config)

        glm = SecondLevelModel(smoothing_fwhm=config["smoothing"])
        glm.fit(second_level_input, design_matrix=design_matrix)

        contrast_path = compute_group_level_contrast(layout, glm, label, config)

        # TODO: there is caveat in this: it does not show the sign of t-score! Direction of effect unknown...
        np_logp_max_mass_path = compute_non_parametric_max_mass(layout, glm, label, config)

        save_significant_contrast_maps(layout, contrast_path, np_logp_max_mass_path, label, config)

    if coords:
        for coord, label in zip(coords, labels):
            save_group_level_contrast_plots(layout, contrast_path, coord, label, config)
            save_max_mass_plot(layout, np_logp_max_mass_path, label, coord, config)

def roi_to_roi_group_analysis(layout, config):
    for connectivity_kind in config.get("connectivity_kinds"):
        # Retrieve connectivity type and other configuration parameters
        method = config.get("method")
        task = config.get("tasks")
        run = config.get("runs")
        session = config.get("sessions")
        space = config.get("spaces")
        analysis_type = config.get("analysis_type")  # Label for the analysis, e.g. "independent"

        entities = {
            "task": task,
            "space": space,
            "session": session,
            "run": run,
            "desc": connectivity_kind
        }

        design_matrix = None  # This will be necessary for regression analyses

        # Perform the appropriate group-level analysis
        if analysis_type == "independent":
            # Load group specifications from config
            # Todo: change terminology from "group" to "samples" when performing independent samples tests so that it is consistent with the terminology when doing a paired test.
            group1_subjects = config["group1_subjects"]
            group2_subjects = config["group2_subjects"]

            # Check each group has at least two subjects, otherwise no permutation testing is possible
            from connectomix.utils.tools import check_group_has_several_members
            check_group_has_several_members(group1_subjects)
            check_group_has_several_members(group2_subjects)

            # Retrieve the connectivity matrices for group 1 and group 2 using BIDSLayout
            from connectomix.utils.loaders import retrieve_connectivity_matrices_from_particpant_level
            group1_matrices = retrieve_connectivity_matrices_from_particpant_level(group1_subjects, layout, entities,
                                                                                   method)
            group2_matrices = retrieve_connectivity_matrices_from_particpant_level(group2_subjects, layout, entities,
                                                                                   method)

            # For independent tests we dontt need to keep track of subjects labels
            group1_matrices = list(group1_matrices.values())
            group2_matrices = list(group2_matrices.values())

            print(f"Group 1 contains {len(group1_matrices)} participants: {group1_subjects}")
            print(f"Group 2 contains {len(group2_matrices)} participants: {group2_subjects}")

            # Convert to 3D arrays: (subjects, nodes, nodes)
            group1_data = np.stack(group1_matrices, axis=0)
            group2_data = np.stack(group2_matrices, axis=0)
            group_data = [group1_data, group2_data]

            # Independent t-test between different subjects
            t_stats, p_values = ttest_ind(group1_data, group2_data, axis=0, equal_var=False)

        elif analysis_type == "paired":
            # Paired t-test within the same subjects
            from connectomix.utils.loaders import retrieve_connectivity_matrices_for_paired_samples
            paired_samples = retrieve_connectivity_matrices_for_paired_samples(layout, entities, config)

            # Get the two samples from paired_samples (with this we are certain that they are in the right order)
            sample1 = np.array(list(paired_samples.values()))[:, 0]
            sample2 = np.array(list(paired_samples.values()))[:, 1]
            group_data = [sample1, sample2]

            if len(sample1) != len(sample2):
                raise ValueError("Paired t-test requires an equal number of subjects in both samples.")

            t_stats, p_values = ttest_rel(sample1, sample2, axis=0)

            from connectomix.utils.tools import remove_pair_making_entity
            entities = remove_pair_making_entity(entities)

        elif analysis_type == "regression":
            subjects = config["subjects_to_regress"]
            group_data = retrieve_connectivity_matrices_from_particpant_level(subjects, layout, entities, method)
            group_data = list(group_data.values())
            from connectomix.utils.loaders import retrieve_info_from_participant_table
            design_matrix = retrieve_info_from_participant_table(layout, subjects, config["covariate"],
                                                                 config["confounds"])
            from connectomix.utils.stats import regression_analysis
            t_stats, p_values = regression_analysis(group_data, design_matrix)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Threshold 1: Uncorrected p-value
        uncorr_alpha = config["uncorrected_alpha"]
        uncorr_mask = p_values < uncorr_alpha

        # Threshold 2: FDR correction
        fdr_alpha = config["fdr_alpha"]
        fdr_mask = multipletests(p_values.flatten(), alpha=fdr_alpha, method="fdr_bh")[0].reshape(p_values.shape)

        # Threshold 3: Permutation-based threshold
        n_permutations = config["n_permutations"]
        if n_permutations < 5000:
            warnings.warn(
                f"Running permutation analysis with less than 5000 permutations (you chose {n_permutations}).")

        from connectomix.utils.stats import generate_permuted_null_distributions
        null_max_distribution, null_min_distribution = generate_permuted_null_distributions(group_data, config, layout,
                                                                                            entities, {
                                                                                                "observed_t_max": np.nanmax(
                                                                                                    t_stats),
                                                                                                "observed_t_min": np.nanmin(
                                                                                                    t_stats)},
                                                                                            design_matrix=design_matrix)

        # Compute thresholds at desired significance
        fwe_alpha = float(config["fwe_alpha"])
        t_max = np.percentile(null_max_distribution, (1 - fwe_alpha / 2) * 100)
        t_min = np.percentile(null_min_distribution, fwe_alpha / 2 * 100)
        print(
            f"Thresholds for max and min stat from null distribution estimated by permutations: {t_max} and {t_min} (n_perms = {n_permutations})")

        perm_mask = (t_stats > t_max) | (t_stats < t_min)

        # Save thresholds to a BIDS-compliant JSON file
        thresholds = {
            "uncorrected_alpha": uncorr_alpha,
            "fdr_alpha": fdr_alpha,
            "fwe_alpha": fwe_alpha,
            "fwe_permutations_results": {
                "max_t": t_max,
                "min_t": t_min,
                "n_permutations": n_permutations
            }
        }

        threshold_file = layout.derivatives["connectomix"].build_path({**entities,
                                                                       "analysis_label": config["analysis_label"],
                                                                       "method": config["method"]
                                                                       },
                                                                      path_patterns=[
                                                                          "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_thresholds.json"],
                                                                      validate=False)

        from connectomix.utils.makers import ensure_directory
        ensure_directory(threshold_file)
        with open(threshold_file, "w") as f:
            json.dump(thresholds, f, indent=4)

        # Get ROIs coords and labels for plotting purposes
        if method == "seeds":
            from connectomix.utils.loaders import load_seed_file
            coords, labels = load_seed_file(config["seeds_file"])

        elif method == "ica":
            extracted_regions_entities = entities.copy()
            extracted_regions_entities.pop("desc")
            extracted_regions_entities["suffix"] = "extractedregions"
            extracted_regions_entities["extension"] = ".nii.gz"
            extracted_regions_filename = layout.derivatives["connectomix"].get(**extracted_regions_entities)[0]
            coords = find_probabilistic_atlas_cut_coords(extracted_regions_filename)
            labels = None
        else:
            # Handle the case where method is an atlas
            from connectomix.utils.loaders import get_atlas_data
            _, labels, coords = get_atlas_data(method, get_cut_coords=True)

        # Create plots of the thresholded group connectivity matrices and connectomes
        from connectomix.utils.makers import generate_group_matrix_plots
        generate_group_matrix_plots(t_stats,
                                    uncorr_mask,
                                    fdr_mask,
                                    perm_mask,
                                    config,
                                    layout,
                                    entities,
                                    labels)

        from connectomix.utils.makers import generate_group_connectome_plots
        generate_group_connectome_plots(t_stats,
                                        uncorr_mask,
                                        fdr_mask,
                                        perm_mask,
                                        config,
                                        layout,
                                        entities,
                                        coords)

        # Refresh BIDS indexing of the derivatives to find data for the report
        output_dir = layout.derivatives["connectomix"].root
        try:
            layout.derivatives.pop("connectomix")
        except KeyError:
            layout.derivatives.pop(os.path.basename(output_dir))
        layout.add_derivatives(output_dir)

        # Generate report
        from connectomix.utils.makers import generate_group_analysis_report
        generate_group_analysis_report(layout, entities, config)
