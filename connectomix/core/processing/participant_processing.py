from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

import os
from nilearn.image import resample_to_img
from nilearn.connectome import ConnectivityMeasure
from bids import BIDSLayout
from pathlib import Path

from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker

import numpy as np

from connectomix.core.utils.tools import denoise


def post_fmriprep_preprocessing(layout, config):
    # Select all files needed for analysis
    from connectomix.core.utils.loaders import load_files_for_analysis
    from connectomix.core.utils.tools import resample_to_reference
    func_files, json_files, confound_files = load_files_for_analysis(layout, config)
    print(f"Found {len(func_files)} functional files:")
    [print(os.path.basename(fn)) for fn in func_files]

    # Resample all functional files to the reference image
    resampled_files = resample_to_reference(layout, func_files, config)
    print("All functional files resampled to match the reference image.")

    denoised_files = denoise(layout, resampled_files, confound_files, json_files, config)
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
        from connectomix.core.utils.loaders import load_seed_file
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
        elif method == "roiToRoi" and not config.get("canica", False):
            from connectomix.core.utils.loaders import load_atlas_data
            imgs, labels, _ = load_atlas_data(config["atlas"])
            imgs = [imgs]

        if config.get("canica", False):
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
    #         from connectomix.core.loaders import get_atlas_data
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


def create_and_fit_participant_glm(t_r, mask_img, func_file, timeseries):
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


def participant_roi_to_voxel(layout, func_file, timeseries_list, labels, config):
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
    from connectomix.core.utils.loaders import load_mask
    mask_img = load_mask(layout, entities)

    for timeseries, label in zip(timeseries_list.T, labels):
        roi_to_voxel_img = create_and_fit_participant_glm(config["t_r"], mask_img, func_file, timeseries.reshape(-1, 1))
        from connectomix.core.utils.bids import build_output_path
        from connectomix.core.utils.writers import write_roi_to_voxel_map
        roi_to_voxel_path = build_output_path(layout, entities, label, "participant", config)
        roi_to_voxel_img.to_filename(roi_to_voxel_path)
        write_roi_to_voxel_map(layout, roi_to_voxel_img, entities, label, config)


def participant_roi_to_roi(layout, func_file, timeseries_list, labels, config):
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
    from connectomix.core.utils.bids import build_output_path
    from connectomix.core.utils.writers import write_matrix_plot
    entities = layout.parse_file_entities(func_file)
    connectivity_measure = ConnectivityMeasure(kind=config["connectivity_kind"])
    conn_matrix = connectivity_measure.fit_transform([timeseries_list])[0]
    np.fill_diagonal(conn_matrix, 0)
    conn_matrix_path = build_output_path(layout, entities, None, "participant", config)
    np.save(conn_matrix_path, conn_matrix)
    write_matrix_plot(layout, conn_matrix, entities, labels, config)


def participant_analysis(layout, func_file, config):

    timeseries_list, labels = extract_timeseries(str(func_file), config)

    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        participant_roi_to_voxel(layout, func_file, timeseries_list, labels, config)
    elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
        participant_roi_to_roi(layout, func_file, timeseries_list, labels, config)
