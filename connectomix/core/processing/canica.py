from nilearn.decomposition import CanICA
from nilearn.regions import RegionExtractor

import os
import json
from bids import BIDSLayout

from connectomix.core.utils.tools import denoise, custom_print


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

    from connectomix.core.utils.loaders import load_entities_from_config
    entities = load_entities_from_config(config)
    entities.pop('subject')

    # Build path to save canICA components
    # TODO: update these path-building steps using build_output_path
    canica_filename = layout.derivatives.get_pipeline("connectomix").build_path(entities,
                                                                   path_patterns=[
                                                                       "canica/[ses-{session}_][run-{run}_]task-{task}_space-{space}_canicacomponents.nii.gz"],
                                                                   validate=False)
    canica_sidecar = layout.derivatives.get_pipeline("connectomix").build_path(entities,
                                                                  path_patterns=[
                                                                      "canica/[ses-{session}_][run-{run}_]task-{task}_space-{space}_canicacomponents.json"],
                                                                  validate=False)
    extracted_regions_filename = layout.derivatives.get_pipeline("connectomix").build_path(entities,
                                                                              path_patterns=[
                                                                                  "canica/[ses-{session}_][run-{run}_]task-{task}_space-{space}_extractedregions.nii.gz"],
                                                                              validate=False)
    extracted_regions_sidecar = layout.derivatives.get_pipeline("connectomix").build_path(entities,
                                                                             path_patterns=[
                                                                                 "canica/[ses-{session}_][run-{run}_]task-{task}_space-{space}_extractedregions.json"],
                                                                             validate=False)

    from connectomix.core.utils.tools import make_parent_dir
    make_parent_dir(canica_filename)
    make_parent_dir(canica_sidecar)
    make_parent_dir(extracted_regions_filename)
    make_parent_dir(extracted_regions_sidecar)

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
        custom_print(f"Saving canica components image to {canica_filename}")
        canica.components_img_.to_filename(canica_filename)
    else:
        custom_print(f"ICA component file {os.path.basename(canica_filename)} already exist, skipping computation.")

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

    custom_print(f"Number of ICA-based components extracted: {extractor.regions_img_.shape[-1]}")

    custom_print(f"Saving extracted ROIs to {extracted_regions_filename}")
    extractor.regions_img_.to_filename(extracted_regions_filename)

    config['components'] = canica_filename
    config['extractor'] = extractor

    return config
