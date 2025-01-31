import json
import shutil
import warnings

import numpy as np
import yaml
from bids import BIDSLayout
from matplotlib import pyplot as plt

from nilearn.glm import threshold_stats_img
from nilearn.image import math_img, binarize_img
from nilearn.masking import apply_mask, unmask
from nilearn.plotting import plot_matrix, plot_connectome, plot_glass_brain, plot_stat_map
from pathlib import Path

from connectomix.core.utils.setup import setup_config
from connectomix.core.utils.tools import img_is_not_empty, \
    make_parent_dir
from connectomix.core.utils.bids import build_output_path, alpha_value_to_bids_valid_string


def write_significant_contrast_maps(layout, contrast_path, non_parametric_neg_pvals_path, label, config):
    for significance_level in ["uncorrected", "fdr", "fwe"]:
        alpha = float(config[f"{significance_level}_alpha"])
        from connectomix.core.utils.loaders import load_entities_from_config
        entities = load_entities_from_config(config)
        entities.pop("subject")

        thresholded_contrast_path = build_output_path(layout,
                                                      entities,
                                                      label,
                                                      "group",
                                                      config,
                                                      suffix=significance_level + alpha_value_to_bids_valid_string(alpha))

        if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
            thresholded_data = None
            match significance_level:
                case "uncorrected":
                    thresholded_data, _ = threshold_stats_img(contrast_path,
                                                             alpha=alpha,
                                                             height_control=None,
                                                             two_sided=True)
                case "fdr":
                    thresholded_data, _ = threshold_stats_img(contrast_path,
                                                             alpha=alpha,
                                                             height_control="fdr",
                                                             two_sided=True)
                case "fwe":
                    mask = math_img(f"img >= -np.log10({alpha})", img=non_parametric_neg_pvals_path)
                    mask = binarize_img(mask)

                    if img_is_not_empty(mask):
                        masked_data = apply_mask(contrast_path, mask)
                        thresholded_data = unmask(masked_data, mask)
                    else:
                        warnings.warn(
                            f"For map {contrast_path}, no voxel survives FWE thresholding at alpha level {alpha}.")
                        thresholded_data = None

            thresholded_data and thresholded_data.to_filename(thresholded_contrast_path)

        elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
            raise ValueError("This is work in progress")


def write_default_config_file(bids_dir, derivatives, level):
    """
    Create default configuration file in YAML format for default parameters, at group level.
    Configuration file is saved at 'derivatives/config/default_group_level_config.yaml'.

    Parameters
    ----------
    bids_dir : str or Path
        Path to BIDS directory.
    derivatives : dict
    level : str

    Returns
    -------
    None.

    """
    from connectomix.core.utils.tools import make_parent_dir

    output_dir = Path(derivatives["connectomix"])

    if level == "participant":
        # Create derivative directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create the dataset_description.json file
        write_dataset_description(output_dir)

    # Create a BIDSLayout to parse the BIDS dataset
    layout = BIDSLayout(bids_dir, derivatives=list(derivatives.values()))

    # Print some stuff for the primate using this function
    print("Generating default configuration file for default parameters, please wait while the dataset is explored...")

    config = setup_config(layout, {}, level)

    # Build filenames for each output
    yaml_file = Path(output_dir) / "config" / f"default_{level}_level_config.yaml"
    make_parent_dir(yaml_file)

    # Save config to yaml
    with open(yaml_file, 'w') as yaml_out:
        yaml.dump(config, yaml_out, default_flow_style=False)

    print(f"Default YAML configuration file saved at {yaml_file}. Go to github.com/ln2t/connectomix for more details.")


def write_copy_of_config(layout, config):
    """
    Save a copy of config to path, for reproducibility.

    Parameters
    ----------
    config : dict or str or Path
        Configuration dict or path to loaded configuration file.
    path : str or Path
        Path to the desired location to dump the config.

    Returns
    -------
    None.

    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save a copy of the config file to the config directory
    path = Path(
        layout.derivatives["connectomix"].root) / "config" / "backups" / f"config_{timestamp}.json"

    # First make sure destination is valid
    make_parent_dir(path)
    # If config is a str, assume it is a path and copy
    if isinstance(config, (str, Path)):
        shutil.copy(config, path)
    # Otherwise, it is a dict and must be dumped to path
    elif isinstance(config, dict):
        with open(path, "w") as fp:
            json.dump(config, fp, indent=4)

    print(f"Configuration file saved to {path}")
    return None


def write_group_matrix_plots(t_stats, uncorr_mask, fdr_mask, perm_mask, config, layout, entities, labels=None):
    """
    Tool to generate thresholded connectivity matrix plots.

    Parameters
    ----------
    t_stats : numpy.array
        The unthresholded t-score matrix.
    uncorr_mask : numpy.array
        Mask defining the supra-threshold connections for the uncorrected strategy.
    fdr_mask : numpy.array
        Mask defining the supra-threshold connections for the fdr strategy.
    perm_mask : numpy.array
        Mask defining the supra-threshold connections for the fwe strategy.
    config : dict
        Configuration.
    layout : BIDSLayout
        Usual BIDS class for the dataset.
    entities : dict
        Entities to build output paths for the figures.
    labels : list, optional
        Labels for the axis of the plots (length is equal to the number of rows of the connectivity matrix). The default is None.

    Returns
    -------
    None.

    """

    fn_uncorr = layout.derivatives["connectomix"].build_path({**entities,
                                                              "analysis_label": config["analysis_label"],
                                                              "method": config["method"],
                                                              "alpha": str(config["uncorrected_alpha"]).replace('.',
                                                                                                                'dot')
                                                              },
                                                             path_patterns=[
                                                                 "group/{analysis_label}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_alpha-{alpha}_uncorrmatrix.svg"],
                                                             validate=False)

    fn_fdr = layout.derivatives["connectomix"].build_path({**entities,
                                                           "analysis_label": config["analysis_label"],
                                                           "method": config["method"],
                                                           "alpha": str(config["fdr_alpha"]).replace('.', 'dot')
                                                           },
                                                          path_patterns=[
                                                              "group/{analysis_label}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_alpha-{alpha}_fdrmatrix.svg"],
                                                          validate=False)

    fn_fwe = layout.derivatives["connectomix"].build_path({**entities,
                                                           "analysis_label": config["analysis_label"],
                                                           "method": config["method"],
                                                           "alpha": str(config["fwe_alpha"]).replace('.', 'dot')
                                                           },
                                                          path_patterns=[
                                                              "group/{analysis_label}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_alpha-{alpha}_fwematrix.svg"],
                                                          validate=False)

    uncorr_percentage = 100 * float(config.get("uncorrected_alpha"))
    uncorr_percentage = str(uncorr_percentage)
    plt.figure(figsize=(10, 10))
    plot_matrix(t_stats * uncorr_mask, labels=labels, colorbar=True,
                title=f"Uncorrected Threshold ({uncorr_percentage}%)")
    plt.savefig(fn_uncorr)
    plt.close()

    fdr_percentage = 100 * float(config.get("fdr_alpha"))
    fdr_percentage = str(fdr_percentage)
    plt.figure(figsize=(10, 10))
    plot_matrix(t_stats * fdr_mask, labels=labels, colorbar=True, title=f"FDR Threshold ({fdr_percentage}%)")
    plt.savefig(fn_fdr)
    plt.close()

    fwe_percentage = 100 * float(config.get("fwe_alpha"))
    fwe_percentage = str(fwe_percentage)
    n_permutations = config.get("n_permutations")
    n_permutations = str(n_permutations)
    plt.figure(figsize=(10, 10))
    plot_matrix(t_stats * perm_mask, labels=labels, colorbar=True,
                title=f"Permutation-Based Threshold ({fwe_percentage}% and {n_permutations} permutations)")
    plt.savefig(fn_fwe)
    plt.close()


def write_group_connectome_plots(t_stats, uncorr_mask, fdr_mask, perm_mask, config, layout, entities, coords):
    """
    Same as generate_group_matrix_plots, but for the connectomes (i.e. glass-brains with connections represented as solid lines between nodes).

    Returns
    -------
    None.

    """

    fn_uncorr = layout.derivatives["connectomix"].build_path({**entities,
                                                              "analysis_label": config["analysis_label"],
                                                              "method": config["method"],
                                                              "alpha": str(config["uncorrected_alpha"]).replace('.',
                                                                                                                'dot')
                                                              },
                                                             path_patterns=[
                                                                 "group/{analysis_label}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_alpha-{alpha}_uncorrconnectome.svg"],
                                                             validate=False)

    fn_fdr = layout.derivatives["connectomix"].build_path({**entities,
                                                           "analysis_label": config["analysis_label"],
                                                           "method": config["method"],
                                                           "alpha": str(config["fdr_alpha"]).replace('.', 'dot')
                                                           },
                                                          path_patterns=[
                                                              "group/{analysis_label}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_alpha-{alpha}_fdrconnectome.svg"],
                                                          validate=False)

    fn_fwe = layout.derivatives["connectomix"].build_path({**entities,
                                                           "analysis_label": config["analysis_label"],
                                                           "method": config["method"],
                                                           "alpha": str(config["fwe_alpha"]).replace('.', 'dot')
                                                           },
                                                          path_patterns=[
                                                              "group/{analysis_label}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_alpha-{alpha}_fweconnectome.svg"],
                                                          validate=False)

    uncorr_percentage = 100 * float(config.get("uncorrected_alpha"))
    uncorr_percentage = str(uncorr_percentage)
    plt.figure(figsize=(10, 10))
    plot_connectome(t_stats * uncorr_mask, node_coords=coords, title=f"Uncorrected Threshold ({uncorr_percentage}%)")
    plt.savefig(fn_uncorr)
    plt.close()

    fdr_percentage = 100 * float(config.get("fdr_alpha"))
    fdr_percentage = str(fdr_percentage)
    plt.figure(figsize=(10, 10))
    plot_connectome(t_stats * fdr_mask, node_coords=coords, title=f"FDR Threshold ({fdr_percentage}%)")
    plt.savefig(fn_fdr)
    plt.close()

    fwe_percentage = 100 * float(config.get("fwe_alpha"))
    fwe_percentage = str(fwe_percentage)
    n_permutations = config.get("n_permutations")
    n_permutations = str(n_permutations)
    plt.figure(figsize=(10, 10))
    plot_connectome(t_stats * perm_mask, node_coords=coords,
                    title=f"Permutation-Based Threshold ({fwe_percentage}% and {n_permutations} permutations)")
    plt.savefig(fn_fwe)
    plt.close()


def write_group_analysis_report(layout, bids_entities, config):
    """
    Generates a group analysis report based on the method and connectivity kind.

    """
    from connectomix.core.utils.bids import apply_nonbids_filter

    method = config.get("method")
    analysis_label = config.get('analysis_label')
    connectivity_kinds = config.get("connectivity_kinds")
    analysis_type = config.get("analysis_type")

    entities = dict(**bids_entities,
                    method=method,
                    analysis=analysis_label)

    report_output_path = layout.derivatives["connectomix"].build_path(entities,
                                                                      path_patterns=[
                                                                          'group/{analysis}/group_[ses-{session}_][run-{run}_]task-{task}_space-{space}_method-{method}_desc-{desc}_analysis-{analysis}_report.html'],
                                                                      validate=False)

    make_parent_dir(report_output_path)

    suffixes = ['uncorrmatrix', 'uncorrconnectome', 'fdrmatrix', 'fdrconnectome', 'fwematrix', 'fweconnectome']

    with open(report_output_path, 'w') as report_file:
        # Write the title of the report
        report_file.write(f"<h1>Group analysis Report for Method: {method}</h1>\n")
        report_file.write(f"<h2>Connectivity Kind: {connectivity_kinds}</h2>\n")
        report_file.write(f"<h3>Analysis type: {analysis_type}, analysis label {config.get('analysis_label')}</h3>\n")
        if analysis_type == 'independent':
            report_file.write(
                f"<h3>Subjects: {config.get('group1_subjects')} versus {config.get('group2_subjects')}</h3>\n")
        elif analysis_type == 'regression':
            report_file.write(f"<h3>Subjects: {config.get('subjects_to_regress')}</h3>\n")
            report_file.write(f"<h3>Covariate: {config.get('covariate')}</h3>\n")
            if config.get('analysis_options')['confounds']:
                report_file.write(f"<h3>Confounds: {config.get('confounds')}</h3>\n")
        for suffix in suffixes:
            figure_files = layout.derivatives["connectomix"].get(**bids_entities,
                                                                 suffix=suffix,
                                                                 extension='.svg',
                                                                 return_type='filename')
            figure_files = apply_nonbids_filter('method', method, figure_files)

            figure_files = apply_nonbids_filter('analysis', analysis_label, figure_files)

            if suffix in ['uncorrmatrix', 'uncorrconnectome']:
                alpha = str(config["uncorrected_alpha"]).replace('.', 'dot')
            if suffix in ['fdrmatrix', 'fdrconnectome']:
                alpha = str(config["fdr_alpha"]).replace('.', 'dot')
            if suffix in ['fwematrix', 'fweconnectome']:
                alpha = str(config["fwe_alpha"]).replace('.', 'dot')

            figure_files = apply_nonbids_filter('alpha', alpha, figure_files)
            if len(figure_files) < 1:
                raise ValueError("Not enough figure files found, maybe this is a bug?")
            elif len(figure_files) >= 2:
                raise ValueError(
                    "f{Too many files found in the group-level outputs, are you sure you aren't mixing up analyses? Use different labels if need be!'}")
            else:
                for figure_file in figure_files:
                    report_file.write(f'<img src="{figure_file}" width="800">\n')

        print(
            "Group analysis report saved. To open, you may try to type the following command (with some minor modification if using Docker)")
        print(f"open {report_output_path}")


def write_max_mass_plot(layout, np_logp_max_mass_path, label, coords, config):
    from connectomix.core.utils.loaders import load_entities_from_config
    entities = load_entities_from_config(config)
    entities.pop("subject")

    np_logp_max_mass_plot_path = build_output_path(layout,
                                                   entities,
                                                   label,
                                                   "group",
                                                   config,
                                                   suffix="logpMaxMass",
                                                   extension=".svg")

    # entities["seed"] = label
    # np_logp_max_mass_plot_path = layout.derivatives["connectomix"].build_path({**entities,
    #                                                                            "analysis_label": config[
    #                                                                                "analysis_label"],
    #                                                                            "method": config["method"],
    #                                                                            "seed": label
    #                                                                            },
    #                                                                           path_patterns=[
    #                                                                               "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_seed-{seed}_analysis-{analysis_label}_logpMaxMass.svg"],
    #                                                                           validate=False)

    # from connectomix.core.makers import ensure_directory
    # ensure_directory(np_logp_max_mass_plot_path)

    plot_glass_brain(
        np_logp_max_mass_path,
        colorbar=True,
        cmap="autumn",
        vmax=2.69,  # this is hardcoded but that's not a problem as it is only for plots
        display_mode="z",
        plot_abs=False,
        cut_coords=coords,
        threshold=-np.log10(float(config["fwe_alpha"]))).savefig(np_logp_max_mass_plot_path)


def write_matrix_plot(layout, conn_matrix, entities, labels, config):
    conn_matrix_plot_path = build_output_path(layout, entities, None, "participant", config, extension=".svg")
    plt.figure(figsize=(10, 10))
    plot_matrix(conn_matrix, labels=labels, colorbar=True)
    plt.savefig(conn_matrix_plot_path)
    plt.close()


def write_roi_to_voxel_map(layout, roi_to_voxel_img, entities, label, config):
    roi_to_voxel_plot_path = build_output_path(layout, entities, label, "participant", config, extension=".svg")

    if config.get("seeds_file", False):
        from connectomix.core.utils.loaders import load_seed_file
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


def write_dataset_description(output_dir):
    """
    Create the dataset_description.json file, mandatory if outputs are to be indexed by BIDSLayout.

    Parameters
    ----------
    output_dir : str or Path
        Path to the output dir where to save the description.

    Returns
    -------
    None.

    """
    from connectomix.version import __version__
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


def write_group_level_contrast_plots(layout, contrast_path, coord, label, config):
    # Create plot of contrast map and save
    from connectomix.core.utils.loaders import load_entities_from_config
    entities = load_entities_from_config(config)
    entities.pop("subject")

    from connectomix.core.utils.bids import build_output_path
    contrast_plot_path = build_output_path(layout,
                                           entities,
                                           label,
                                           "group",
                                           config,
                                           suffix="zScore",
                                           extension=".svg")

    # entities["seed"] = label
    # contrast_plot_path = layout.derivatives["connectomix"].build_path({**entities,
    #                                                                    "analysis_label": config["analysis_label"],
    #                                                                    "method": config["method"],
    #                                                                    "seed": label
    #                                                                    },
    #                                                                   path_patterns=[
    #                                                                       "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_seed-{seed}_analysis-{analysis_label}_zScore.svg"],
    #                                                                   validate=False)
    # from connectomix.core.makers import ensure_directory
    # ensure_directory(contrast_plot_path)

    contrast_plot = plot_stat_map(contrast_path,
                                  threshold=3.0,
                                  title=f"roi-to-voxel contrast for seed {label} (coords {coord})",
                                  cut_coords=coord)
    contrast_plot.add_markers(marker_coords=[coord],
                              marker_color="k",
                              marker_size=2 * config["radius"])

    contrast_plot.savefig(contrast_plot_path)
