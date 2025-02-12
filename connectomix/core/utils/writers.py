import json
import shutil

import numpy as np
import yaml
from bids import BIDSLayout
from matplotlib import pyplot as plt

from nilearn.plotting import plot_matrix, plot_connectome, plot_stat_map, plot_design_matrix
from pathlib import Path

from connectomix.core.utils.setup import setup_config
from connectomix.core.utils.tools import make_parent_dir
from connectomix.core.utils.bids import build_output_path, alpha_value_to_bids_valid_string

def write_design_matrix(layout, design_matrix, label, config):
    from connectomix.core.utils.loaders import load_entities_from_config
    entities = load_entities_from_config(config)
    entities.pop("subject")
    from connectomix.core.utils.bids import build_output_path
    design_matrix_plot_path = build_output_path(layout,
                                                entities,
                                                label,
                                                "group",
                                                config,
                                                contrast=None,
                                                suffix="designMatrix",
                                                extension=".svg")

    plot_design_matrix(design_matrix, output_file=design_matrix_plot_path)
    plt.close('all')


def write_permutation_dist(layout, permutation_dist, label, config):
    if permutation_dist is not None:
        from connectomix.core.utils.bids import build_output_path
        from connectomix.core.utils.loaders import load_entities_from_config
        entities = load_entities_from_config(config)
        entities.pop("subject")

        permutation_dist_path = build_output_path(layout,
                                                  entities,
                                                  label,
                                                  "group",
                                                  config,
                                                  suffix="permutationDistribution",
                                                  extension=".npy")

        histogram_path = build_output_path(layout,
                            entities,
                            label,
                            "group",
                            config,
                            suffix="permutationDistribution",
                            extension=".svg")

        plt.hist(permutation_dist, bins=10, edgecolor='black')

        plt.title("Permutation-generated null distribution")
        plt.xlabel('t-values')
        plt.ylabel('Frequency')

        plt.savefig(histogram_path)
        plt.close("all")
        np.save(permutation_dist_path, permutation_dist)


def write_contrast_scores(layout, contrast_scores, label, coord, config):
    from connectomix.core.utils.loaders import load_entities_from_config
    from connectomix.core.utils.bids import build_output_path

    entities = load_entities_from_config(config)
    entities.pop("subject")

    for score_type in contrast_scores.keys():
        if score_type == "p_values":
            suffix = "p"
        elif score_type == "z_values":
            suffix = "z"
        elif score_type == "t_values":
            suffix = "t"  # because that's only what we do

        score_path = build_output_path(layout,
                                          entities,
                                          label,
                                          "group",
                                          config,
                                          suffix=suffix)

        plot_path = build_output_path(layout,
                                      entities,
                                      label,
                                      "group",
                                      config,
                                      suffix=suffix,
                                      extension='.svg')

        if contrast_scores[score_type] is not None:
            if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
                contrast_scores[score_type].to_filename(score_path)
                write_map_at_cut_coords(contrast_scores[score_type], plot_path, coord, config)
            elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
                np.save(score_path, contrast_scores[score_type])
                write_matrix_plot(contrast_scores[score_type], plot_path, label=label)


def write_significant_data(layout, significant_data, label, coords, config):
    for thresholding_strategy in config["thresholding_strategies"]:
        alpha = float(config[f"{thresholding_strategy}_alpha"])
        from connectomix.core.utils.loaders import load_entities_from_config
        entities = load_entities_from_config(config)
        entities.pop("subject")

        significant_data_path = build_output_path(layout,
                                                      entities,
                                                      label,
                                                      "group",
                                                      config,
                                                      suffix=thresholding_strategy + alpha_value_to_bids_valid_string(alpha))
        matrix_plot_path = build_output_path(layout,
                                      entities,
                                      label,
                                      "group",
                                      config,
                                      suffix=thresholding_strategy + alpha_value_to_bids_valid_string(alpha),
                                      extension='.svg')

        connectome_plot_path = build_output_path(layout,
                                      entities,
                                      label,
                                      "group",
                                      config,
                                      suffix=thresholding_strategy + alpha_value_to_bids_valid_string(alpha) + "Connectome",
                                      extension='.svg')

        if significant_data[thresholding_strategy] is not None:
            if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
                significant_data[thresholding_strategy].to_filename(significant_data_path)
                # TODO: save glassbrain of significant_data[thresholding_strategy] to plot_path
            elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
                np.save(significant_data_path, significant_data[thresholding_strategy])
                write_matrix_plot(significant_data[thresholding_strategy], matrix_plot_path, label=label)
                write_connectome_plot(significant_data[thresholding_strategy], connectome_plot_path, coords)


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


def write_results(layout, results, label, coords, config):
    write_design_matrix(layout, results["design_matrix"], label, config)
    write_permutation_dist(layout, results["permutation_dist"], label, config)
    write_contrast_scores(layout, results["contrast_results"], label, coords, config)
    write_significant_data(layout, results["significant_data"], label, coords, config)
    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        write_cluster_tables(layout, results["cluster_tables"], label, config)


def write_cluster_tables(layout, cluster_tables, label, config):
    for thresholding_strategy in cluster_tables.keys():
        if cluster_tables[thresholding_strategy] is not None:
            alpha = float(config[f"{thresholding_strategy}_alpha"])
            from connectomix.core.utils.loaders import load_entities_from_config
            entities = load_entities_from_config(config)
            entities.pop("subject")

            table_path = build_output_path(layout,
                                           entities,
                                           label,
                                           "group",
                                           config,
                                           suffix=thresholding_strategy + alpha_value_to_bids_valid_string(alpha),
                                           extension=".csv")

            cluster_tables[thresholding_strategy].to_csv(table_path, sep=',')


def write_matrix_plot(matrix, path, label=None):
    plt.figure(figsize=(10, 10))
    plot_matrix(matrix, labels=label, colorbar=True)
    plt.savefig(path)
    plt.close('all')


def write_connectome_plot(matrix, path, coords):
    plt.figure(figsize=(10, 10))
    plot_connectome(matrix, node_coords=coords)
    plt.savefig(path)
    plt.close()


def write_map_at_cut_coords(map, path, coord, config):

    plot = plot_stat_map(map, cut_coords=coord)
    if config["method"] == "seedToVoxel":
        plot.add_markers(marker_coords=[coord],
                                      marker_color="k",
                                      marker_size=2 * config["radius"])
    plot.savefig(path)


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