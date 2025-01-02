import json
import shutil
from pathlib import Path

from matplotlib import pyplot as plt
from nilearn.plotting import plot_connectome, plot_matrix

# MAKERS
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
    Path(file_path).parents[0].mkdir(exist_ok=True, parents=True)


# Helper function to generate a dataset_description.json file
def create_dataset_description(output_dir):
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


def generate_group_analysis_report(layout, bids_entities, config):
    """
    Generates a group analysis report based on the method and connectivity kind.

    """
    from connectomix.utils.tools import apply_nonbids_filter

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

    ensure_directory(report_output_path)

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


# Helper function to create and save connectome plots for each thresholding strategy
def generate_group_connectome_plots(t_stats, uncorr_mask, fdr_mask, perm_mask, config, layout, entities, coords):
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


# Helper function to create and save matrix plots for each thresholding strategy
def generate_group_matrix_plots(t_stats, uncorr_mask, fdr_mask, perm_mask, config, layout, entities, labels=None):
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


# Function to copy config to path
def save_copy_of_config(config, path):
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
    # First make sure destination is valid
    ensure_directory(path)
    # If config is a str, assume it is a path and copy
    if isinstance(config, (str, Path)):
        shutil.copy(config, path)
    # Otherwise, it is a dict and must be dumped to path
    elif isinstance(config, dict):
        with open(path, "w") as fp:
            json.dump(config, fp, indent=4)
    return None
