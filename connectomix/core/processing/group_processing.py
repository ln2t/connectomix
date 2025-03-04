import pandas as pd
import warnings
import numpy as np
import statsmodels.api as sm
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix

from connectomix.core.utils.tools import find_labels_and_coords, get_cluster_tables
from connectomix.core.processing.stats import compute_significant_data, compute_z_from_t
from connectomix.core.utils.loaders import load_group_level_covariates
from connectomix.core.utils.writers import write_results

def make_group_input(layout, config, label=None):
    from connectomix.core.utils.loaders import load_entities_from_config
    from connectomix.core.utils.bids import apply_nonbids_filter, add_new_entities

    entities = load_entities_from_config(config)
    entities_with_non_bids_fields = add_new_entities(entities.copy(), config, label)
    extension = entities_with_non_bids_fields["extension"]

    second_level_input = pd.DataFrame(columns=["subject_label",
                                               "map_name",
                                               "effects_map_path"])
    files = layout.derivatives.get_pipeline("connectomix").get(return_type="filename",
                                                  extension=extension,
                                                  **entities)

    if entities_with_non_bids_fields["new_entity_val"] is not None:
        files = apply_nonbids_filter(entities_with_non_bids_fields["new_entity_key"],
                                     entities_with_non_bids_fields["new_entity_val"],
                                     files)
    files = apply_nonbids_filter("method",
                                 config["method"],
                                 files)

    # TODO: for paired analysis, compute difference of maps and save result to folder.
    if config["paired_tests"]:
        raise ValueError("Paired analysis not yet supported")

    if len(files) == 0:
        raise ValueError("No participant-level file found.")
    else:
        print(f"Found {len(files)} files at participant-level.")

    for file in files:
        file_entities = layout.parse_file_entities(file)
        second_level_input.loc[len(second_level_input)] = [f"sub-{file_entities['subject']}", "effectOfSeedOrRoi", file]

    return second_level_input


def make_group_design_matrix(layout, second_level_input, config):
    subjects_label = list(second_level_input["subject_label"])
    confounds = load_group_level_covariates(layout, subjects_label, config)

    design_matrix = make_second_level_design_matrix(subjects_label, confounds=confounds)

    if "group" in config["covariates"] and config["add_intercept"]:
        warnings.warn("Adding an intercept when including group factor in the design matrix is always"
                      "producing a singular matrix. Are you sure this is what you want to do?")

    if not config["add_intercept"]:
        design_matrix.drop(columns=["intercept"], inplace=True)

    return design_matrix


def compute_group_contrast(glm, config):
    print(f"Computing contrast label named \'{config['contrast']}\'")

    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        contrast_dict = glm.compute_contrast(config["contrast"],
                             first_level_contrast="effectOfSeedOrRoi",
                             output_type="all")
        t_values = contrast_dict["stat"]
        z_values = contrast_dict["z_score"]
        p_values = contrast_dict["p_value"]

    elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
        contrast_vector = expression_to_contrast_vector(config["contrast"], glm["design_matrix"].columns)

        vect_z = []
        vect_t = []
        vect_p = []
        for result in glm["result"]:
            if result:
                t_val = result.t_test(contrast_vector).tvalue[0][0]
                z_val = compute_z_from_t(t_val, result.df_resid)
                vect_z.append(z_val)
                vect_t.append(t_val)
                vect_p.append(result.t_test(contrast_vector).pvalue)
            else:
                vect_z.append(None)
                vect_t.append(0)
                vect_p.append(1)

        z_values = vec_to_sym_matrix(np.array(vect_z))
        t_values = vec_to_sym_matrix(np.array(vect_t))
        p_values = vec_to_sym_matrix(np.array(vect_p))

    return {"z_values": z_values, "t_values": t_values, "p_values": p_values}


def create_and_fit_group_glm(second_level_input, design_matrix, config):

    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        glm = SecondLevelModel(smoothing_fwhm=config["smoothing"])
        glm.fit(second_level_input, design_matrix=design_matrix)
    elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
        data = [sym_matrix_to_vec(np.load(file)) for file in second_level_input["effects_map_path"]]
        data = np.vstack(data)

        glm = {"data": data, "design_matrix": design_matrix, "model": [], "result": []}

        for i in range(data.shape[1]):
            y_i = data[:, i]
            model = sm.GLM(y_i, design_matrix, family=sm.families.Gaussian())
            glm["model"].append(model)

            if np.all(y_i == 0):
                result = None
            else:
                result = model.fit()

            glm["result"].append(result)
    else:
        glm = None
    return glm


def run_single_group_analysis(layout, label, config):
    results = {}
    second_level_input = make_group_input(layout, config, label)
    design_matrix = make_group_design_matrix(layout, second_level_input, config)
    glm = create_and_fit_group_glm(second_level_input, design_matrix, config)
    contrast_results = compute_group_contrast(glm, config)
    significant_data, permutation_dist = compute_significant_data(contrast_results, glm, config)
    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        cluster_tables = get_cluster_tables(significant_data, config)
    else:
        cluster_tables = None

    results["design_matrix"] = design_matrix
    results["permutation_dist"] = permutation_dist
    results["contrast_results"] = contrast_results
    results["significant_data"] = significant_data
    results["cluster_tables"] = cluster_tables

    return results


def group_analysis(layout, config):
    labels, coords = find_labels_and_coords(config)
    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        for (label, coord) in zip(labels, coords):
            print(f"Analysis for seed/roi {label}")
            results = run_single_group_analysis(layout, label, config)
            write_results(layout, results, label, coord, config)
    elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
        results = run_single_group_analysis(layout, None, config)
        write_results(layout, results, labels, coords, config)

    # Save plots
    # if config["method"] == "seedToVoxel":
    #     # TODO: make this compatible with roiToVoxel

    # for coord, label in zip(coords, labels):
    #     write_group_level_contrast_plots(layout, contrast_path, coord, label, config)
    #     from connectomix.core.utils.writers import write_max_mass_plot
    #     write_max_mass_plot(layout, non_parametric_neg_pvals_path, label, coord, config)
