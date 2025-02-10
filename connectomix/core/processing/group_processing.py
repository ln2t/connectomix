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
    files = layout.derivatives["connectomix"].get(return_type="filename",
                                                  extension=extension,
                                                  **entities)

    files = apply_nonbids_filter(entities_with_non_bids_fields["new_entity_key"],
                                 entities_with_non_bids_fields["new_entity_val"],
                                 files)
    files = apply_nonbids_filter("method",
                                 config["method"],
                                 files)

    # TODO: for paired analysis, compute difference of maps and save result to folder.
    if config["paired_tests"]:
        raise ValueError("Paired analysis not yet supported")

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


# def group_roi_to_roi(layout, config):
#
#     second_level_input = make_group_input(layout, None, config)
#     design_matrix = make_group_design_matrix(layout, second_level_input, None, config)
#     glm = create_and_fit_group_glm(second_level_input, design_matrix, config)
#     contrast_path = compute_group_contrast(glm, None, config)
#     # compute_non_parametric_stats()
#     # save_results()
#
#     raise ValueError("This is under construction.")
#
#     for connectivity_kind in config.get("connectivity_kinds"):
#         # Retrieve connectivity type and other configuration parameters
#         method = config.get("method")
#         task = config.get("tasks")
#         run = config.get("runs")
#         session = config.get("sessions")
#         space = config.get("spaces")
#         analysis_type = config.get("analysis_type")  # Label for the analysis, e.g. "independent"
#
#         entities = {
#             "task": task,
#             "space": space,
#             "session": session,
#             "run": run,
#             "desc": connectivity_kind
#         }
#
#         design_matrix = None  # This will be necessary for regression analyses
#
#         # Perform the appropriate group-level analysis
#         if analysis_type == "independent":
#             # Load group specifications from config
#             # Todo: change terminology from "group" to "samples" when performing independent samples tests so that it is consistent with the terminology when doing a paired test.
#             group1_subjects = config["group1_subjects"]
#             group2_subjects = config["group2_subjects"]
#
#             # Check each group has at least two subjects, otherwise no permutation testing is possible
#             from connectomix.core.utils.tools import check_group_has_several_members
#             check_group_has_several_members(group1_subjects)
#             check_group_has_several_members(group2_subjects)
#
#             # Retrieve the connectivity matrices for group 1 and group 2 using BIDSLayout
#             from connectomix.core.utils.loaders import load_connectivity_matrices_from_particpant_level
#             group1_matrices = load_connectivity_matrices_from_particpant_level(group1_subjects, layout, entities,
#                                                                                method)
#             group2_matrices = load_connectivity_matrices_from_particpant_level(group2_subjects, layout, entities,
#                                                                                method)
#
#             # For independent tests we dontt need to keep track of subjects labels
#             group1_matrices = list(group1_matrices.values())
#             group2_matrices = list(group2_matrices.values())
#
#             print(f"Group 1 contains {len(group1_matrices)} participants: {group1_subjects}")
#             print(f"Group 2 contains {len(group2_matrices)} participants: {group2_subjects}")
#
#             # Convert to 3D arrays: (subjects, nodes, nodes)
#             group1_data = np.stack(group1_matrices, axis=0)
#             group2_data = np.stack(group2_matrices, axis=0)
#             group_data = [group1_data, group2_data]
#
#             # Independent t-test between different subjects
#             t_stats, p_values = ttest_ind(group1_data, group2_data, axis=0, equal_var=False)
#
#         elif analysis_type == "paired":
#             # Paired t-test within the same subjects
#             from connectomix.core.utils.loaders import load_connectivity_matrices_for_paired_samples
#             paired_samples = load_connectivity_matrices_for_paired_samples(layout, entities, config)
#
#             # Get the two samples from paired_samples (with this we are certain that they are in the right order)
#             sample1 = np.array(list(paired_samples.values()))[:, 0]
#             sample2 = np.array(list(paired_samples.values()))[:, 1]
#             group_data = [sample1, sample2]
#
#             if len(sample1) != len(sample2):
#                 raise ValueError("Paired t-test requires an equal number of subjects in both samples.")
#
#             t_stats, p_values = ttest_rel(sample1, sample2, axis=0)
#
#             from connectomix.core.utils.bids import remove_pair_making_entity
#             entities = remove_pair_making_entity(entities)
#
#         elif analysis_type == "regression":
#             subjects = config["subjects_to_regress"]
#             group_data = load_connectivity_matrices_from_particpant_level(subjects, layout, entities, method)
#             group_data = list(group_data.values())
#             from connectomix.core.utils.loaders import load_info_from_participant_table
#             design_matrix = load_info_from_participant_table(layout, subjects, config["covariate"],
#                                                              config["confounds"])
#             from connectomix.core.processing.stats import regression_analysis
#             t_stats, p_values = regression_analysis(group_data, design_matrix)
#         else:
#             raise ValueError(f"Unknown analysis type: {analysis_type}")
#
#         # Threshold 1: Uncorrected p-value
#         uncorr_alpha = config["uncorrected_alpha"]
#         uncorr_mask = p_values < uncorr_alpha
#
#         # Threshold 2: FDR correction
#         fdr_alpha = config["fdr_alpha"]
#         fdr_mask = multipletests(p_values.flatten(), alpha=fdr_alpha, method="fdr_bh")[0].reshape(p_values.shape)
#
#         # Threshold 3: Permutation-based threshold
#         n_permutations = config["n_permutations"]
#         if n_permutations < 5000:
#             warnings.warn(
#                 f"Running permutation analysis with less than 5000 permutations (you chose {n_permutations}).")
#
#         from connectomix.core.processing.stats import generate_permuted_null_distributions
#         null_max_distribution, null_min_distribution = generate_permuted_null_distributions(group_data, config, layout,
#                                                                                             entities, {
#                                                                                                 "observed_t_max": np.nanmax(
#                                                                                                     t_stats),
#                                                                                                 "observed_t_min": np.nanmin(
#                                                                                                     t_stats)},
#                                                                                             design_matrix=design_matrix)
#
#         # Compute thresholds at desired significance
#         fwe_alpha = float(config["fwe_alpha"])
#         t_max = np.percentile(null_max_distribution, (1 - fwe_alpha / 2) * 100)
#         t_min = np.percentile(null_min_distribution, fwe_alpha / 2 * 100)
#         print(
#             f"Thresholds for max and min stat from null distribution estimated by permutations: {t_max} and {t_min} (n_perms = {n_permutations})")
#
#         perm_mask = (t_stats > t_max) | (t_stats < t_min)
#
#         # Save thresholds to a BIDS-compliant JSON file
#         thresholds = {
#             "uncorrected_alpha": uncorr_alpha,
#             "fdr_alpha": fdr_alpha,
#             "fwe_alpha": fwe_alpha,
#             "fwe_permutations_results": {
#                 "max_t": t_max,
#                 "min_t": t_min,
#                 "n_permutations": n_permutations
#             }
#         }
#
#         threshold_file = layout.derivatives["connectomix"].build_path({**entities,
#                                                                        "analysis_label": config["analysis_label"],
#                                                                        "method": config["method"]
#                                                                        },
#                                                                       path_patterns=[
#                                                                           "group/{analysis_label}/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_thresholds.json"],
#                                                                       validate=False)
#
#         from connectomix.core.utils.tools import make_parent_dir
#         make_parent_dir(threshold_file)
#         with open(threshold_file, "w") as f:
#             json.dump(thresholds, f, indent=4)
#
#         # Get ROIs coords and labels for plotting purposes
#         if method == "data":
#             from connectomix.core.utils.loaders import load_seed_file
#             coords, labels = load_seed_file(config["seeds_file"])
#
#         elif method == "ica":
#             extracted_regions_entities = entities.copy()
#             extracted_regions_entities.pop("desc")
#             extracted_regions_entities["suffix"] = "extractedregions"
#             extracted_regions_entities["extension"] = ".nii.gz"
#             extracted_regions_filename = layout.derivatives["connectomix"].get(**extracted_regions_entities)[0]
#             coords = find_probabilistic_atlas_cut_coords(extracted_regions_filename)
#             labels = None
#         else:
#             # Handle the case where method is an atlas
#             from connectomix.core.utils.loaders import load_atlas_data
#             _, labels, coords = load_atlas_data(method, get_cut_coords=True)
#
#         # Create plots of the thresholded group connectivity matrices and connectomes
#         from connectomix.core.utils.writers import write_group_matrix_plots
#         write_group_matrix_plots(t_stats,
#                                  uncorr_mask,
#                                  fdr_mask,
#                                  perm_mask,
#                                  config,
#                                  layout,
#                                  entities,
#                                  labels)
#
#         from connectomix.core.utils.writers import write_group_connectome_plots
#         write_group_connectome_plots(t_stats,
#                                      uncorr_mask,
#                                      fdr_mask,
#                                      perm_mask,
#                                      config,
#                                      layout,
#                                      entities,
#                                      coords)
#
#         # Refresh BIDS indexing of the derivatives to find data for the report
#         output_dir = layout.derivatives["connectomix"].root
#         try:
#             layout.derivatives.pop("connectomix")
#         except KeyError:
#             layout.derivatives.pop(os.path.basename(output_dir))
#         layout.add_derivatives(output_dir)
#
#         # Generate report
#         from connectomix.core.utils.writers import write_group_analysis_report
#         write_group_analysis_report(layout, entities, config)


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
