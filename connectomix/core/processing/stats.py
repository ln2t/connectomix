import warnings
import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from nilearn.glm.second_level import non_parametric_inference
from nilearn.glm import threshold_stats_img, fdr_threshold
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.mass_univariate import permuted_ols
from nilearn.image import math_img, binarize_img
from nilearn.masking import apply_mask, unmask

from scipy.stats import permutation_test
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from connectomix.core.utils.tools import img_is_not_empty


def compute_z_from_t(t_score, degree_of_freedom):
    from scipy.stats import t, norm
    cdf_t = t.cdf(t_score, degree_of_freedom)
    return norm.ppf(cdf_t)


# def generate_permuted_null_distributions(group_data, config, layout, entities, observed_stats, design_matrix=None):
#     """
#     Perform a two-sided permutation test to determine positive and negative thresholds separately.
#     Returns separate maximum and minimum thresholds for positive and negative t-values.
#     """
#     # Todo: create plots of null distributions for the report
#     # Extract values from config
#     n_permutations = config.get("n_permutations")
#     analysis_type = config.get("analysis_type")
#
#     # Load pre-existing permuted data, if any
#     perm_files = layout.derivatives["connectomix"].get(desc=config["connectivity_kinds"],
#                                                        extension=".npy",
#                                                        suffix="permutations",
#                                                        return_type="filename")
#     from connectomix.core.utils.bids import apply_nonbids_filter
#     perm_files = apply_nonbids_filter("analysis",
#                                       config["analysis_label"],
#                                       perm_files)
#     perm_files = apply_nonbids_filter("method",
#                                       config["method"],
#                                       perm_files)
#
#     if len(perm_files) > 1:
#         raise ValueError(
#             f"Too many permutation files associated with analysis {config['analysis_label']}: {perm_files}. This should not happen, maybe a bug?")
#     elif len(perm_files) == 1:
#         perm_file = perm_files[0]
#         perm_data = np.load(perm_file)
#         print(f"Loading {perm_data.shape[0]} pre-existing permutations from {perm_file}")
#     else:
#         # Note: if we compare task, then the task entity must disappear in the path, so we make it optional in the path_patterns
#         perm_file = layout.derivatives["connectomix"].build_path({**entities,
#                                                                   "analysis_label": config["analysis_label"],
#                                                                   "method": config["method"],
#                                                                   },
#                                                                  path_patterns=[
#                                                                      "group/{analysis_label}/permutations/group_[ses-{session}_][run-{run}_][task-{task}]_space-{space}_method-{method}_desc-{desc}_analysis-{analysis_label}_permutations.npy"],
#                                                                  validate=False)
#         from connectomix.core.utils.tools import make_parent_dir
#         make_parent_dir(perm_file)
#         # If nothing has to be loaded, then initiate the null distribution with the observed values
#         perm_data = np.array([list(observed_stats.values())])  # Size is (1,2) and order is max followed by min
#
#     # Run the permutations until goal is reached
#     print(f"Running permutations (target is {n_permutations} permutations)...", end="", flush=True)
#     while perm_data.shape[0] <= n_permutations:
#         print(".", end="", flush=True)
#         if analysis_type in ["independent", "paired"]:
#             group1_data = group_data[0]
#             group2_data = group_data[1]
#             if analysis_type == "independent":
#                 permutation_type = "independent"
#             elif analysis_type == "paired":
#                 permutation_type = "samples"
#             perm_test = permutation_test((group1_data, group2_data),
#                                          stat_func,
#                                          vectorized=False,
#                                          n_resamples=1,
#                                          permutation_type=permutation_type)
#             permuted_t_scores = perm_test.null_distribution
#
#         elif analysis_type == "regression":
#             permuted_t_scores, _ = regression_analysis(group_data, design_matrix, permutate=True)
#
#         null_data = np.array([np.nanmax(permuted_t_scores), np.nanmin(permuted_t_scores)])
#         perm_data = np.vstack((perm_data, null_data.reshape(1, -1)))
#
#         # Save to file
#         np.save(perm_file, perm_data)
#
#     print(".")
#     print("Permutations computed.")
#     return perm_data.reshape([-1, 1])[0:], perm_data.reshape([-1, 1])[1:]  # Returning all maxima and all minima


# def regression_analysis(group_data, design_matrix, permutate=False):
#     """
#     Performs regression analysis on symmetric connectivity matrices using vectorization.
#     Assumes the covariate is the first column of the design matrix and optionally permutes it.
#
#     Parameters:
#     - group_data: A numpy array of shape (n_subjects, n_nodes, n_nodes), where each entry is a symmetric connectivity matrix.
#     - design_matrix: A pandas DataFrame used as the design matrix for the regression.
#     - permutate: A boolean indicating whether to shuffle the covariate before performing the regression.
#
#     Returns:
#     - t_values_matrix: A symmetric matrix of t-values for the covariate, with shape (n_nodes, n_nodes).
#     - p_values_matrix: A symmetric matrix of p-values for the covariate, with shape (n_nodes, n_nodes).
#     """
#
#     # Get the number of subjects, nodes from group_data
#     group_data = np.array(group_data)
#     n_subjects, n_nodes, _ = group_data.shape
#
#     # Extract name of columns to permute
#     covariable_to_permute = design_matrix.columns[0]
#
#     # Since we add a constant in the design matrix, we must de-mean the columns of the design matrix
#     design_matrix = design_matrix.apply(lambda x: x - x.mean(), axis=0)
#
#     # Extract the covariate (first column of design matrix) and other covariates
#     X = add_constant(design_matrix)  # Add constant for the intercept
#
#     # If permutate is True, shuffle the first column (covariate) of the design matrix
#     if permutate:
#         X[covariable_to_permute] = np.random.permutation(X[covariable_to_permute])
#
#     # Vectorize the symmetric connectivity matrices (extract upper triangular part)
#     vec_group_data = np.array([sym_matrix_to_vec(matrix) for matrix in group_data])
#
#     # Get the number of unique connections (upper triangular part)
#     n_connections = vec_group_data.shape[1]
#
#     # Initialize arrays to store t-values and p-values for the vectorized form
#     t_values_vec = np.zeros(n_connections)
#     p_values_vec = np.zeros(n_connections)
#
#     # Run the regression for each unique connection
#     for idx in range(n_connections):
#         # Connectivity values (y) for this connection across subjects
#         y = vec_group_data[:, idx]
#
#         # Fit the OLS model
#         model = OLS(y, X).fit()
#
#         # Extract t-value and p-value for the covariate (first column)
#         t_values_vec[idx] = model.tvalues[covariable_to_permute]
#         p_values_vec[idx] = model.pvalues[covariable_to_permute]
#
#     # Convert the vectorized t-values and p-values back to symmetric matrices
#     t_values_matrix = vec_to_sym_matrix(t_values_vec)
#     p_values_matrix = vec_to_sym_matrix(p_values_vec)
#
#     return t_values_matrix, p_values_matrix


# def stat_func(x, y):
#     """
#     Function defining the statistics to compute for the permutation-based analysis.
#     Essentially calls ttest_ind(x, y).
#
#     Parameters
#     ----------
#     x : as in ttest_ind(x, y)
#     y : as in ttest_ind(x, y)
#
#     Returns
#     -------
#     t_stat : float
#         t-statistics, as computed from ttest_ind(x, y).
#
#     """
#     from scipy.stats import ttest_ind
#     t_stat, _ = ttest_ind(x, y)
#     return t_stat


def non_parametric_stats(glm, config):
    if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
        non_parametric_outputs = non_parametric_inference(glm.second_level_input_,
                                                          design_matrix=glm.design_matrix_,
                                                          second_level_contrast=config["contrast"],
                                                          first_level_contrast="effectOfSeedOrRoi",
                                                          smoothing_fwhm=config["smoothing"],
                                                          two_sided_test=config["two_sided_test"],
                                                          n_jobs=config["n_jobs"],
                                                          threshold=float(config["cluster_forming_alpha"]),
                                                          n_perm=config["n_permutations"])
        logp_max_stat = non_parametric_outputs["logp_max_mass"]
        permutation_dist = None  # TODO: extract t-null distribution from nilearn permuted_OLS function

    elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
        contrast = expression_to_contrast_vector(config["contrast"], glm["design_matrix"].columns)
        tested_variables = np.dot(np.array(glm["design_matrix"].values), contrast)
        non_parametric_outputs = permuted_ols(tested_variables,
                                              glm["data"],
                                              two_sided_test=config["two_sided_test"],
                                              n_jobs=config["n_jobs"],
                                              threshold=None,
                                              output_type="dict",
                                              n_perm=config["n_permutations"])
        logp_max_stat = vec_to_sym_matrix(non_parametric_outputs["logp_max_t"])
        permutation_dist = non_parametric_outputs["h0_max_t"][0,:]

    return logp_max_stat, permutation_dist


def compute_significant_data(contrast_results, glm, config):
    significant_data = {}
    permutation_dist= None
    for thresholding_strategy in config["thresholding_strategies"]:
        alpha = float(config[f"{thresholding_strategy}_alpha"])
        match thresholding_strategy:
            case "uncorrected":
                if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
                    significant_data[thresholding_strategy], _ = threshold_stats_img(contrast_results["z_values"],
                                                                                     alpha=alpha,
                                                                                     height_control=None,
                                                                                     two_sided=True)
                elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
                    uncorr_mask = contrast_results["p_values"] < alpha
                    significant_data[thresholding_strategy] = contrast_results["z_values"] * uncorr_mask

            case "fdr":
                if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
                    significant_data[thresholding_strategy], _ = threshold_stats_img(contrast_results["z_values"],
                                                                                     alpha=alpha,
                                                                                     height_control="fdr",
                                                                                     two_sided=True)
                elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
                    fdr_z_threshold = fdr_threshold(sym_matrix_to_vec(contrast_results["z_values"]), alpha)
                    if fdr_z_threshold is np.inf:
                        significant_data[thresholding_strategy] = None
                        print("Computed FDR threshold is infinite.")
                    else:
                        fdr_mask = contrast_results["z_values"] < fdr_z_threshold
                        significant_data[thresholding_strategy] = contrast_results["z_values"] * fdr_mask

            case "fwe":
                print("Fwe correction requested, computing permutations (this may take a while)...")
                logp_max_stat, permutation_dist = non_parametric_stats(glm, config)

                if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
                    mask = math_img(f"img >= -np.log10({alpha})", img=logp_max_stat)
                    mask = binarize_img(mask)

                    if img_is_not_empty(mask):
                        masked_data = apply_mask(contrast_results["z_values"], mask)
                        significant_data[thresholding_strategy] = unmask(masked_data, mask)
                    else:
                        warnings.warn(
                            f"No voxel survives FWE thresholding at alpha level {alpha} for this analysis.")
                        significant_data[thresholding_strategy] = None

                elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
                    fwe_mask = contrast_results["p_values"] < alpha
                    if np.any(fwe_mask != 0):
                        significant_data[thresholding_strategy] = contrast_results["z_values"] * fwe_mask
                    else:
                        warnings.warn(
                            f"No voxel survives FWE thresholding at alpha level {alpha} for this analysis.")
                        significant_data[thresholding_strategy] = None

    return significant_data, permutation_dist
