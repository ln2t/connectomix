import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
# from nilearn.glm.second_level import non_parametric_inference
from connectomix.core.processing.nilearn_tools import non_parametric_inference
from nilearn.glm import threshold_stats_img, fdr_threshold
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.mass_univariate import permuted_ols
from nilearn.image import math_img, binarize_img
from nilearn.masking import apply_mask, unmask

from connectomix.core.utils.tools import img_is_not_empty, custom_print


def compute_z_from_t(t_score, degree_of_freedom):
    from scipy.stats import t, norm
    cdf_t = t.cdf(t_score, degree_of_freedom)
    return norm.ppf(cdf_t)


def non_parametric_stats(glm, config):
    custom_print(f"Fwe correction requested, computing permutations with {config['n_permutations']} permutations"
          f" (this may take a while)...")
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
        permutation_dist = non_parametric_outputs["h0_max_t"][0, :]

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
                        custom_print("Computed FDR threshold is infinite.")
                    else:
                        fdr_mask = contrast_results["z_values"] < fdr_z_threshold
                        significant_data[thresholding_strategy] = contrast_results["z_values"] * fdr_mask

            case "fwe":
                logp_max_stat, permutation_dist = non_parametric_stats(glm, config)

                if config["method"] == "seedToVoxel" or config["method"] == "roiToVoxel":
                    mask = math_img(f"img >= -np.log10({alpha})", img=logp_max_stat)
                    mask = binarize_img(mask)

                    if img_is_not_empty(mask):
                        masked_data = apply_mask(contrast_results["z_values"], mask)
                        significant_data[thresholding_strategy] = unmask(masked_data, mask)
                    else:
                        custom_print(
                            f"No voxel survives FWE thresholding at alpha level {alpha} for this analysis.")
                        significant_data[thresholding_strategy] = None

                elif config["method"] == "seedToSeed" or config["method"] == "roiToRoi":
                    fwe_mask = contrast_results["p_values"] < alpha
                    if np.any(fwe_mask != 0):
                        significant_data[thresholding_strategy] = contrast_results["z_values"] * fwe_mask
                    else:
                        custom_print(
                            f"No voxel survives FWE thresholding at alpha level {alpha} for this analysis.")
                        significant_data[thresholding_strategy] = None

    return significant_data, permutation_dist
