import time, sys
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone

from nilearn._utils import  logger
from nilearn._utils.glm import check_and_load_tables

from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.surface.surface import (
    SurfaceImage,
)
from nilearn.glm.second_level.second_level import (_check_second_level_input,
                                                   _check_confounds,
                                                   _process_second_level_input,
                                                   _sort_input_dataframe,
                                                   _get_con_val,
                                                   _infer_effect_maps,
                                                   _check_n_rows_desmat_vs_n_effect_maps,
                                                   )
# This is an alternate version of nilearn.glm.second_level.non_parametric_inference
# such that it returns the null distribution ("h0_max_t")
def non_parametric_inference(
    second_level_input,
    confounds=None,
    design_matrix=None,
    second_level_contrast=None,
    first_level_contrast=None,
    mask=None,
    smoothing_fwhm=None,
    model_intercept=True,
    n_perm=10000,
    two_sided_test=False,
    random_state=None,
    n_jobs=1,
    verbose=0,
    threshold=None,
    tfce=False,
):
    """Generate p-values corresponding to the contrasts provided \
    based on permutation testing.

    This function is a light wrapper around
    :func:`~nilearn.mass_univariate.permuted_ols`, with additional steps to
    ensure compatibility with the :mod:`~nilearn.glm.second_level` module.

    Parameters
    ----------
    %(second_level_input)s

    %(second_level_confounds)s

    %(second_level_design_matrix)s

    %(second_level_contrast)s

    first_level_contrast : :obj:`str` or None, default=None
        In case a pandas DataFrame was provided as second_level_input this
        is the map name to extract from the pandas dataframe map_name column.
        It has to be a 't' contrast.

        .. versionadded:: 0.9.0

    %(second_level_mask)s

    %(smoothing_fwhm)s

        .. warning::

            Smoothing is not implemented for surface data.

    model_intercept : :obj:`bool`, default=True
        If ``True``, a constant column is added to the confounding variates
        unless the tested variate is already the intercept.

    %(n_perm)s

    %(two_sided_test)s

    %(random_state)s
        Use this parameter to have the same permutations in each
        computing units.

    %(n_jobs)s

    %(verbose0)s

    threshold : None or :obj:`float`, default=None
        Cluster-forming threshold in p-scale.
        This is only used for cluster-level inference.
        If None, no cluster-level inference will be performed.

        .. versionadded:: 0.9.2

        .. warning::

            Performing cluster-level inference will increase the computation
            time of the permutation procedure.

        .. warning::

            Cluster analysis are not implemented for surface data.

    %(tfce)s

        .. versionadded:: 0.9.2

        .. warning::

            TFCE analysis are not implemented for surface data.

    Returns
    -------
    neg_log10_vfwe_pvals_img : :class:`~nibabel.nifti1.Nifti1Image`
        The image which contains negative logarithm of the
        voxel-level FWER-corrected p-values.

        .. note::
            This is returned if ``threshold`` is None (the default).

    outputs : :obj:`dict`
        Output images, organized in a dictionary.
        Each image is 3D/4D, with the potential fourth dimension corresponding
        to the regressors.

        .. note::
            This is returned if ``tfce`` is True or ``threshold`` is not None.

        .. versionadded:: 0.9.2

        Here are the keys:

        =============== =======================================================
        key             description
        =============== =======================================================
        t               T-statistics associated with the significance test of
                        the n_regressors explanatory variates against the
                        n_descriptors target variates.
        logp_max_t      Negative log10 family-wise error rate-corrected
                        p-values corrected based on the distribution of maximum
                        t-statistics from permutations.
        size            Cluster size values associated with the significance
                        test of the n_regressors explanatory variates against
                        the n_descriptors target variates.

                        Returned only if ``threshold`` is not ``None``.
        logp_max_size   Negative log10 family-wise error rate-corrected
                        p-values corrected based on the distribution of maximum
                        cluster sizes from permutations.
                        This map is generated through cluster-level methods, so
                        the values in the map describe the significance of
                        clusters, rather than individual voxels.

                        Returned only if ``threshold`` is not ``None``.
        mass            Cluster mass values associated with the significance
                        test of the n_regressors explanatory variates against
                        the n_descriptors target variates.

                        Returned only if ``threshold`` is not ``None``.
        logp_max_mass   Negative log10 family-wise error rate-corrected
                        p-values corrected based on the distribution of maximum
                        cluster masses from permutations.
                        This map is generated through cluster-level methods, so
                        the values in the map describe the significance of
                        clusters, rather than individual voxels.

                        Returned only if ``threshold`` is not ``None``.
        tfce            :term:`TFCE` values associated
                        with the significance test of
                        the n_regressors explanatory variates against the
                        n_descriptors target variates.

                        Returned only if ``tfce`` is ``True``.
        logp_max_tfce   Negative log10 family-wise error rate-corrected
                        p-values corrected based on the distribution of maximum
                        TFCE values from permutations.

                        Returned only if ``tfce`` is ``True``.
        =============== =======================================================

    See Also
    --------
    :func:`~nilearn.mass_univariate.permuted_ols` : For more information on \
        the permutation procedure.

    References
    ----------
    .. footbibliography::
    """
    # check_params(locals()) # removed to simplify dependencies
    print("DEBUG: using custom nilearn function")
    _check_second_level_input(second_level_input, design_matrix)
    _check_confounds(confounds)
    design_matrix = check_and_load_tables(design_matrix, "design_matrix")[0]

    if isinstance(second_level_input, pd.DataFrame):
        second_level_input = _sort_input_dataframe(second_level_input)
    sample_map, _ = _process_second_level_input(second_level_input)

    if isinstance(sample_map, SurfaceImage) and smoothing_fwhm is not None:
        warn(
            "Parameter 'smoothing_fwhm' is not "
            "yet supported for surface data.",
            UserWarning,
            stacklevel=2,
        )
        smoothing_fwhm = None

    if (isinstance(sample_map, SurfaceImage)) and (tfce or threshold):
        tfce = False
        threshold = None
        warn(
            (
                "Cluster level inference not yet implemented "
                "for surface data.\n"
                f"Setting {tfce=} and {threshold=}."
            ),
            UserWarning,
            stacklevel=2,
        )

    # Report progress
    t0 = time.time()
    logger.log("Fitting second level model...", verbose=verbose)

    # Learn the mask. Assume the first level imgs have been masked.
    if isinstance(mask, (NiftiMasker, SurfaceMasker)):
        masker = clone(mask)
        if smoothing_fwhm is not None and masker.smoothing_fwhm is not None:
            warn("Parameter 'smoothing_fwhm' of the masker overridden.")
            masker.smoothing_fwhm = smoothing_fwhm

    elif isinstance(sample_map, SurfaceImage):
        masker = SurfaceMasker(
            mask_img=mask,
            smoothing_fwhm=smoothing_fwhm,
            memory=Memory(None),
            verbose=max(0, verbose - 1),
            memory_level=1,
        )
    else:
        masker = NiftiMasker(
            mask_img=mask,
            smoothing_fwhm=smoothing_fwhm,
            memory=Memory(None),
            verbose=max(0, verbose - 1),
            memory_level=1,
        )

    masker.fit(sample_map)

    # Report progress
    logger.log(
        "\nComputation of second level model done in "
        f"{time.time() - t0} seconds\n",
        verbose=verbose,
    )

    # Check and obtain the contrast
    contrast = _get_con_val(second_level_contrast, design_matrix)
    # Get first-level effect_maps
    effect_maps = _infer_effect_maps(second_level_input, first_level_contrast)

    _check_n_rows_desmat_vs_n_effect_maps(effect_maps, design_matrix)

    # Obtain design matrix vars
    var_names = design_matrix.columns.tolist()

    # Obtain tested_var
    column_mask = [bool(val) for val in contrast]
    tested_var = np.dot(design_matrix, contrast)

    # Remove tested var from remaining var names
    var_names = [var for var, mask in zip(var_names, column_mask) if not mask]

    # Obtain confounding vars
    # No other vars in design matrix by default
    confounding_vars = None
    if var_names:
        # Use remaining vars as confounding vars
        confounding_vars = np.asarray(design_matrix[var_names])

    # Mask data
    target_vars = masker.transform(effect_maps)

    # Perform massively univariate analysis with permuted OLS
    outputs = permuted_ols(
        tested_var,
        target_vars,
        confounding_vars=confounding_vars,
        model_intercept=model_intercept,
        n_perm=n_perm,
        two_sided_test=two_sided_test,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=max(0, verbose - 1),
        masker=masker,
        threshold=threshold,
        tfce=tfce,
        output_type="dict",
    )
    neg_log10_vfwe_pvals_img = masker.inverse_transform(
        np.ravel(outputs["logp_max_t"])
    )

    if (not tfce) and (threshold is None):
        return neg_log10_vfwe_pvals_img

    t_img = masker.inverse_transform(np.ravel(outputs["t"]))

    out = {
        "t": t_img,
        "logp_max_t": neg_log10_vfwe_pvals_img,
        "h0_max_t": outputs["h0_max_t"]
    }

    if tfce:
        neg_log10_tfce_pvals_img = masker.inverse_transform(
            np.ravel(outputs["logp_max_tfce"]),
        )
        out["tfce"] = masker.inverse_transform(np.ravel(outputs["tfce"]))
        out["logp_max_tfce"] = neg_log10_tfce_pvals_img

    if threshold is not None:
        # Cluster size-based p-values
        neg_log10_csfwe_pvals_img = masker.inverse_transform(
            np.ravel(outputs["logp_max_size"]),
        )

        # Cluster mass-based p-values
        neg_log10_cmfwe_pvals_img = masker.inverse_transform(
            np.ravel(outputs["logp_max_mass"]),
        )

        out["size"] = masker.inverse_transform(np.ravel(outputs["size"]))
        out["logp_max_size"] = neg_log10_csfwe_pvals_img
        out["mass"] = masker.inverse_transform(np.ravel(outputs["mass"]))
        out["logp_max_mass"] = neg_log10_cmfwe_pvals_img

    return out