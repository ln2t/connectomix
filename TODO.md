# Misc
- complete/update documentation/docstrings
- tune verbosity (using config file)
- check if nilearn orthogonalizes coriates automatically or not in group-level analyzes
- check nilearn warning about singular design matrix while it should not...?

# Features
- report at individual level with denoising information
- report at group level
- add overlay with roi in roiToVoxel plots (both at participant- and group-level)
- add a config value config["categorial_variable"]=nameOfCovar to imply that the group-level covariate "nameOfCovar" must be treated as a categorial variable (like it is now done by default for "group")
- allow for regexes when defining noise confounds at participant-level
- save various z-thresholds (uncorr., FDR, and FWE) in json file

# Analyzes
- Paired samples testing: inter-session OR inter-task OR inter-run comparison
- ReHo, (f)ALFF, MVPA
- Dynamical Functional Connectivity
- Effective Connectivity, Granger causality