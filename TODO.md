# Misc
- complete/update documentation
- tune verbosity (using config file)
- check if nilearn orthogonalizes the covariates automatically or not in group-level analyzes

# Features
- report at individual level with denoising information
- report at group level
- add overlay with roi in roiToVoxel plots (both at participant- and group-level)
- add a config value config["categorial_variable"]=nameOfCovar to imply that the group-level covariate "nameOfCovar" must be treated as a categorial variable (like it is now done by default for "group")
- allow for regexes when defining noise confounds at participant-level
- export null distribution of max-stat also for xToVoxel methods
- save various z-thresholds (uncorr., FDR, and FWE) in json file

# Analyzes
- Paired samples testing: inter-session OR inter-task OR inter-run comparison
- ReHo, (f)ALFF, MVPA
- Dynamical Functional Connectivity
- Effective Connectivity, Granger causality