# Misc
- complete/update documentation
- tune verbosity (using config file)
- check if nilearn orthogonalizes the covariates automatically or not in group-level analyzes

# Features
- report at individual level with denoising information
- report at group level
- add overlay with roi in roiToVoxel plots (both at participant- and group-level)
- add a config value config["categorial_variable"]=nameOfCovar to imply that the group-level covariate "nameOfCovar" must be treated as a categorial variable (like it is now done by default for "group")
- add standard predefined denoising strategies
- add a hash code for denoising fingerprint

# Analyzes
- Paired samples testing: inter-session OR inter-task OR inter-run comparison
- ReHo, (f)ALFF, MVPA
- Dynamical Functional Connectivity
- Effective Connectivity, Granger causality