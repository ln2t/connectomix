# Connectomix: Functional Connectivity from fMRIPrep Outputs Using BIDS Structure

![License](https://img.shields.io/github/license/ln2t/connectomix) [![codecov](https://codecov.io/gh/ln2t/connectomix/graph/badge.svg?token=3PAEFC703X)](https://codecov.io/gh/ln2t/connectomix)

## Overview

Connectomix is a BIDS app designed to compute functional connectivity measures from fMRI data preprocessed with **fMRIPrep**. It facilitates both participant-level and group-level analyses using various methods for defining regions of interest (ROIs) and computing connectivity measures. Connectomix leverages the Brain Imaging Data Structure (BIDS) to organize data and outputs, ensuring compatibility with existing neuroimaging workflows.

### Available Methods

Four methods are available: seed-to-voxel, ROI-to-voxel, seed-to-seed, and ROI-to-ROI:

- **Seed-to-Voxel**: The user specifies the coordinates of the seed. The average signal in a sphere centered on that seed is compared to the signal in any other voxel. The radius is customizable, and the pipeline can loop over a list of seeds if several are provided. The result is a set of statistical maps, one for each seed.
- **ROI-to-Voxel**: The user provides a mask from which the average signal is compared to other voxels. Several masks can be provided, and the pipeline repeats the analysis for each. The outputs are statistical maps, one for each mask.
- **Seed-to-Seed**: The user specifies a list of seeds, from which a set of time courses is extracted by defining a sphere on each seed. Correlations are then computed between these time courses, and the outputs are connectivity matrices.
- **ROI-to-ROI**: Similar to seed-to-seed, but the signal is averaged on ROIs defined by an atlas. Several common atlases are supported, and the pipeline can also build a data-driven atlas using CanICA.

For each of these methods, the analysis can also be performed at the group level, where participant-level data are analyzed using the General Linear Model. The user can specify confounds and covariates, and request any type of contrasts. Statistical inference follows, testing for significant correlations. Correction for multiple comparisons includes FDR and FWE, the latter using permutation testing to estimate the max-stat distribution under the null hypothesis.

### Reproducibility

Reproducibility is a central concern for the contributors of this project. The pipeline complies with the BIDS-app guidelines, leveraging the well-established BIDS framework to organize raw and processed data. The complete set of parameters or options are defined (and can be customized) using configuration files. These files use a human-readable text format, making them easy to understand, share, and modify. If a parameter is not explicitly set by the user, the pipeline uses default values from the literature and verbosely reports any of these choices.

Another important feature at the participant level is the ability to denoise the data using conventional denoising strategies.

### Similar Software

Similar software includes giga_connectome, xcp_d, and CONN.

## What Connectomix Does NOT Do

The project is in constant development. If one of your needs is not met today, please check back later or feel free to introduce a feature request using the issue tracker.

Currently, the following features are not supported:
- Paired analysis (coming soon)
- Participant-level reports, including metrics to assess denoising efficiency
- Group-level reports
- ReHo and (f)ALFF analysis

## Configuration Files for Reproducible Analyses

One crucial issue when running an analysis, especially with software that has a graphical user interface (GUI), is knowing exactly what has been done to your data to arrive at the results you want to report. This is particularly important when writing scientific papers, where transparency regarding the various steps and parameters is essential.

To address these issues, two components are necessary:
1. The code itself should be freely available, preferably in a version-controlled fashion.
2. The exact parameters fed into the code to process your data are required.

The first component is managed by the Git repository structure and the versioning system it allows. The second component is made possible through the use of configuration files, which are central to the pipeline. These files specify the parameters or options in the pipeline. If a parameter is missing, Connectomix uses its default value and warns the user.

Each time the software is launched, it creates a copy of the configuration file, allowing you to "roll back" to previous analyses if things get messed up.

### Available Options/Parameters

The following elements are for both participant- and group-level analyses. Options specific to group-level processing are given below.

**BIDS Entities**

These fields allow Connectomix to select the data you want to analyze using BIDS entities to filter appropriate files in the dataset. The following types of BIDS entities can be specified: "subject", "tasks", "sessions", "runs", "space".

- **Participant-Level**: Each field can be a single value (e.g., "task: restingstate") or a list (e.g., "sessions: [1,2]"). If unspecified, these fields are filled automatically by parsing the outputs of fMRIPrep.
- **Group-Level**: Only "subjects" can be a list, corresponding to the list of subjects to include in the analysis. The other fields must be a single value. The default behavior is to check what is present in the output of Connectomix at the participant level.

**Post-fMRIPrep Processing: Denoising**

Once the data to process is determined, the pipeline needs to know how you want it to be denoised. Denoising is crucial for fMRI data analysis, and there are many factors to consider. fMRIPrep is agnostic about the exact denoising strategy and prepares the ground by generating a table of potential confounds for each subject. The user must specify the list of confounds to remove. For example, to include the standard 6 motion parameters:

```yaml
confounds: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
```

Band-pass filtering can be tuned using the `high_pass` and `low_pass` fields. Default values are 0.1Hz and 0.08Hz, respectively.

Images are resampled to a reference image to prepare for group analysis. The reference image is by default the first found when listing the functional files, but it can be set to any path pointing to a functional image.

**Participant-Level Analysis: Methods and Options**

Once the data are resampled, band-pass filtered, and denoised, we are ready to compute connectivity measures. The general idea is to estimate connectivity in fMRI signals by computing correlations between time signals. There are several ways to define the signal of interest from the data, such as using seeds or regions of interest (ROIs). Combining these options yields the four methods implemented in Connectomix:

- Seed-to-Voxel
- ROI-to-Voxel
- Seed-to-Seed
- ROI-to-ROI

The outputs of x-to-voxel methods are brain maps (saved in `.nii.gz` format), while for x-to-x methods, the results are connectivity matrices.

**Method-Specific Details**

- **Seed-to-Voxel**: Set the `method` field to `seedToVoxel` and specify the seed coordinates in a `.tsv` file. The `radius` field sets the radius of the spheres centered on the coordinates. The default value is 5 mm.
- **ROI-to-Voxel**: Set the `method` field to `roiToVoxel` and define the ROI using the `roi_masks` field, providing the path to the mask in `.nii.gz` format.
- **Seed-to-Seed**: Set the `method` field to `seedToSeed` and specify the seeds in a `.tsv` file. The `radius` field can be used to tune the radius of the spheres.
- **ROI-to-ROI**: Set the `method` field to `roiToRoi` and provide the `atlas` field with values such as `aal`, `schaeffer100`, or `harvardoxford`.

**Group-Level Analyses**

Participant-level results can be analyzed at the group level using the General Linear Model. The design matrix is central to this analysis, and the user can choose the columns by specifying the `group_confounds` field in the configuration file. The value of this field must be a list of strings corresponding to the names of columns from the `participants.tsv` file (e.g., `age` or `clinical_score`). If `group` is chosen, Connectomix assumes it is a categorical variable and creates one covariate for each group.

The model is fitted to the data, and a contrast is computed according to the `group_contrast` field. The contrast is a string made of the names of the columns in the design matrix combined with `+` or `-` signs. For example, to test for differences between controls and patients, use `control-patient` as a contrast.

Connectomix performs statistical tests based on the t-scores of the contrast maps. The first test keeps only the values with a significance level set by the `uncorrected_alpha` field (default: 0.01). Correction for multiple comparisons includes FDR (default: 0.05) and FWE using permutation testing. The significance level for FWE is set using the `fwe_alpha` field.

For each statistical test, one can choose to perform either one-sided or two-sided tests by setting the `two_sided_test` value to `False` or `True`, respectively.

## Installation

### Docker Installation (MacOS users: this is the less painful method)

You can run Connectomix using **Docker**, which encapsulates the application along with all its dependencies in a container. This method ensures a consistent environment without the need to manually install Python packages or manage dependencies. You must install Docker beforehand, e.g., following this link [Docker Installation](https://www.docker.com/products/docker-desktop/).

**Pulling the Docker Image**

A pre-built Docker image is available on Docker Hub. Pull the image using the following command:

```bash
sudo docker pull arovai/connectomix
```

**Running Connectomix with Docker**

To execute Connectomix using Docker, use the following command structure:

```bash
sudo docker run --rm -it \
    -v /path/to/bids_dataset:/bids_dataset:ro \
    -v /path/to/derivatives:/derivatives \
    arovai/connectomix \
    /bids_dataset /derivatives participant
```

- **Mounting Volumes**:
  - `-v /path/to/bids_dataset:/bids_dataset:ro` mounts your BIDS dataset directory into the container at `/bids_dataset` as read-only (`ro`).
  - `-v /path/to/derivatives:/derivatives` mounts your derivatives directory where outputs will be written.
- **Command Arguments**:
  - `/bids_dataset` and `/derivatives` are the paths inside the container for the BIDS dataset and derivatives directory, respectively.
  - `participant` specifies the analysis level.
  - `--fmriprep_dir /bids_dataset/derivatives/fmriprep` tells Connectomix where to find the fMRIPrep outputs inside the container.

**Notes**:

- **Adjusting for Group-Level Analysis**: To perform a group-level analysis, change the analysis level to `group` and provide the necessary configuration:

  ```bash
  sudo docker run --rm -it \
      -v /path/to/bids_dataset:/bids_dataset:ro \
      -v /path/to/derivatives:/derivatives \
      arovai/connectomix \
      /bids_dataset /derivatives group --config /derivatives/config/group_config.yaml
  ```

  Ensure that your configuration file is accessible within the container. If it's stored in the derivatives directory, as in the example above, it will be available at `/derivatives/config/group_config.yaml` inside the container.

For more information on Docker and managing containers, visit the [Docker Documentation](https://docs.docker.com/get-started/).

### Python Installation

Connectomix requires **Python 3.11** and several Python packages. Install the necessary dependencies using the provided `requirements.txt` file.

We strongly recommend using a virtual environment to manage dependencies. For instance:

```bash
git clone git@github.com:ln2t/connectomix.git
virtualenv venv python=3.11
source venv/bin/activate
pip install -e .
```

This will add Connectomix to your path, so typing `connectomix -h` should work.

## Usage

Connectomix can be executed with the following syntax:

```bash
connectomix bids_dir derivatives_dir analysis_level [options]
```

- **`bids_dir`**: The root directory of the BIDS dataset.
- **`derivatives_dir`**: The directory where outputs will be stored.
- **`analysis_level`**: The analysis level to run. Choose between `participant` or `group`.

### Command-Line Arguments

- `--derivatives fmriprep=/path/to/fmriprep`: Path to the fMRIPrep outputs directory. If not provided, Connectomix will look for it in `bids_dir/derivatives/fmriprep`.
- `--config`: Path to a configuration file (`.json`, `.yaml`, or `.yml`) specifying analysis parameters. If not provided, default parameters will be used.

### Examples

#### Participant-Level Analysis

To run a participant-level analysis, assuming fmriprep derivatives are stored in `/path/to/bids_dataset/derivatives/fmriprep`:

```bash
connectomix /path/to/bids_dataset /path/to/derivatives/connectomix participant
```

This will use the default configuration for the analysis. If the fmriprep directory is located elsewhere, you can add the option `--derivatives fmriprep=/path/to/fmriprep` to manually specify this.

To generate a default participant-level configuration file:

```bash
connectomix /path/to/bids_dataset /path/to/derivatives/connectomix participant --derivatives fmriprep=/path/to/fmriprep --helper
```

The configuration file is then created in `/path/to/derivatives/connectomix/config` (the result is also printed in the terminal). Once you are happy with the configuration, you can launch the analysis with your configuration by using the option `--config /path/to/participant_config.yaml`:

```bash
connectomix /path/to/bids_dataset /path/to/derivatives/connectomix participant --derivatives fmriprep=/path/to/fmriprep --config /path/to/participant_config.yaml
```

**Important note about denoising**: The configuration file contains the confounding time-series that one wishes to remove from the signal, as well as other denoising options such as low- and high-pass filtering cutoffs. Unfortunately, there is no one-size-fits-all method for denoising fMRI data. By default, Connectomix will select 6 motion parameters as well as white matter + CSF signal regression. The signal is also demeaned, de-trended, and filtered to keep the 0.01Hz-0.08Hz band. This is a standard choice. For more information, please consult Ciric et al., "Benchmarking of participant-level confound regression strategies for the control of motion artifact in studies of functional connectivity," NeuroImage, 2017.

#### Group-Level Analysis

To run a group-level analysis:

```bash
connectomix /path/to/bids_dataset /path/to/derivatives/connectomix group --config /path/to/group_config.yaml
```

To generate a default group-level configuration file:

```bash
connectomix /path/to/bids_dataset /path/to/derivatives group --helper
```

The result of this default configuration file is then saved at `/path/to/derivatives/connectomix/config` and can be edited before launching the analysis (the result is also printed in the terminal). Caution: by default, the number of permutations is set to 10,000. This will take a lot of time to run, so if your goal is to test your configuration or if you don't care about this, set the field `n_permutations` to a very low value (e.g., 10) in the configuration file.

### Participant-Level Configuration

**Example participant-level configuration (`participant_config.yaml`):**

```yaml
confounds:
- trans_x
- trans_y
- trans_z
- rot_x
- rot_y
- rot_z
- csf_wm
high_pass: 0.01
ica_aroma: false
low_pass: 0.08
method: seedToVoxel
overwrite_denoised_files: true
radius: 5
reference_functional_file: first_functional_file
runs: []
seeds_file: /path/to/seeds.tsv
sessions:
- '1'
- '2'
- '3'
- '4'
spaces:
- MNI152NLin2009cAsym
subject:
- 01
- 02
- 03
- 04
- 05
tasks:
- restingstate
```

### Group-Level Configuration

**Example group-level configuration (`group_config.yaml`):**

```yaml
method: seedToVoxel
add_intercept: true
analysis_name: customName
cluster_forming_alpha: '0.01'
contrast: intercept
covariates: []
fdr_alpha: 0.05
fwe_alpha: 0.05
n_jobs: 1
n_permutations: 20
paired_tests: false
radius: 5
runs: []
seeds_file: null
sessions: []
smoothing: 8
spaces: []
subject: []
tasks: []
thresholding_strategies:
- uncorrected
- fdr
- fwe
two_sided_test: true
uncorrected_alpha: 0.001
```

## Troubleshooting

- **No Functional Files Found**: Ensure your fMRIPrep outputs are correctly placed in the derivatives directory and that the `space`, `task`, `run`, and `session` specified in the configuration match the available data.
- **Mismatch in Number of Functional and Confound Files**: Verify that each functional file has a corresponding confound file from fMRIPrep.
- **Multiple Matches for Subjects**: Ensure your configuration parameters uniquely identify each subject's data.

## Contact

For questions, issues, or contributions, please visit the GitHub repository:

[https://github.com/ln2t/connectomix](https://github.com/ln2t/connectomix)

## Acknowledgments

Connectomix leverages open-source neuroimaging tools and standards such as nilearn and BIDS, and we thank the developers and contributors of these projects.
