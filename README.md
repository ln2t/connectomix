![License](https://img.shields.io/github/license/ln2t/connectomix)

# Connectomix: Functional Connectivity from fMRIPrep Outputs Using BIDS Structure

## Overview

Connectomix is a BIDS app designed to compute functional connectomes from fMRI data preprocessed with **fMRIPrep**.
It facilitates both participant-level and group-level analyses using various methods for defining regions of interest (ROIs) and computing connectivity measures.
Connectomix leverages the Brain Imaging Data Structure (BIDS) to organize data and outputs, ensuring compatibility with existing neuroimaging workflows.

**Key features include:**

- Extraction of time series data from preprocessed fMRI data, with flexible denoising strategies.
- Computation of functional connectivity matrices using methods like atlas-based parcellation, seed-based analysis, and Independent Component Analysis (ICA).
- Support for multiple connectivity measures (e.g., correlation, partial correlation).
- Generation of group-level statistical comparisons, including permutation testing.
- Automatic generation of default configuration files tailored to your dataset.
- BIDS-compliant outputs for seamless integration with other neuroimaging tools.
- An Autonomous mode where paths and configuration are guessed before launching the analysis, without any other input required.
- Report generation containing summary of the findings at group-level.

## Installation

### Docker Installation

You can run Connectomix using **Docker**, which encapsulates the application along with all its dependencies in a container. This method ensures a consistent environment without the need to manually install Python packages or manage dependencies. Of course, you must install docker by yourself beforehand, e.g. following this link [Docker Installation](https://www.docker.com/products/docker-desktop/).

**Pulling the Docker Image**

A pre-built Docker image is available on Docker Hub. Pull the image using the following command:

```bash
docker pull arovai/connectomix
```

**Running Connectomix with Docker**

To execute Connectomix using Docker, use the following command structure:

```bash
docker run --rm -it \
    -v /path/to/bids_dataset:/bids_dataset:ro \
    -v /path/to/derivatives:/derivatives \
    arovai/connectomix \
    /bids_dataset /derivatives participant
```

- **Mounting Volumes**:
  - `-v /path/to/bids_dataset:/bids_dataset:ro` mounts your BIDS dataset directory into the container at `/bids_dataset` as read-only (`ro`).
  - `-v /path/to/derivatives:/derivatives` mounts your derivatives directory where outputs will be written.
- **Command Arguments**:
  - `/bids_dataset` and `/derivatives` are the paths inside the container (as defined by the `-v` options) for the BIDS dataset and derivatives directory, respectively.
  - `participant` specifies the analysis level.
  - `--fmriprep_dir /bids_dataset/derivatives/fmriprep` tells Connectomix where to find the fMRIPrep outputs inside the container.

**Notes**:

- **Replace Paths**: Ensure you replace `/path/to/bids_dataset` and `/path/to/derivatives` with the actual paths on your system.
- **Permissions**: Docker must have permission to read from the BIDS dataset directory and write to the derivatives directory.
- **Adjusting for Group-Level Analysis**: To perform a group-level analysis, change the analysis level to `group` and provide the necessary configuration:

  ```bash
  docker run --rm -it \
      -v /path/to/bids_dataset:/bids_dataset:ro \
      -v /path/to/derivatives:/derivatives \
      arovai/connectomix \
      /bids_dataset /derivatives group --config /derivatives/config/group_config.yaml
  ```

  Ensure that your configuration file is accessible within the container. If it's stored in the derivatives directory, as in the example above, it will be available at `/derivatives/config/group_config.yaml` inside the container.

For more information on Docker and managing containers, visit the [Docker Documentation](https://docs.docker.com/get-started/).

### Python Installations

Connectomix requires **Python 3** and several Python packages. Install the necessary dependencies using `pip`:

```bash
pip install nibabel nilearn pandas numpy matplotlib scipy statsmodels bids yaml
```

Alternatively, you can use a virtual environment or conda environment to manage dependencies:

```bash
conda create -n connectomix_env python=3.8
conda activate connectomix_env
pip install nibabel nilearn pandas numpy matplotlib scipy statsmodels bids yaml
```

## Usage

Connectomix can be executed from the command line with the following syntax:

```bash
python connectomix.py bids_dir derivatives_dir analysis_level [options]
```

- **`bids_dir`**: The root directory of the BIDS dataset.
- **`derivatives_dir`**: The directory where outputs will be stored.
- **`analysis_level`**: The analysis level to run. Choose between `participant` or `group`.

### Command-Line Arguments

- `--fmriprep_dir`: Path to the fMRIPrep outputs directory. If not provided, Connectomix will look for it in `bids_dir/derivatives/fmriprep`.
- `--config`: Path to a configuration file (`.json`, `.yaml`, or `.yml`) specifying analysis parameters. If not provided, default parameters will be used.
- `--helper`: Generates a default configuration file based on the dataset contents.
- `--autonomous`: Runs Connectomix in autonomous mode, automatically guessing paths and settings. In that case only, all the other arguments can be ignored, but the command must be run at a specific place in your directory tree (see below).

### Examples

#### Participant-Level Analysis

To run a participant-level analysis, assuming fmriprep derivatives are stored in `/path/to/bids_dataset/derivatives/fmriprep`:

```bash
python connectomix.py /path/to/bids_dataset /path/to/derivatives/connectomix participant
```

This will use the default configuration for the analysis.
Moreover, if the fmriprep directory is located somewhere else, you can add the option `--fmriprep_dir /path/to/fmriprep` to manually specify this.

To generate a default participant-level configuration file:

```bash
python connectomix.py /path/to/bids_dataset /path/to/derivatives/connectomix participant --helper
```

The configuration file is then created in `/path/to/derivatives/connectomix/config` (the result is also printed in the terminal). Once you are happy with the configuration, you can launch the analysis with you configuration by using the option `--config /path/to/participant_config.yaml`

#### Group-Level Analysis

To run a group-level analysis:

```bash
python connectomix.py /path/to/bids_dataset /path/to/derivatives/connectomix group --config /path/to/group_config.yaml
```

Note that a configuration file is mandatory in that case, as this is the place where the two groups of subjects to compare are specified.

To generate a default group-level configuration file:

```bash
python connectomix.py /path/to/bids_dataset /path/to/derivatives group --helper
```

The result of this default configuration file is then saved at `/path/to/derivatives/connectomix/config` and can be edited before launching the analysis (the result is also printed in the terminal). Caution: by default, the number of permutations is set to 10000. This will take a lot of time to run, so if your goal is to test your configuration or simply if you don't care about this, set the field `n_permutations` to a very low value (e.g. 10) in the configuration file.

#### Autonomous Mode

Connectomix can be run in autonomous mode to automatically determine paths and settings:

```bash
python connectomix.py --autonomous
```

The strategy for this mode is as follows:
- 1. It checks if it has been run at the root of a BIDS directory. If yes, it uses this directory. If not, it checks if the current path contains a directory called `rawdata`. If yes, it checks if it is a BIDS directory. If yes, it use it for the analysis. If not, it throws an error.
- 2. It checks if there is a `derivatives` directory in the current path. If yes, it checks if there is a directory starting with `derivatives/fmriprep`. If there is a single match, it uses it for the analysis. If not, it throws an error.
- 3. Finally, it checks if there is a directory starting with `derivatives/connectomix`. If not, it launches the participant-level analysis with default configuration. If yes, it assumes that the participant-level analysis has been already completed, and run the `--helper` tool for the group level analysis.

Make sure to execute this once your are inside the root of the BIDS directory or, alternatively, one level up a folder called `rawdata`.

## Configuration

Connectomix uses configuration files in YAML or JSON format to specify analysis parameters. Separate configuration files are used for participant-level and group-level analyses.
See the `--helper` tools to generate default configuration files for your dataset (already fetching the subjects label, tasks, runs, sessions, etc), which are easy modified in a text editor.

### Participant-Level Configuration

A participant-level configuration file specifies parameters such as:

- **Subjects** to process.
- **Tasks**, **runs**, **sessions**, and **spaces** to include.
- **Confound variables** to use during time series extraction. They correspond to columns in the confounds computed by fMRIPrep.
- **Connectivity measures** to compute (e.g., correlation, partial correlation).
- **Methods** for defining ROIs (e.g., atlas-based, seed-based, ICA).

**Example participant-level configuration (`participant_config.yaml`):**

```yaml
# Connectomix Configuration File
# This file is generated automatically. Please modify the parameters as needed.
# Full documentation is located at github.com/ln2t/connectomix

# List of subjects
subjects:
  - sub-01
  - sub-02

# List of tasks
tasks:
  - restingstate

# List of runs
runs: []

# List of sessions
sessions: []

# List of output spaces
spaces:
  - MNI152NLin2009cAsym

# Confounding variables to include when extracting timeseries
confound_columns:
  - trans_x
  - trans_y
  - trans_z
  - rot_x
  - rot_y
  - rot_z
  - global_signal

# Kind of connectivity measure to compute
connectivity_kind: correlation  # Options: covariance, correlation, partial correlation, precision

# Method to define regions of interest
method: atlas  # Options: atlas, seeds, ica

# Method-specific options
method_options:
  n_rois: 100  # Number of ROIs in the atlas
  high_pass: 0.008  # High-pass filter frequency in Hz
  low_pass: 0.1  # Low-pass filter frequency in Hz
```

### Group-Level Configuration

A group-level configuration file specifies parameters such as:

- **group1_subjects** and **group2_subjects** in each group to compare.
- **Statistical thresholds** for uncorrected, FDR, and FWE corrections (enter here the alpha value for significance, or, in other words, the threshold p-value). Only performs two-sided tests.
- **Number of permutations** for permutation testing. Large value increases computational times. Note that permutations are stored in the derivatives, so if the analysis is interrupted it re-loads all pre-computed permutations.
- **Analysis type**, indicating whether the comparison is independent or paired (not implemented yet - work in progress).

**Example group-level configuration (`group_config.yaml`):**

```yaml
# Connectomix Configuration File
# This file is generated automatically. Please modify the parameters as needed.
# Full documentation is located at github.com/ln2t/connectomix

# Groups to compare
group1_subjects:
  - sub-01
  - sub-02
group2_subjects:
  - sub-03
  - sub-04
comparison_label: ControlsVsPatients  # Custom label for the comparison

# Type of statistical comparison
analysis_type: independent  # Options: independent, paired

# Statistical alpha-level thresholds
uncorrected_alpha: 0.001
fdr_alpha: 0.05
fwe_alpha: 0.05

# Number of permutations to estimate the null distributions
n_permutations: 10000

# Selected task
task: restingstate

# Selected run
run: []

# Selected session
session: []

# Selected space
space: MNI152NLin2009cAsym

# Kind of connectivity used at participant-level
connectivity_kind: correlation

# Method used at participant-level
method: atlas

# Method-specific options
method_options:
  n_rois: 100
```

## Outputs

Connectomix generates BIDS-compliant outputs in the specified derivatives directory. The outputs include:

- **Time Series Data**: Extracted time series for each ROI, saved as `.npy` files.
- **Connectivity Matrices**: Computed connectivity matrices for each subject, saved as `.npy` files.
- **Figures**: Visualizations of connectivity matrices and connectomes, saved as `.svg` files.
- **Group-Level Statistics**: T-statistics, p-values, and thresholded connectivity matrices from group comparisons.
- **Permutation Results**: Null distributions from permutation tests, saved as `.npy` files.
- **Reports**: HTML reports summarizing group-level analyses and including figures.
- **Configuration Backups**: Copies of the configuration files used, saved with timestamps for reproducibility.

**Output Organization:**

- Participant-level outputs are organized under `derivatives/connectomix/sub-XX/`.
- Group-level outputs are organized under `derivatives/connectomix/group/`.

## Dependencies

Ensure the following Python packages are installed:

- **Python 3**
- `nibabel`
- `nilearn`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `statsmodels`
- `bids`
- `yaml`
- `json`
- `hashlib`
- `datetime`
- `pathlib`

## Troubleshooting

- **No Functional Files Found**: Ensure your fMRIPrep outputs are correctly placed in the derivatives directory and that the `space`, `task`, `run`, and `session` specified in the configuration match the available data.
- **Mismatch in Number of Functional and Confound Files**: Verify that each functional file has a corresponding confound file from fMRIPrep.
- **Multiple Matches for Subjects**: Ensure your configuration parameters uniquely identify each subject's data.
- **Affine Mismatches**: If you encounter warnings about affine mismatches, Connectomix will attempt to resample the images. Ensure all functional images are in the same space and have compatible affines.

## Contact

For questions, issues, or contributions, please visit the GitHub repository:

[https://github.com/ln2t/connectomix](https://github.com/ln2t/connectomix)

## Acknowledgments

Connectomix leverages open-source neuroimaging tools and standards such as nilearn and BIDS, and we thank the developers and contributors of these projects.
