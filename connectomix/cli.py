"""Command-line interface for Connectomix."""

import argparse
import sys
import textwrap
from pathlib import Path
from connectomix.core.version import __version__


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ColoredHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter with colored output and better organization."""
    
    def __init__(self, prog, indent_increment=2, max_help_position=40, width=100):
        super().__init__(prog, indent_increment, max_help_position, width)
    
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = f'{Colors.BOLD}Usage:{Colors.END} '
        return super()._format_usage(usage, actions, groups, prefix)
    
    def start_section(self, heading):
        # Add color to section headings
        if heading:
            heading = f'{Colors.BOLD}{Colors.CYAN}{heading}{Colors.END}'
        super().start_section(heading)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance with detailed help.
    """
    
    # Detailed description
    description = textwrap.dedent(f"""
    {Colors.BOLD}{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
    ║                              CONNECTOMIX v{__version__}                              ║
    ║         Functional Connectivity Analysis from fMRIPrep Outputs              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
    
    {Colors.BOLD}Description:{Colors.END}
      Connectomix performs functional connectivity analysis on fMRI data that has
      been preprocessed with fMRIPrep. It supports multiple connectivity methods
      at both participant and group levels.
    
    {Colors.BOLD}Connectivity Methods:{Colors.END}
      • {Colors.CYAN}seed-to-voxel{Colors.END}  - Correlation between seed regions and all brain voxels
      • {Colors.CYAN}roi-to-voxel{Colors.END}   - Correlation between atlas ROIs and all brain voxels
      • {Colors.CYAN}seed-to-seed{Colors.END}   - Correlation matrix between user-defined seeds
      • {Colors.CYAN}roi-to-roi{Colors.END}     - Correlation matrix between atlas regions
    
    {Colors.BOLD}Analysis Levels:{Colors.END}
      • {Colors.CYAN}participant{Colors.END}    - Process individual subjects (first-level analysis)
      • {Colors.CYAN}group{Colors.END}          - Statistical analysis across subjects (second-level GLM)
    
    {Colors.BOLD}Workflow:{Colors.END}
      1. Run participant-level analysis for each subject
      2. Run group-level analysis to perform statistical inference
    """)
    
    # Detailed epilog with examples
    epilog = textwrap.dedent(f"""
    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}EXAMPLES{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    
    {Colors.BOLD}Basic Usage:{Colors.END}
    
      {Colors.YELLOW}# Run participant-level analysis with default settings{Colors.END}
      connectomix /data/bids /data/derivatives/connectomix participant
    
      {Colors.YELLOW}# Run group-level analysis{Colors.END}
      connectomix /data/bids /data/derivatives/connectomix group
    
    {Colors.BOLD}With Configuration File:{Colors.END}
    
      {Colors.YELLOW}# Use a YAML configuration file{Colors.END}
      connectomix /data/bids /data/output participant --config analysis_config.yaml
    
      {Colors.YELLOW}# Use a JSON configuration file{Colors.END}
      connectomix /data/bids /data/output group --config group_config.json
    
    {Colors.BOLD}Specifying fMRIPrep Location:{Colors.END}
    
      {Colors.YELLOW}# When fMRIPrep output is not in default location{Colors.END}
      connectomix /data/bids /data/output participant \\
          --derivatives fmriprep=/data/derivatives/fmriprep
    
      {Colors.YELLOW}# Multiple derivatives sources{Colors.END}
      connectomix /data/bids /data/output participant \\
          --derivatives fmriprep=/path/to/fmriprep \\
          --derivatives freesurfer=/path/to/freesurfer
    
    {Colors.BOLD}Filtering Subjects/Sessions:{Colors.END}
    
      {Colors.YELLOW}# Process only subject 01{Colors.END}
      connectomix /data/bids /data/output participant --participant-label 01
    
      {Colors.YELLOW}# Process specific task, session, and run{Colors.END}
      connectomix /data/bids /data/output participant \\
          --participant-label 01 \\
          --task restingstate \\
          --session 1 \\
          --run 1
    
      {Colors.YELLOW}# Process only data in specific space{Colors.END}
      connectomix /data/bids /data/output participant --space MNI152NLin2009cAsym
    
    {Colors.BOLD}Using Denoising Strategies:{Colors.END}
    
      {Colors.YELLOW}# Use minimal denoising (motion parameters only){Colors.END}
      connectomix /data/bids /data/output participant --denoising minimal
    
      {Colors.YELLOW}# Use CSF+WM with 6 motion parameters{Colors.END}
      connectomix /data/bids /data/output participant --denoising csfwm_6p
    
      {Colors.YELLOW}# Include global signal regression{Colors.END}
      connectomix /data/bids /data/output participant --denoising gs_csfwm_12p
    
    {Colors.BOLD}Selecting Atlas and Method:{Colors.END}
    
      {Colors.YELLOW}# Use AAL atlas for ROI-to-ROI connectivity{Colors.END}
      connectomix /data/bids /data/output participant --atlas aal --method roiToRoi
    
      {Colors.YELLOW}# Use Schaefer 200-parcel atlas{Colors.END}
      connectomix /data/bids /data/output participant --atlas schaefer2018n200
    
      {Colors.YELLOW}# Seed-to-voxel connectivity (requires seeds_file in config){Colors.END}
      connectomix /data/bids /data/output participant --method seedToVoxel -c config.yaml
    
    {Colors.BOLD}Verbose Output:{Colors.END}
    
      {Colors.YELLOW}# Enable debug-level logging{Colors.END}
      connectomix /data/bids /data/output participant --verbose
    
    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}CONFIGURATION FILE{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    
    Configuration files (YAML or JSON) allow fine-grained control over analysis
    parameters. See documentation for full configuration options.
    
    {Colors.BOLD}Example participant config (YAML):{Colors.END}
    
      method: seed-to-voxel
      seeds_file: /path/to/seeds.tsv
      space: MNI152NLin2009cAsym
      high_pass: 0.008
      low_pass: 0.1
      denoising_strategy: csfwm_6p
    
    {Colors.BOLD}Example group config (YAML):{Colors.END}
    
      contrast: "patient - control"
      covariates: [age, sex]
      threshold_method: fdr
      alpha: 0.05
      n_permutations: 5000
    
    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}DENOISING STRATEGIES{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    
      {Colors.CYAN}minimal{Colors.END}        6 motion parameters only
      {Colors.CYAN}csfwm_6p{Colors.END}       CSF + WM + 6 motion parameters
      {Colors.CYAN}csfwm_12p{Colors.END}      CSF + WM + 12 motion params (6 + derivatives)
      {Colors.CYAN}gs_csfwm_6p{Colors.END}    Global signal + CSF + WM + 6 motion params
      {Colors.CYAN}gs_csfwm_12p{Colors.END}   Global signal + CSF + WM + 12 motion params
      {Colors.CYAN}csfwm_24p{Colors.END}      CSF + WM + 24 motion params (6 + deriv + squares)
      {Colors.CYAN}compcor_6p{Colors.END}     6 aCompCor components + 6 motion params
      {Colors.CYAN}simpleGSR{Colors.END}      Global + CSF + WM + 24 motion (preserves time series)
      {Colors.CYAN}scrubbing5{Colors.END}     CSF/WM derivatives + 24 motion + FD=0.5cm + scrub=5

    {Colors.BOLD}Choosing between simpleGSR and scrubbing5 (Wang et al. 2024):{Colors.END}
      • Use {Colors.CYAN}simpleGSR{Colors.END} for continuous time series (autoregressive, spectral analysis)
      • Use {Colors.CYAN}scrubbing5{Colors.END} for high-motion data when denoising quality is priority
      • Note: scrubbing5 is rigid (cannot combine with --fd-threshold or --scrub)
    
    {Colors.BOLD}WILDCARD SUPPORT:{Colors.END}
      Confound names support wildcards: {Colors.CYAN}*{Colors.END} matches any chars, {Colors.CYAN}?{Colors.END} matches one char
      
      {Colors.YELLOW}# Example: Select all aCompCor components{Colors.END}
      confounds: ["a_comp_cor_*", "trans_*", "rot_*"]
    
    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}OUTPUT STRUCTURE{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    
    Connectomix outputs are BIDS-compliant derivatives:
    
      output_dir/
      ├── dataset_description.json
      ├── sub-01/
      │   └── func/
      │       ├── sub-01_task-rest_space-MNI_desc-connectivity_bold.nii.gz
      │       └── sub-01_task-rest_space-MNI_desc-connectivity_bold.json
      └── group/
          ├── group_task-rest_contrast-patientVsControl_stat-t_statmap.nii.gz
          └── group_task-rest_contrast-patientVsControl_clusters.tsv
    
    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}MORE INFORMATION{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    
      Documentation:  https://github.com/ln2t/connectomix
      Report Issues:  https://github.com/ln2t/connectomix/issues
      Version:        {__version__}
    """)
    
    parser = argparse.ArgumentParser(
        prog="connectomix",
        description=description,
        epilog=epilog,
        formatter_class=ColoredHelpFormatter,
        add_help=False,  # We'll add custom help
    )
    
    # =========================================================================
    # REQUIRED ARGUMENTS
    # =========================================================================
    required = parser.add_argument_group(
        f'{Colors.BOLD}Required Arguments{Colors.END}'
    )
    
    required.add_argument(
        "bids_dir",
        type=Path,
        metavar="BIDS_DIR",
        help="Path to the BIDS dataset root directory. Must contain a valid "
             "dataset_description.json file.",
    )
    
    required.add_argument(
        "output_dir",
        type=Path,
        metavar="OUTPUT_DIR",
        help="Path to output directory where Connectomix derivatives will be "
             "stored. Will be created if it does not exist.",
    )
    
    required.add_argument(
        "analysis_level",
        choices=["participant", "group"],
        metavar="{participant,group}",
        help="Analysis level to perform. 'participant' processes individual "
             "subjects. 'group' performs second-level statistical analysis.",
    )
    
    # =========================================================================
    # OPTIONAL ARGUMENTS - General
    # =========================================================================
    general = parser.add_argument_group(
        f'{Colors.BOLD}General Options{Colors.END}'
    )
    
    general.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    
    general.add_argument(
        "--version",
        action="version",
        version=f"connectomix {__version__}",
        help="Show program version and exit.",
    )
    
    general.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging). Useful for "
             "troubleshooting.",
    )
    
    general.add_argument(
        "-c", "--config",
        type=Path,
        metavar="FILE",
        help="Path to configuration file (.json, .yaml, or .yml). Configuration "
             "files allow detailed control over analysis parameters. Command-line "
             "arguments override config file settings.",
    )
    
    # =========================================================================
    # OPTIONAL ARGUMENTS - Derivatives
    # =========================================================================
    derivatives = parser.add_argument_group(
        f'{Colors.BOLD}Derivatives Options{Colors.END}'
    )
    
    derivatives.add_argument(
        "-d", "--derivatives",
        action="append",
        metavar="NAME=PATH",
        dest="derivatives",
        help="Specify location of BIDS derivatives. Format: name=path "
             "(e.g., fmriprep=/data/derivatives/fmriprep). Can be specified "
             "multiple times for different derivatives. If not specified, "
             "Connectomix searches for 'fmriprep' in BIDS_DIR/derivatives/.",
    )
    
    # =========================================================================
    # OPTIONAL ARGUMENTS - BIDS Filters
    # =========================================================================
    filters = parser.add_argument_group(
        f'{Colors.BOLD}BIDS Entity Filters{Colors.END}',
        "Filter which data to process based on BIDS entities. "
        "Useful for processing subsets of data."
    )
    
    filters.add_argument(
        "-p", "--participant-label",
        metavar="LABEL",
        dest="participant_label",
        help="Process only this participant. Specify without 'sub-' prefix "
             "(e.g., '01' not 'sub-01'). Can be a single label.",
    )
    
    filters.add_argument(
        "-t", "--task",
        metavar="TASK",
        help="Process only this task (e.g., 'restingstate', 'nback'). "
             "Specify without 'task-' prefix.",
    )
    
    filters.add_argument(
        "-s", "--session",
        metavar="SESSION",
        help="Process only this session (e.g., '1', 'pre', 'post'). "
             "Specify without 'ses-' prefix.",
    )
    
    filters.add_argument(
        "-r", "--run",
        metavar="RUN",
        type=int,
        help="Process only this run number (e.g., 1, 2).",
    )
    
    filters.add_argument(
        "--space",
        metavar="SPACE",
        help="Process only data in this template space "
             "(e.g., 'MNI152NLin2009cAsym', 'MNI152NLin6Asym'). "
             "Must match fMRIPrep output space.",
    )
    
    filters.add_argument(
        "--label",
        metavar="STRING",
        help="Add a custom label to ALL output filenames as a BIDS-style entity "
             "(e.g., --label myanalysis will add 'label-myanalysis' to filenames). "
             "Useful for distinguishing different analysis runs.",
    )
    
    # =========================================================================
    # OPTIONAL ARGUMENTS - Temporal Censoring
    # =========================================================================
    censoring = parser.add_argument_group(
        f'{Colors.BOLD}Temporal Censoring Options{Colors.END}',
        "Remove specific timepoints (volumes) before connectivity analysis. "
        "Disabled by default. Enable with --conditions or --fd-threshold."
    )
    
    censoring.add_argument(
        "--conditions",
        metavar="COND",
        nargs="+",
        help="Enable condition-based censoring for task fMRI. "
             "Specify one or more condition names from the events.tsv file. "
             "A separate connectivity matrix will be computed for each condition. "
             "Use 'baseline' to select inter-trial intervals (timepoints not in any task). "
             "Example: --conditions face house  OR  --conditions baseline",
    )
    
    censoring.add_argument(
        "--events-file",
        metavar="FILE",
        dest="events_file",
        help="Path to events.tsv file (default: auto-detect from BIDS). "
             "Only used with --conditions.",
    )
    
    censoring.add_argument(
        "--include-baseline",
        action="store_true",
        dest="include_baseline",
        help="When using --conditions, also compute connectivity for baseline "
             "(timepoints not in any condition). Equivalent to adding 'baseline' to --conditions.",
    )
    
    censoring.add_argument(
        "--transition-buffer",
        metavar="SEC",
        type=float,
        dest="transition_buffer",
        default=0.0,
        help="Seconds to exclude around condition boundaries (default: 0). "
             "Accounts for hemodynamic response lag.",
    )
    
    censoring.add_argument(
        "--fd-threshold",
        metavar="CM",
        type=float,
        dest="fd_threshold",
        help=("Enable motion censoring. Remove volumes with framewise displacement "
              "above this threshold. Note: fMRIPrep reports FD values in centimeters (cm), "
              "so this argument expects a value in cm. Typical FD thresholds reported in the "
              "literature are 0.2–0.5 cm. Uses the 'framewise_displacement' "
              "column from fMRIPrep confounds."),
    )
    
    censoring.add_argument(
        "--fd-extend",
        metavar="N",
        type=int,
        dest="fd_extend",
        default=0,
        help="Number of volumes to also censor before AND after high-motion volumes "
             "(default: 0). Example: --fd-extend 1 censors ±1 volume around high-FD.",
    )
    
    censoring.add_argument(
        "--scrub",
        metavar="N",
        type=int,
        dest="scrub",
        default=0,
        help="Minimum contiguous segment length to keep after motion censoring. "
             "If N > 0, continuous segments of kept volumes shorter than N are also "
             "censored. This ensures only sufficiently long data segments remain for "
             "reliable connectivity estimation. Default: 0 (disabled). "
             "Requires --fd-threshold to be set.",
    )
    
    censoring.add_argument(
        "--drop-initial",
        metavar="N",
        type=int,
        dest="drop_initial",
        default=0,
        help="Number of initial volumes to drop (dummy scans). Default: 0.",
    )
    
    # =========================================================================
    # OPTIONAL ARGUMENTS - Preprocessing
    # =========================================================================
    preproc = parser.add_argument_group(
        f'{Colors.BOLD}Preprocessing Options{Colors.END}'
    )
    
    preproc.add_argument(
        "--denoising",
        metavar="STRATEGY",
        choices=["minimal", "csfwm_6p", "csfwm_12p", "gs_csfwm_6p", "gs_csfwm_12p", "csfwm_24p", "compcor_6p", "simpleGSR", "scrubbing5"],
        help="Use a predefined denoising strategy. Choices: "
             "%(choices)s. "
             "Note: 'scrubbing5' includes FD censoring (0.5 cm) and segment filtering "
             "(min 5 volumes) and cannot be combined with --fd-threshold or --scrub. "
             "See DENOISING STRATEGIES section below for details. "
             "Can also be specified in config file.",
    )
    
    # =========================================================================
    # OPTIONAL ARGUMENTS - Analysis Method & Atlas
    # =========================================================================
    analysis_opts = parser.add_argument_group(
        f'{Colors.BOLD}Analysis Options{Colors.END}',
        "Connectivity method and atlas selection (applies to both participant and group level)."
    )
    
    analysis_opts.add_argument(
        "--atlas",
        metavar="ATLAS",
        help="Atlas for ROI-to-ROI connectivity. "
             "Available: schaefer2018n100, schaefer2018n200, aal, harvardoxford. "
             "Default: schaefer2018n100.",
    )
    
    analysis_opts.add_argument(
        "--method",
        metavar="METHOD",
        choices=["seedToVoxel", "roiToVoxel", "seedToSeed", "roiToRoi"],
        help="Connectivity method. "
             "Choices: %(choices)s. Default: roiToRoi.",
    )
    
    # =========================================================================
    # OPTIONAL ARGUMENTS - Group Analysis
    # =========================================================================
    group_opts = parser.add_argument_group(
        f'{Colors.BOLD}Group Analysis Options{Colors.END}',
        "Options specific to group-level tangent space connectivity analysis."
    )
    
    group_opts.add_argument(
        "--participant-derivatives",
        metavar="PATH",
        dest="participant_derivatives",
        type=Path,
        help="Path to participant-level connectomix outputs. "
             "Required for group analysis if not in default location.",
    )
    
    return parser


def parse_derivatives_arg(derivatives_list: list) -> dict:
    """Parse derivatives arguments into dictionary.
    
    Args:
        derivatives_list: List of strings in format "name=path"
    
    Returns:
        Dictionary mapping derivative names to paths
    
    Raises:
        ValueError: If derivative argument format is invalid
    """
    if not derivatives_list:
        return {}
    
    derivatives_dict = {}
    for derivative_arg in derivatives_list:
        if "=" not in derivative_arg:
            raise ValueError(
                f"Invalid derivatives argument: {derivative_arg}. "
                f"Expected format: name=path (e.g., fmriprep=/path/to/fmriprep)"
            )
        
        name, path = derivative_arg.split("=", 1)
        derivatives_dict[name] = Path(path)
    
    return derivatives_dict
