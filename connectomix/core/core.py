import argparse
import warnings
from pathlib import Path
from bids import BIDSLayout

from connectomix.core.utils.writers import write_default_config_file, write_copy_of_config
from connectomix.core.processing.participant_processing import post_fmriprep_preprocessing


def participant_level_pipeline(bids_dir, output_dir, derivatives, config):
    """
    Main function to run the participant analysis

    Parameters
    ----------
    bids_dir : str or Path
        Path to bids_dir.
    output_dir : str or Path
        Path to connectomix derivatives.
    derivatives : dict
        Paths to data preprocessed with fMRIPrep and, optionally, fmripost-aroma: derivatives["fmriprep"]="/path/to/fmriprep", etc.
    config : dict or str or Path
        Configuration dict or path to configuration file (can be a .json or .yaml or .yml).

    Returns
    -------
    None.

    """
    from connectomix.version import __version__
    from connectomix.core.utils.tools import print_subject
    from connectomix.core.utils.bids import setup_bidslayout
    from connectomix.core.processing.canica import compute_canica_components
    from connectomix.core.processing.participant_processing import participant_analysis
    from connectomix.core.utils.loaders import load_repetition_time
    from connectomix.core.utils.setup import setup_config

    print(f"Running connectomix (Participant-level) version {__version__}")

    layout = setup_bidslayout(bids_dir, output_dir, derivatives)
    write_copy_of_config(layout, config)
    config = setup_config(layout, config, "participant")

    print(f"Selected method for connectivity analysis: {config['method']}")

    denoised_files, json_files = post_fmriprep_preprocessing(layout, config)

    # Compute CanICA components if necessary and store it in the methods options
    config = compute_canica_components(layout, denoised_files, config) if config.get("atlas", None) == "canica" else config

    for (func_file, json_file) in zip(denoised_files, json_files):
        print_subject(layout, func_file)
        config["t_r"] = load_repetition_time(json_file)
        participant_analysis(layout, func_file, config)

    print("Participant-level analysis completed.")


def group_level_pipeline(bids_dir, output_dir, config):
    """
    Main function to launch group-level analysis.

    Parameters
    ----------
    bids_dir : str or Path
        Path to bids_dir.
    output_dir : str or Path
        Path to connectomix derivatives.
    config : dict or str or Path
        Configuration or path to configuration (can be a .json or .yaml or .yml).

    Returns
    -------
    None.

    """
    # Print version information
    from connectomix.version import __version__
    print(f"Running connectomix (Group-level) version {__version__}")

    # Create BIDSLayout with pipeline and other derivatives
    from connectomix.core.utils.bids import setup_bidslayout
    layout = setup_bidslayout(bids_dir, output_dir)

    # Load and backup the configuration file
    from connectomix.core.utils.setup import setup_config
    write_copy_of_config(layout, config)
    config = setup_config(layout, config, "group")

    from connectomix.core.processing.group_processing import group_analysis
    group_analysis(layout, config)

    print("Group-level analysis completed.")


def autonomous_mode(run=False):
    """ Function to automatically guess the analysis paths and settings. """

    current_dir = Path.cwd()

    # Step 1: Find BIDS directory (bids_dir)
    if (current_dir / "dataset_description.json").exists():
        bids_dir = current_dir
    elif (current_dir / "rawdata" / "dataset_description.json").exists():
        bids_dir = current_dir / "rawdata"
    else:
        raise FileNotFoundError(
            "Could not find 'dataset_description.json'. Ensure the current directory or 'rawdata' folder contains it.")

    # Step 2: Find derivatives directory and fMRIPrep directory
    derivatives_dir = current_dir / "derivatives"
    if not derivatives_dir.exists():
        raise FileNotFoundError(
            "The 'derivatives' folder was not found. Ensure the folder exists in the current directory.")

    # Look for the fMRIPrep folder in derivatives
    fmriprep_folders = [f for f in derivatives_dir.iterdir() if f.is_dir() and f.name.lower().startswith("fmriprep")]

    if len(fmriprep_folders) == 1:
        fmriprep_dir = fmriprep_folders[0]
    elif len(fmriprep_folders) > 1:
        raise FileNotFoundError(
            "Multiple 'fMRIPrep' directories found in 'derivatives'. Please resolve this ambiguity.")
    else:
        raise FileNotFoundError("No 'fMRIPrep' directory found in 'derivatives'.")

    # Step 3: Check if a "connectomix" folder already exists in derivatives
    connectomix_folder = [f for f in derivatives_dir.iterdir() if
                          f.is_dir() and f.name.lower().startswith("connectomix")]

    if len(connectomix_folder) == 0:
        # No connectomix folder found, assume participant-level analysis
        connectomix_folder = Path(derivatives_dir) / "connectomix"
        analysis_level = "participant"

    elif len(connectomix_folder) == 1:
        # Connectomix folder exists and is unique, checking if something has already been run at participant-level
        connectomix_folder = connectomix_folder[0]
        layout = BIDSLayout(bids_dir, derivatives=[connectomix_folder])
        if len(layout.derivatives["connectomix"].get_subjects()) == 0:
            print("No participant-level result detected, assuming participant-level analysis")
            analysis_level = "participant"
        else:
            print(
                f"Detected participant-level results for subjects {layout.derivatives['connectomix'].get_subjects()}, assuming group-level analysis")
            analysis_level = "group"

    else:
        raise ValueError(
            f"Several connectomix directories where found ({connectomix_folder}). Please resolve this ambiguity.")

    # Step 4: Call the main function with guessed paths and settings
    if run:
        print("... and now launching the analysis!")
        if analysis_level == "participant":
            participant_level_pipeline(bids_dir, connectomix_folder, {"fmriprep": fmriprep_dir}, {})
        elif analysis_level == "group":
            group_level_pipeline(bids_dir, connectomix_folder, {})
    else:
        if analysis_level == "participant":
            from connectomix.core.utils.setup import create_participant_level_default_config_file
            create_participant_level_default_config_file(bids_dir, connectomix_folder, fmriprep_dir)
        elif analysis_level == "group":
            from connectomix.core.utils.setup import create_group_level_default_config_file
            create_group_level_default_config_file(bids_dir, connectomix_folder)

        cmd = f"python connectomix.py {bids_dir} {connectomix_folder} {analysis_level} --derivatives fmriprep={fmriprep_dir}"
        print(f"Autonomous mode suggests the following command:\n{cmd}")
        print(
            "If you are happy with this configuration, run this command or simply relaunch the autonomous mode add the --run flag.")


def main():
    """
    Main function to launch the software. Ir reads arguments from sys.argv, which is filled automatically when calling the script from command line.

    Returns
    -------
    None.

    """


    # Set warnings to appear only once
    warnings.simplefilter("once")

    parser = argparse.ArgumentParser(
        description="Connectomix: Functional Connectivity from fMRIPrep outputs using BIDS structure")

    # Define the autonomous flag
    parser.add_argument("--autonomous", action="store_true",
                        help="Run the script in autonomous mode, guessing paths and settings.")

    # Define the run flag
    parser.add_argument("--run", action="store_true", help="Run the analysis based on what the autonomous mode found.")

    # Define positional arguments for bids_dir, derivatives_dir, and analysis_level
    parser.add_argument("bids_dir", nargs="?", type=str, help="BIDS root directory containing the dataset.")
    parser.add_argument("output_dir", nargs="?", type=str, help="Directory where to store the outputs.")
    parser.add_argument("analysis_level", nargs="?", choices=["participant", "group"],
                        help="Analysis level: either 'participant' or 'group'.")

    # Define optional arguments that apply to both analysis levels
    parser.add_argument("-d", "--derivatives", nargs="+",
                        help="Specify pre-computed derivatives as 'key=value' pairs (e.g., -d fmriprep=/path/to/fmriprep fmripost-aroma=/path/to/fmripost-aroma).")
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file.")
    parser.add_argument("-p", "--participant_label", type=str, help="Participant label to process (e.g., 'sub-01').")
    parser.add_argument("--helper", help="Helper function to write default configuration files.", action="store_true")

    args = parser.parse_args()

    # Convert the list of "key=value" pairs to a dictionary
    from connectomix.core.utils.loaders import load_derivatives
    derivatives = load_derivatives(args.derivatives)

    # Run autonomous mode if flag is used
    if args.autonomous:
        autonomous_mode(run=args.run)
    else:
        # Set derivatives to default values if unset by user
        derivatives["fmriprep"] = derivatives.get("fmriprep", Path(args.bids_dir) / "derivatives" / "fmriprep")
        derivatives["fmripost-aroma"] = derivatives.get("fmripost-aroma",
                                                        Path(args.bids_dir) / "derivatives" / "fmripost-aroma")

        # First check if only helper function must be called
        if args.helper:
            write_default_config_file(args.bids_dir,
                                      {"connectomix": args.output_dir, "fmriprep": derivatives["fmriprep"]},
                                      args.analysis_level)
        else:

            # Run the appropriate analysis level
            if args.analysis_level == "participant":

                # Check if fMRIPrep directory exists
                if not Path(derivatives["fmriprep"]).exists():
                    raise FileNotFoundError(
                        f"fMRIPrep directory {derivatives['fmriprep']} not found. Use '--derivatives fmriprep=/path/to/fmriprep' to specify path manually.")
                else:
                    participant_level_pipeline(args.bids_dir, args.output_dir, derivatives, args.config)

            elif args.analysis_level == "group":
                group_level_pipeline(args.bids_dir, args.output_dir, args.config)
