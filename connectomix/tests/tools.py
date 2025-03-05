import os
import json, yaml
import numpy as np
import nibabel as nib
import pandas as pd
from bids import BIDSLayout

def generate_bids_dataset(bids_dir,
                          num_subjects,
                          task_names,
                          sessions=None,
                          runs=None):

    # Create the main BIDS directory structure
    os.makedirs(bids_dir, exist_ok=True)

    # Set anat shape
    anat_shape = (16, 16, 16)
    func_shape = (8, 8, 8, 40)

    # Create dataset description file
    dataset_description = {
        "Name": "Synthetic MRI and fMRI Dataset",
        "BIDSVersion": "1.6.0",
        "License": "CC0"
    }
    with open(os.path.join(bids_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, indent=4)

    # Prepare participants data
    participant_data = {
        "participant_id": [f'sub-{sub_id:02d}' for sub_id in range(1, num_subjects + 1)],
        "score": np.random.rand(num_subjects),
        "group": np.random.choice(['group1', 'group2'], num_subjects)
    }
    participants_df = pd.DataFrame(participant_data)
    participants_path = os.path.join(bids_dir, 'participants.tsv')
    participants_df.to_csv(participants_path, sep='\t', index=False)

    # Define repetition time
    repetition_time = 2.0

    # Loop over subjects
    for sub_id in range(1, num_subjects + 1):
        sub_dir = os.path.join(bids_dir, f'sub-{sub_id:02d}')
        os.makedirs(sub_dir, exist_ok=True)

        # Determine session structure
        if sessions is not None:
            ses_dirs = [os.path.join(sub_dir, f'ses-{ses_id:02d}') for ses_id in sessions]
            for ses_dir in ses_dirs:
                os.makedirs(ses_dir, exist_ok=True)
        else:
            ses_dirs = [sub_dir]

        for ses_dir in ses_dirs:
            # Create anatomical data
            anat_dir = os.path.join(ses_dir, 'anat')
            os.makedirs(anat_dir, exist_ok=True)
            filename = f'sub-{sub_id:02d}'
            if sessions is not None:
                ses_id = int(ses_dir.split('-')[-1])  # Extract session ID from directory name
                filename += f'_ses-{ses_id:02d}'
            filename += '_T1w'
            create_nifti_file(anat_dir, filename, anat_shape)

            # Create sidecar JSON file for anatomical data
            anat_json = {
                "MagneticFieldStrength": 3.0,
                "Manufacturer": "SyntheticScanner Inc.",
                "ManufacturersModelName": "SyntheticScanner 3T"
            }
            anat_json_path = os.path.join(anat_dir, f'{filename}.json')
            with open(anat_json_path, 'w') as f:
                json.dump(anat_json, f, indent=4)

            # Loop over tasks
            for task in task_names:
                func_dir = os.path.join(ses_dir, 'func')
                os.makedirs(func_dir, exist_ok=True)

                # Determine run structure
                if runs is not None:
                    run_ids = runs
                else:
                    run_ids = [None]

                for run_id in run_ids:
                    filename = f'sub-{sub_id:02d}'
                    if sessions is not None:
                        ses_id = int(ses_dir.split('-')[-1])  # Extract session ID from directory name
                        filename += f'_ses-{ses_id:02d}'
                    filename += f'_task-{task}'
                    if run_id is not None:
                        filename += f'_run-{run_id:02d}'
                    filename += '_bold'
                    create_nifti_file(func_dir, filename, func_shape, repetition_time=repetition_time)

                    # Create sidecar JSON file for functional data
                    sidecar_json = {
                        "TaskName": task,
                        "RepetitionTime": repetition_time,
                        "MagneticFieldStrength": 3.0
                    }
                    sidecar_path = os.path.join(func_dir, f'{filename}.json')
                    with open(sidecar_path, 'w') as f:
                        json.dump(sidecar_json, f, indent=4)


def create_synthetic_fmriprep_outputs(output_dir, num_subjects, task_names, sessions=None, runs=None):
    # Create the main fMRIPrep directory structure
    fmriprep_dir = os.path.join(output_dir, 'fmriprep')
    os.makedirs(fmriprep_dir, exist_ok=True)

    # Create dataset description file
    dataset_description = {
        "Name": "Synthetic fMRIPrep outputs",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "License": "CC0",
        "GeneratedBy": [
            {
                "Name": "fMRIPrep",
                "Version": "xx.yy.zz"
            }
        ]
    }
    with open(os.path.join(fmriprep_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, indent=4)

    anat_shape = (16, 16, 16)
    func_shape = (8, 8, 8, 50)

    # Loop over subjects
    for sub_id in range(1, num_subjects + 1):
        sub_dir = os.path.join(fmriprep_dir, f'sub-{sub_id:02d}')
        os.makedirs(sub_dir, exist_ok=True)

        # Determine session structure
        if sessions is not None:
            ses_dirs = [os.path.join(sub_dir, f'ses-{ses_id:02d}') for ses_id in sessions]
            for ses_dir in ses_dirs:
                os.makedirs(ses_dir, exist_ok=True)
        else:
            ses_dirs = [sub_dir]

        for ses_dir in ses_dirs:
            # Create anatomical outputs
            anat_dir = os.path.join(ses_dir, 'anat')
            os.makedirs(anat_dir, exist_ok=True)
            filename = f'sub-{sub_id:02d}'
            if sessions is not None:
                ses_id = int(ses_dir.split('-')[-1])  # Extract session ID from directory name
                filename += f'_ses-{ses_id:02d}'
            filename += '_desc-preproc_T1w'
            create_nifti_file(anat_dir, filename, anat_shape)

            # Create sidecar JSON file for anatomical data
            anat_json = {
                "MagneticFieldStrength": 3.0,
                "Manufacturer": "SyntheticScanner Inc.",
                "ManufacturersModelName": "SyntheticScanner 3T"
            }
            anat_json_path = os.path.join(anat_dir, f'{filename}.json')
            with open(anat_json_path, 'w') as f:
                json.dump(anat_json, f, indent=4)

            # Loop over tasks
            for task in task_names:
                func_dir = os.path.join(ses_dir, 'func')
                os.makedirs(func_dir, exist_ok=True)

                # Determine run structure
                if runs is not None:
                    run_ids = runs
                else:
                    run_ids = [None]

                for run_id in run_ids:
                    filename = f'sub-{sub_id:02d}'
                    if sessions is not None:
                        ses_id = int(ses_dir.split('-')[-1])  # Extract session ID from directory name
                        filename += f'_ses-{ses_id:02d}'
                    filename += f'_task-{task}'
                    filename += '_space-MNI152NLin6Asym'
                    if run_id is not None:
                        filename += f'_run-{run_id:02d}'
                    filename += '_desc-preproc_bold'
                    create_nifti_file(func_dir, filename, func_shape, repetition_time=2.0)

                    # Create sidecar JSON file for functional data
                    sidecar_json = {
                        "TaskName": task,
                        "RepetitionTime": 2.0,
                        "MagneticFieldStrength": 3.0
                    }
                    sidecar_path = os.path.join(func_dir, f'{filename}.json')
                    with open(sidecar_path, 'w') as f:
                        json.dump(sidecar_json, f, indent=4)

                    # Create mask for functional data
                    mask_filename = filename.replace('_bold', '_mask').replace('-preproc', '-brain')
                    create_nifti_file(func_dir, mask_filename, func_shape[:-1], is_mask=True)

                    # Create confounds file
                    confounds_filename = filename.replace('_space-MNI152NLin6Asym', '').replace('_desc-preproc', '').replace('_bold', '_desc-confounds_timeseries')
                    create_confounds_file(func_dir, confounds_filename, func_shape[-1])


def create_nifti_file(directory, filename, shape, repetition_time=None, is_mask=False):
    from pathlib import Path

    # Generate random data
    if is_mask:
        data = np.random.choice([0.0, 1.0], size=shape)  # Binary mask
    else:
        data = np.random.random(shape)

    # Create a NIfTI image
    img = nib.Nifti1Image(data, affine=np.eye(4))

    Path(directory).mkdir(exist_ok=True, parents=True)

    if len(shape) == 3:
        img.header.set_zooms((1.0, 1.0, 1.0))
        img.header.set_xyzt_units(xyz=2)  # Set units for spatial (mm) and time (sec)
    elif len(shape) == 4 and repetition_time is not None:
        img.header.set_zooms((1.0, 1.0, 1.0, repetition_time))
        img.header.set_xyzt_units(xyz=2, t=8)  # Set units for spatial (mm) and time (sec)
    else:
        raise ValueError("Set repetition time when shape is of length 4.")

    # Save the NIfTI file
    nifti_path = os.path.join(directory, f'{filename}.nii.gz')
    nib.save(img, nifti_path)

    return nifti_path


def create_confounds_file(directory, filename, num_timepoints):
    # Define some common confounds
    confounds = {
        'csf_wm': np.random.rand(num_timepoints),
        'trans_x': np.random.rand(num_timepoints),
        'trans_y': np.random.rand(num_timepoints),
        'trans_z': np.random.rand(num_timepoints),
        'rot_x': np.random.rand(num_timepoints),
        'rot_y': np.random.rand(num_timepoints),
        'rot_z': np.random.rand(num_timepoints)
    }

    # Create a DataFrame
    confounds_df = pd.DataFrame(confounds)

    # Save the confounds file
    confounds_path = os.path.join(directory, f'{filename}.tsv')
    confounds_df.to_csv(confounds_path, sep='\t', index=False)


def create_synthetic_seeds_file(seeds_path, num_seeds=2):
    coord_max = 8
    x_max = coord_max
    y_max = coord_max
    z_max = coord_max

    seed_df = pd.DataFrame(columns=["seed_label", "x", "y", "z"])

    for seed_id in range(1, num_seeds + 1):
        seed_data = [f"seed{seed_id:02d}",
                     np.random.randint(x_max),
                     np.random.randint(y_max),
                     np.random.randint(z_max)]
        seed_df.loc[len(seed_df)] = seed_data

    seed_df.to_csv(seeds_path, sep='\t', index=False, header=False)


def create_synthetic_config_file(bids_dir, derivative, level, config_path, method, seeds_path=None):
    from connectomix.core.utils.setup import setup_config
    layout = BIDSLayout(bids_dir, derivatives=derivative)
    config = setup_config(layout, {}, level)
    config["seeds_file"] = seeds_path
    config["method"] = method
    config["radius"] = 0
    config["spaces"] = ["MNI152NLin6Asym"]
    with open(config_path, "w") as f:
        yaml.dump(config, f)