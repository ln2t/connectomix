import os
import json
import numpy as np
import nibabel as nib
import pandas as pd

def generate_bids_dataset(bids_dir,
                          num_subjects,
                          task_names,
                          sessions=None,
                          runs=None,
                          fmriprep=False):

    # Create the main BIDS directory structure
    os.makedirs(bids_dir, exist_ok=True)

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
            create_nifti_file(anat_dir, filename, (16, 16, 16))

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
                    create_nifti_file(func_dir, filename, (8, 8, 8, 5), repetition_time=repetition_time, is_func=True)

                    # Create sidecar JSON file for functional data
                    sidecar_json = {
                        "TaskName": task,
                        "RepetitionTime": repetition_time,
                        "MagneticFieldStrength": 3.0
                    }
                    sidecar_path = os.path.join(func_dir, f'{filename}.json')
                    with open(sidecar_path, 'w') as f:
                        json.dump(sidecar_json, f, indent=4)

def create_nifti_file(directory, filename, shape, repetition_time=None, is_func=False):
    from pathlib import Path
    # Generate random data
    data = np.random.random(shape)

    # Create a NIfTI image
    img = nib.Nifti1Image(data, affine=np.eye(4))

    Path(directory).mkdir(exist_ok=True, parents=True)

    # Set the repetition time in the header if provided and if it's functional data
    if repetition_time is not None and is_func:
        img.header.set_zooms((1.0, 1.0, 1.0, repetition_time))
        img.header.set_xyzt_units(xyz=2, t=8)  # Set units for spatial (mm) and time (sec)

    # Save the NIfTI file
    nifti_path = os.path.join(directory, f'{filename}.nii.gz')
    nib.save(img, nifti_path)

    return nifti_path