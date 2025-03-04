import tempfile
import os
import json
import numpy as np
import pandas as pd

from bids import BIDSLayout

from connectomix.core.utils.tools import (config_helper,
                                          check_affines_match,
                                          img_is_not_empty,
                                          resample_to_reference,
                                          make_parent_dir,
                                          denoise,
                                          find_labels_and_coords,
                                          get_cluster_tables)
from connectomix.tests.tools import create_nifti_file


def test_config_helper():
    result = config_helper({}, "dummyKey", "dummyDefault")
    assert result == "dummyDefault"


def test_check_affines_match():
    from connectomix.tests.tools import create_nifti_file
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nifti_file(temp_dir, "dummyImage1", [10,10,10])
        create_nifti_file(temp_dir, "dummyImage2", [10, 10, 10])
        imgs = [os.path.join(temp_dir, "dummyImage1.nii.gz"),
                os.path.join(temp_dir, "dummyImage2.nii.gz")]

        assert check_affines_match(imgs)


def test_img_is_not_empty():
    from connectomix.tests.tools import create_nifti_file
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nifti_file(temp_dir, "dummyImage", [10, 10, 10])
        assert img_is_not_empty(os.path.join(temp_dir, "dummyImage1.nii.gz"))


def test_resample_to_reference():
    from connectomix.tests.tools import generate_bids_dataset, create_nifti_file
    from connectomix.core.utils.writers import write_dataset_description
    with tempfile.TemporaryDirectory() as temp_dir:

        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        layout = BIDSLayout(temp_dir, derivatives=output_dir)

        img1_dir = os.path.join(output_dir, "sub-001")
        img1_name = "sub-001_task-restingstate_space-dummySpace"

        img2_dir = os.path.join(output_dir, "sub-002")
        img2_name = "sub-002_task-restingstate_space-dummySpace"

        create_nifti_file(img1_dir, img1_name, [10, 10, 10])
        create_nifti_file(img2_dir, img2_name, [10, 10, 10])

        img1_path = os.path.join(img1_dir, img1_name + ".nii.gz")
        img2_path = os.path.join(img2_dir, img2_name + ".nii.gz")

        func_files = [img1_path, img2_path]

        config = {"reference_functional_file": "first_functional_file"}

        resampled_files = resample_to_reference(layout, func_files, config)

        assert len(resampled_files) == len(func_files)


def test_make_parent_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_dir", "dummyFile.xyz")
        make_parent_dir(file_path)

        assert os.path.exists(os.path.join(temp_dir, "test_dir"))


def test_denoise():
    from connectomix.tests.tools import generate_bids_dataset, create_nifti_file
    from connectomix.core.utils.writers import write_dataset_description
    with tempfile.TemporaryDirectory() as temp_dir:

        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        layout = BIDSLayout(temp_dir, derivatives=output_dir)

        img1_dir = os.path.join(output_dir, "sub-001")
        img1_name = "sub-001_task-restingstate_space-dummySpace"

        img2_dir = os.path.join(output_dir, "sub-002")
        img2_name = "sub-002_task-restingstate_space-dummySpace"

        time_span = 34

        shape = [10, 10, 10, time_span]

        img1_path = create_nifti_file(img1_dir,
                          img1_name,
                          shape,
                          repetition_time=3)
        img2_path = create_nifti_file(img2_dir,
                          img2_name,
                          shape,
                          repetition_time=3)

        input_imgs = [img1_path, img2_path]

        config = {"high_pass": 0.01,
                  "low_pass": 0.1,
                  "ica_aroma": False,
                  "confounds": ["rot_x", "rot_y"]}

        # Create sidecar JSON file for functional data
        sidecar_json = {
                        "RepetitionTime": 3
                        }
        sidecar1_path = os.path.join(img1_dir, f'{img1_name}.json')
        sidecar2_path = os.path.join(img2_dir, f'{img2_name}.json')

        confound_files = ["conf1.tsv", "conf2.tsv"]
        confound_files = [os.path.join(temp_dir, file) for file in confound_files]

        for file in confound_files:
            conf = {"rot_x": np.random.rand(time_span),
                    "rot_y": np.random.rand(time_span),
                    "rot_z": np.random.rand(time_span)}
            df_conf = pd.DataFrame(conf)
            df_conf.to_csv(file, sep='\t')

        input_json = [sidecar1_path, sidecar2_path]

        with open(sidecar1_path, 'w') as f:
            json.dump(sidecar_json, f, indent=4)
        with open(sidecar2_path, 'w') as f:
            json.dump(sidecar_json, f, indent=4)

        denoised_paths = denoise(layout,
                                 input_imgs,
                                 confound_files,
                                 input_json,
                                 config)

        assert len(denoised_paths) == len(input_imgs)
        assert os.path.isfile(denoised_paths[0])


def test_find_labels_and_coords():
    config = {"method": "roiToVoxel",
              "roi_masks": {"dummyRoi1": "path1",
                            "dummyRoi2": "path2"}}
    labels, coords = find_labels_and_coords(config)

    assert len(labels) == len(coords)
    assert labels == ["dummyRoi1", "dummyRoi2"]


def test_get_cluster_tables():
    significant_data = {}
    config = {"two_sided_test": True,
              "cluster_forming_alpha": 0.000001}
    with tempfile.TemporaryDirectory() as temp_dir:
        significant_data["fwe"] = create_nifti_file(temp_dir,
                                                    "dummyName",
                                                    [10, 10, 10])
        cluster_table = get_cluster_tables(significant_data,
                                           config)
        assert len(cluster_table) > 0