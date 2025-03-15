import tempfile
import os

import pandas as pd

from connectomix.core.utils.writers import (write_design_matrix,
                                            write_permutation_dist,
                                            write_contrast_scores,
                                            write_significant_data,
                                            write_default_config_file,
                                            write_copy_of_config,
                                            write_cluster_tables,
                                            write_matrix_plot,
                                            write_connectome_plot,
                                            write_map_at_cut_coords,
                                            write_roi_to_voxel_plot,
                                            write_dataset_description)

def test_write_design_matrix():
    from connectomix.tests.tools import generate_bids_dataset
    from bids import BIDSLayout
    import numpy as np
    with tempfile.TemporaryDirectory() as temp_dir:
            generate_bids_dataset(temp_dir, 1, ['restingstate'])
            output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
            write_dataset_description(output_dir)
            layout = BIDSLayout(temp_dir, derivatives=output_dir)
            num_rows = 20
            data = {
                'Column1': np.random.rand(num_rows),
                'Column2': np.random.rand(num_rows)
            }
            design_matrix = pd.DataFrame(data)
            config = {"subject": "001",
                      "tasks": "restingstate",
                      "spaces": "dummySpace",
                      "method": "seedToVoxel",
                      "analysis_name": "dummyAnalysisName"}
            design_matrix_plot_path = write_design_matrix(layout,
                                                          design_matrix,
                                                          "dummyLabel",
                                                          config)
            assert os.path.isfile(design_matrix_plot_path)


def test_write_permutation_dist():
    from connectomix.tests.tools import generate_bids_dataset
    from bids import BIDSLayout
    import numpy as np
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        layout = BIDSLayout(temp_dir, derivatives=output_dir)

        permutation_dist = np.random.rand(100)
        config = {"subject": "001",
                  "tasks": "restingstate",
                  "spaces": "dummySpace",
                  "method": "seedToVoxel",
                  "analysis_name": "dummyAnalysisName"}
        permutation_dist_path = write_permutation_dist(layout,
                                                         permutation_dist,
                                                         "dummyLabel",
                                                         config)
        assert os.path.isfile(permutation_dist_path)


def test_write_contrast_scores():
    from connectomix.tests.tools import generate_bids_dataset
    from bids import BIDSLayout
    import numpy as np
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        layout = BIDSLayout(temp_dir, derivatives=output_dir)

        contrast_scores = {"p_values": np.random.rand(10,10)}
        random_labels = np.random.rand(10)
        config = {"subject": "001",
                  "tasks": "restingstate",
                  "spaces": "dummySpace",
                  "method": "seedToSeed",
                  "analysis_name": "dummyAnalysisName",
                  "custom_seeds_name": "customSeedName",
                  "connectivity_kind": "dummyConnectivityKind"}

        plot_path = write_contrast_scores(layout,
                                          contrast_scores,
                                          random_labels,
                                          None,
                                          config)
        assert os.path.isfile(plot_path)


def test_write_significant_data():
    from connectomix.tests.tools import generate_bids_dataset
    from bids import BIDSLayout
    import numpy as np
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        layout = BIDSLayout(temp_dir, derivatives=output_dir)

        significant_data = {"fwe": np.random.rand(10,10)}
        random_labels = np.random.rand(10)
        coords = [np.random.rand(3) for _ in random_labels]
        config = {"subject": "001",
                  "tasks": "restingstate",
                  "spaces": "dummySpace",
                  "method": "seedToSeed",
                  "analysis_name": "dummyAnalysisName",
                  "custom_seeds_name": "customSeedName",
                  "connectivity_kind": "dummyConnectivityKind",
                  "thresholding_strategies": ["fwe"],
                  "fwe_alpha": 0.05}

        significant_data_path_list = write_significant_data(layout,
                                                      significant_data,
                                                      random_labels,
                                                      coords,
                                                      config)

        for file_path in significant_data_path_list:
            assert os.path.isfile(file_path)


def test_write_default_config_file():
    from connectomix.tests.tools import generate_bids_dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        yaml_file = write_default_config_file(temp_dir,
                                    {"connectomix": output_dir},
                                         "group")
        assert os.path.isfile(yaml_file)


def test_write_copy_of_config():
    from connectomix.tests.tools import generate_bids_dataset
    from bids import BIDSLayout
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        layout = BIDSLayout(temp_dir, derivatives=output_dir)
        path = write_copy_of_config(layout, {})
        assert os.path.isfile(path)


def test_write_cluster_tables():
    from connectomix.tests.tools import generate_bids_dataset
    from bids import BIDSLayout
    import numpy as np
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        layout = BIDSLayout(temp_dir, derivatives=output_dir)
        data = {
            'Column1': np.random.rand(10),
            'Column2': np.random.rand(10)
        }
        fake_clusters = pd.DataFrame(data)
        cluster_tables =  {"fwe": fake_clusters}
        config = {"subject": "001",
                  "tasks": "restingstate",
                  "spaces": "dummySpace",
                  "method": "seedToSeed",
                  "analysis_name": "dummyAnalysisName",
                  "custom_seeds_name": "customSeedName",
                  "connectivity_kind": "dummyConnectivityKind",
                  "thresholding_strategies": ["fwe"],
                  "fwe_alpha": 0.05}
        path = write_cluster_tables(layout, cluster_tables, None, config)
        assert os.path.isfile(path)


def test_write_matrix_plot():
    import numpy as np
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "dummyName.png")
        matrix = np.random.rand(10,10)
        write_matrix_plot(matrix, path, None)
        assert os.path.isfile(path)


def test_write_connectome_plot():
    import numpy as np
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "dummyName.png")
        matrix = np.random.rand(10,10)
        coords = [np.random.rand(3) for _ in np.arange(10)]
        write_connectome_plot(matrix, path, coords)
        assert os.path.isfile(path)


def test_write_map_at_cut_coords():
    from connectomix.tests.tools import create_nifti_file
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "dummyName.png")
        config = {"method": None}
        create_nifti_file(temp_dir, "dummyImage", [10,10,10])
        write_map_at_cut_coords(os.path.join(temp_dir, "dummyImage.nii.gz"),
                                path,
                                [2,2,2],
                                config)
        assert os.path.isfile(path)


def test_write_roi_to_voxel_plot():
    from connectomix.tests.tools import generate_bids_dataset, create_nifti_file
    from bids import BIDSLayout
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
        write_dataset_description(output_dir)
        layout = BIDSLayout(temp_dir, derivatives=output_dir)
        create_nifti_file(temp_dir, "dummyImage", [10, 10, 10])
        roi_to_voxel_img = os.path.join(temp_dir, "dummyImage.nii.gz")
        entities = {"subject": "001",
                    "task": "restingstate",
                    "space": "dummySpace"}
        config = {"method": "seedToVoxel",
                  "analysis_name": "dummyAnalysisName"}
        roi_to_voxel_plot_path = write_roi_to_voxel_plot(layout,
                                                         roi_to_voxel_img,
                                                         entities,
                                                         "dummyLabel",
                                                         config)

        assert os.path.isfile(roi_to_voxel_plot_path)


def test_write_dataset_description():
    from pathlib import Path
    with tempfile.TemporaryDirectory() as temp_dir:
        write_dataset_description(temp_dir)
        path = Path(temp_dir) / "dataset_description.json"

        assert os.path.isfile(path)
