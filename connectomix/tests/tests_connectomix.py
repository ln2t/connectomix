import unittest
import os
import tempfile
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from nibabel import Nifti1Image
from ..connectomix import load_config, select_confounds, get_repetition_time, resample_to_reference, autonomous_mode, main

class TestConnectomix(unittest.TestCase):

    def setUp(self):
        # Set up temporary directories for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bids_dir = Path(self.temp_dir.name) / 'bids'
        self.derivatives_dir = Path(self.bids_dir) / 'derivatives'
        self.fmriprep_dir = self.derivatives_dir / 'fmriprep'
        self.config_file = Path(self.temp_dir.name) / 'config.yaml'
        self.confounds_file = Path(self.temp_dir.name) / 'confounds.tsv'
        os.makedirs(self.bids_dir, exist_ok=True)
        os.makedirs(self.derivatives_dir, exist_ok=True)
        os.makedirs(self.fmriprep_dir, exist_ok=True)

        # Create dummy confounds file
        confounds_data = pd.DataFrame({
            'trans_x': [0.1, 0.2],
            'trans_y': [0.0, 0.1],
            'trans_z': [0.1, 0.3],
            'rot_x': [0.0, 0.1],
            'rot_y': [0.2, 0.3],
            'rot_z': [0.1, 0.0],
            'global_signal': [0.5, 0.6]
        })
        confounds_data.to_csv(self.confounds_file, sep='\t', index=False)
        
        # Create dummy dataset_description at root of BIDS directory
        bids_description = {
            "Name": "connectomix",
            "BIDSVersion": "1.6.0",
            "PipelineDescription": {
                "Name": "connectomix",
                "Version": 'dummyversion'
            }
        }
        with open(self.bids_dir / "dataset_description.json", 'w') as f:
            json.dump(bids_description, f, indent=4)

        # Create dummy dataset_description for fMRIPrep
        bids_description = {
            "Name": "fMRIPrep",
            "BIDSVersion": "1.6.0",
            "PipelineDescription": {
                "Name": "fMRIPrep",
                "Version": 'dummyversion'
            }
        }
        with open(self.fmriprep_dir / "dataset_description.json", 'w') as f:
            json.dump(bids_description, f, indent=4)


    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_load_config(self):
        # Test YAML loading
        config_dict = {"key": "value"}
        with open(self.config_file, 'w') as file:
            yaml.dump(config_dict, file)
        config = load_config(self.config_file)
        self.assertEqual(config, config_dict)

        # Test config file not found
        non_existent_file = "non_existent_file.txt"
        self.assertFalse(os.path.exists(non_existent_file))        
        with self.assertRaises(FileNotFoundError):
            load_config(non_existent_file)

        # Test JSON loading
        json_file = Path(self.temp_dir.name) / 'config.json'
        with open(json_file, 'w') as file:
            json.dump(config_dict, file)
        config = load_config(json_file)
        self.assertEqual(config, config_dict)

        # Test invalid file type
        invalid_file = Path(self.temp_dir.name) / 'config.txt'
        with open(invalid_file, 'w') as file:
            file.write("Invalid file")
        with self.assertRaises(TypeError):
            load_config(invalid_file)

    def test_select_confounds(self):
        config = {"confound_columns": ['trans_x', 'trans_y', 'trans_z']}
        confounds = select_confounds(self.confounds_file, config)
        self.assertEqual(confounds.shape[1], 3)
        self.assertListEqual(list(confounds.columns), ['trans_x', 'trans_y', 'trans_z'])

    def test_get_repetition_time(self):
        # Test reading repetition time from JSON
        json_file = Path(self.temp_dir.name) / 'test.json'
        metadata = {"RepetitionTime": 2.0}
        with open(json_file, 'w') as file:
            json.dump(metadata, file)
        repetition_time = get_repetition_time(json_file)
        self.assertEqual(repetition_time, 2.0)

    # def test_resample_to_reference(self):
    #     # Mocking nilearn's resample_img and check_affines_match
    #     func_files = [str(Path(self.temp_dir.name) / 'sub-01_task-rest_bold.nii.gz')]
    #     reference_img = MagicMock(spec=Nifti1Image)

    #     layout_mock = MagicMock()
    #     layout_mock.derivatives['connectomix'].parse_file_entities.return_value = {'subject': '01'}
    #     layout_mock.derivatives['connectomix'].build_path.return_value = str(self.temp_dir.name) + '/resampled.nii.gz'

    #     with patch('nilearn.image.load_img') as load_img_mock:
    #         load_img_mock.return_value = reference_img
    #         with patch('nilearn.image.resample_img') as resample_img_mock:
    #             resample_img_mock.return_value = reference_img
    #             resampled_files = resample_to_reference(layout_mock, func_files, reference_img)

    #     self.assertTrue(os.path.isfile(resampled_files[0]))

    # def test_extract_timeseries_atlas_method(self):
    #     # Placeholder: test timeseries extraction using atlas method
    #     config = {
    #         'method': 'atlas',
    #         'method_options': {'n_rois': 100, 'high_pass': 0.01, 'low_pass': 0.1},
    #         'confound_columns': ['trans_x', 'trans_y', 'trans_z']
    #     }
    #     t_r = 2.0
    #     func_file = 'dummy_func.nii.gz'
    #     # Mock necessary objects and methods here
    #     # Add actual assertions when the method is implemented

    # def test_extract_timeseries_seeds_method(self):
    #     # Placeholder: test timeseries extraction using seeds method
    #     config = {
    #         'method': 'seeds',
    #         'method_options': {'radius': 5, 'seeds_file': 'dummy_seeds.tsv'},
    #         'confound_columns': ['trans_x', 'trans_y', 'trans_z']
    #     }
    #     t_r = 2.0
    #     func_file = 'dummy_func.nii.gz'
    #     # Mock necessary objects and methods here
    #     # Add actual assertions when the method is implemented

    # def test_compute_canica_components(self):
    #     # Placeholder: test canica component computation
    #     func_filenames = ['func1.nii.gz', 'func2.nii.gz']
    #     layout = MagicMock()
    #     entities = {}
    #     options = {}
    #     # Mock necessary objects and methods here
    #     # Add actual assertions when the method is implemented

    # def test_generate_permuted_null_distributions(self):
    #     # Placeholder: test permuted null distribution generation
    #     group1_data = np.random.rand(10, 20, 20)
    #     group2_data = np.random.rand(10, 20, 20)
    #     config = {'n_permutations': 100}
    #     layout = MagicMock()
    #     entities = {}
    #     # Mock necessary objects and methods here
    #     # Add actual assertions when the method is implemented

    # def test_generate_group_matrix_plots(self):
    #     # Placeholder: test group matrix plot generation
    #     t_stats = np.random.rand(10, 10)
    #     uncorr_mask = np.random.rand(10, 10) > 0.5
    #     fdr_mask = np.random.rand(10, 10) > 0.5
    #     perm_mask = np.random.rand(10, 10) > 0.5
    #     config = {'uncorrected_alpha': 0.05, 'fdr_alpha': 0.05, 'fwe_alpha': 0.05}
    #     layout = MagicMock()
    #     entities = {}
    #     labels = ['ROI1', 'ROI2']
    #     # Mock necessary objects and methods here
    #     # Add actual assertions when the method is implemented

    def test_autonomous_mode(self):
        # Test autonomous mode
        os.chdir(self.bids_dir)
        with self.assertRaises(FileNotFoundError):
            autonomous_mode()
            
    # def test_main_participant(self):
    #     # Test main function for participant-level analysis
    #     args = ['bids_dir', 'derivatives_dir', 'participant', '--fmriprep_dir', 'fmriprep_dir']
    #     with patch('sys.argv', args):
    #         with patch('builtins.print') as mocked_print:
    #             main()
    #             self.assertTrue(mocked_print.called)

    # def test_main_group(self):
    #     # Test main function for group-level analysis
    #     args = ['bids_dir', 'derivatives_dir', 'group', '--fmriprep_dir', 'fmriprep_dir']
    #     with patch('sys.argv', args):
    #         with patch('builtins.print') as mocked_print:
    #             main()
    #             self.assertTrue(mocked_print.called)


if __name__ == '__main__':
    unittest.main()
