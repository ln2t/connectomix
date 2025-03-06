import shutil
import tempfile
import os
from argparse import Namespace
from connectomix.core.core import (main)

def test_main_participant():
    num_subjects = 1
    task_names = ['restingstate']

    from connectomix.tests.tools import (generate_bids_dataset,
                                         create_synthetic_fmriprep_outputs,
                                         create_synthetic_seeds_file,
                                         create_synthetic_config_file)
    with tempfile.TemporaryDirectory() as temp_dir:
        bids_dir = os.path.join(temp_dir, "rawdata")
        output_dir = os.path.join(temp_dir, "derivatives/connectomix")
        fmriprep_dir = os.path.join(temp_dir, "derivatives/fmriprep")
        generate_bids_dataset(bids_dir,
                              num_subjects,
                              task_names)

        create_synthetic_fmriprep_outputs(fmriprep_dir,
                                          num_subjects,
                                          task_names)

        seeds_path = os.path.join(temp_dir, "seeds.tsv")
        create_synthetic_seeds_file(seeds_path, num_seeds=1)

        config_path = os.path.join(temp_dir, "config.yaml")
        create_synthetic_config_file(bids_dir, fmriprep_dir, "participant", config_path, "seedToVoxel", seeds_path=seeds_path)

        args = Namespace()
        args.bids_dir = bids_dir
        args.output_dir = output_dir
        args.analysis_level = "participant"
        args.derivatives = [f"fmriprep={fmriprep_dir}"]
        args.autonomous = False
        args.session = None
        args.task = None
        args.run = None
        args.config = config_path
        args.participant_label = '01'
        args.helper = False
        main(args)
        assert os.path.isfile(os.path.join(output_dir,
                                           "sub-01",
                                           "sub-01_task-restingstate_space-MNI152NLin6Asym_method-seedToVoxel_seed-seed01_effectSize.svg"))
        assert os.path.isfile(os.path.join(output_dir,
                                           "sub-01",
                                           "sub-01_task-restingstate_space-MNI152NLin6Asym_method-seedToVoxel_seed-seed01_effectSize.nii.gz"))

        create_synthetic_seeds_file(seeds_path, num_seeds=3)
        create_synthetic_config_file(bids_dir, fmriprep_dir, "participant", config_path, "seedToSeed", seeds_path=seeds_path)

        shutil.rmtree(output_dir)
        main(args)
        assert os.path.isfile(os.path.join(output_dir,
                                           "sub-01",
                                           "sub-01_task-restingstate_space-MNI152NLin6Asym_method-seedToSeed_data-correlation.npy"))

def test_main_group():
    num_subjects = 10
    task_names = ['restingstate']

    from connectomix.tests.tools import (generate_bids_dataset,
                                         create_synthetic_fmriprep_outputs,
                                         create_synthetic_seeds_file,
                                         create_synthetic_config_file)
    with tempfile.TemporaryDirectory() as temp_dir:
        bids_dir = os.path.join(temp_dir, "rawdata")
        output_dir = os.path.join(temp_dir, "derivatives/connectomix")
        fmriprep_dir = os.path.join(temp_dir, "derivatives/fmriprep")

        generate_bids_dataset(bids_dir,
                              num_subjects,
                              task_names)

        create_synthetic_fmriprep_outputs(fmriprep_dir,
                                          num_subjects,
                                          task_names)

        seeds_path = os.path.join(temp_dir, "seeds.tsv")
        create_synthetic_seeds_file(seeds_path, num_seeds=1)

        config_path = os.path.join(temp_dir, "config.yaml")
        create_synthetic_config_file(bids_dir, fmriprep_dir, "participant", config_path, "seedToVoxel", seeds_path=seeds_path)

        args = Namespace()
        args.bids_dir = bids_dir
        args.output_dir = output_dir
        args.analysis_level = "participant"
        args.derivatives = [f"fmriprep={fmriprep_dir}"]
        args.autonomous = False
        args.session = None
        args.task = None
        args.run = None
        args.config = config_path
        args.participant_label = ['01', '02']
        args.helper = False
        main(args)
        create_synthetic_config_file(bids_dir, output_dir, "group", config_path, "seedToVoxel", seeds_path=seeds_path)
        args.analysis_level = "group"
        main(args)
        assert os.path.isfile(os.path.join(output_dir,
                                           "group",
                                           "seedToVoxel",
                                           "customName",
                                           "task-restingstate_space-MNI152NLin6Asym_method-seedToVoxel_seed-seed01_analysis-customName_z.nii.gz"))