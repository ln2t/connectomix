import os
from bids import BIDSLayout
import tempfile

from connectomix.core.utils.setup import (setup_config,
                                          setup_config_bids,
                                          setup_config_stats)

def test_setup_config():
    from connectomix.tests.tools import generate_bids_dataset, create_synthetic_fmriprep_outputs

    with tempfile.TemporaryDirectory() as bids_dir:
        generate_bids_dataset(bids_dir, 1, task_names=["restingstate"])
        fmriprep_dir = os.path.join(bids_dir, "fmriprep")
        create_synthetic_fmriprep_outputs(fmriprep_dir, 1, task_names=["restingstate"])
        layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir)
        config = setup_config(layout, {}, "participant")
        assert type(config) == dict
        assert config["method"] == "roiToRoi"
        assert config["atlas"] == "schaeffer100"


def test_setup_config_bids():
    from connectomix.tests.tools import generate_bids_dataset, create_synthetic_fmriprep_outputs

    with tempfile.TemporaryDirectory() as bids_dir:
        generate_bids_dataset(bids_dir, 1, task_names=["restingstate"])
        fmriprep_dir = os.path.join(bids_dir, "fmriprep")
        create_synthetic_fmriprep_outputs(fmriprep_dir, 1, task_names=["restingstate"])
        layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir)
        config = setup_config_bids({}, layout, "participant")
        assert type(config) == dict


def test_setup_config_stats():
    config = setup_config_stats({"method": "roiToVoxel"})
    assert config["cluster_forming_alpha"] == "0.01"
    assert config["n_permutations"] == 20