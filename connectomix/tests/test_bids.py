import tempfile
import os

from connectomix.core.utils.bids import (add_new_entities,
                                         setup_bidslayout,
                                         apply_nonbids_filter,
                                         build_output_path,
                                         alpha_value_to_bids_valid_string)

def test_add_new_entities():
    entities = {}
    config = {}
    config["method"] = "seedToVoxel"
    label = "dummyLabel"
    entities = add_new_entities(entities, config, label=label)

    assert entities["suffix"] == "effectSize"


def test_setup_bidslayout():
    from connectomix.tests.tools import generate_bids_dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        layout = setup_bidslayout(temp_dir, os.path.join(temp_dir, 'output'), derivatives=dict())

    assert layout.derivatives.get_pipeline("connectomix")


def test_apply_nonbids_filter():
    files = apply_nonbids_filter("dummyEntity", "dummyValue", ["dummyEntity-dummyValue", "dummyName"])
    assert files == ["dummyEntity-dummyValue"]


def test_build_output_path():
    from connectomix.tests.tools import generate_bids_dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_bids_dataset(temp_dir, 1, ['restingstate'])
        layout = setup_bidslayout(temp_dir, os.path.join(temp_dir, 'output'), derivatives=dict())
        entities = {"subject": "001",
                    "task": "dummyTask",
                    "space": "dummySpace",
                    "method": "seedToVoxel"}
        label = None
        level = "participant"
        config = {"method": "seedToVoxel"}
        output_path = build_output_path(layout, entities, label, level, config)

        assert os.path.isdir(os.path.dirname(output_path))
    assert os.path.basename(output_path) == "sub-001_task-dummyTask_space-dummySpace_method-seedToVoxel_seed-effectSize.nii.gz"


def test_alpha_value_to_bids_valid_string():
    alpha = 0.05
    new_alpha = alpha_value_to_bids_valid_string(alpha)
    assert new_alpha == "0Dot05"