import os, json
import tempfile
import pandas as pd
import numpy as np
import yaml
from connectomix.core.utils.loaders import (load_atlas_data,
                                            load_repetition_time,
                                            load_confounds,
                                            replace_nans_with_mean,
                                            load_config,
                                            load_seed_file,
                                            load_mask)

def test_load_atlas_data():
    atlas_name = "aal"
    maps, labels, coords = load_atlas_data(atlas_name, get_cut_coords=True)
    assert len(labels) == len(coords)
    assert os.path.isfile(maps)


def test_load_repetition_time():
    with tempfile.TemporaryDirectory() as temp_dir:
        json_file = os.path.join(temp_dir, "dummyFile.json")
        sidecar_json = {"RepetitionTime": 42}
        with open(json_file, 'w') as f:
            json.dump(sidecar_json, f, indent=4)
        t_r = load_repetition_time(json_file)
    assert t_r == 42


def test_load_confounds():
    confounds_data = {"col1": np.random.rand(10),
                      "col2": np.random.rand(10)}
    confounds_data = pd.DataFrame(confounds_data)

    config = {}
    config["confounds"] = ["col1"]
    config["ica_aroma"] = False

    with tempfile.TemporaryDirectory() as temp_dir:
        confounds_file = os.path.join(temp_dir, "dummyConfounds.tsv")
        confounds_data.to_csv(confounds_file, sep='\t')

        selected_confounds = load_confounds(confounds_file, config)

    assert selected_confounds.keys() == ["col1"]


def test_replace_nans_with_mean():
    df = pd.DataFrame({"col1": [1, 3, np.nan]})
    new_df = replace_nans_with_mean(df)
    assert new_df["col1"].values[2] == 2


def test_load_config():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {"field1": "dummyValue"}
        config_file = os.path.join(temp_dir, "dummyConfig.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        loaded_config = load_config(config_file)
    assert type(loaded_config) == dict
    assert loaded_config["field1"] == "dummyValue"


def test_load_seed_file():

    with tempfile.TemporaryDirectory() as temp_dir:
        seeds_file = os.path.join(temp_dir, "dummySeeds.tsv")

        coord1 = [np.random.randint(50),
                  np.random.randint(50),
                  np.random.randint(50)]
        coord2 = [np.random.randint(50),
                  np.random.randint(50),
                  np.random.randint(50)]
        df = pd.DataFrame([["label1", *coord1],
                           ["label2", *coord2]])
        df.to_csv(seeds_file,
                  index=False,
                  header=False,
                  sep='\t')

        coords, labels = load_seed_file(seeds_file)
    assert len(labels) == 2
    assert len(coords) == 2
    assert len(coords[0]) == 3

# WIP
# def test_load_mask():
#     from connectomix.tests.tools import generate_bids_dataset, create_nifti_file
#     from connectomix.core.utils.writers import write_dataset_description
#     with tempfile.TemporaryDirectory() as temp_dir:
#         generate_bids_dataset(temp_dir,
#                               1,
#                               ['restingstate'],
#                               fmriprep=True)
#         output_dir = os.path.join(temp_dir, "derivatives", "connectomix")
#         write_dataset_description(output_dir)
#         layout = BIDSLayout(temp_dir, derivatives=output_dir)
#
#         entities = {}
#         mask_img = load_mask(layout, entities)