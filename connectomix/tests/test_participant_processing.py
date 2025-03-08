import tempfile

from connectomix.core.processing.participant_processing import (extract_timeseries)

def test_extract_timeseries():
    from connectomix.tests.tools import create_nifti_file

    with tempfile.TemporaryDirectory() as temp_dir:
        func_file = create_nifti_file(temp_dir,
                                      "fmri.nii.gz",
                                      shape=[64,64,64,64],
                                      repetition_time=3)
        config = {}
        config["method"] = "roiToRoi"
        config["atlas"] = "aal"
        config["t_r"] = 3
        timeseries, labels = extract_timeseries(func_file, config)

    assert timeseries.shape[0] == 64
    assert timeseries.shape[1] == 21
    assert len(labels) == 116