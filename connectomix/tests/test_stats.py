import numpy as np

from connectomix.core.processing.stats import (compute_z_from_t,
                                               non_parametric_stats)

def test_compute_z_from_t():
    assert 4.968 == np.round(compute_z_from_t(5 ,1000), 3)