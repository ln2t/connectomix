import os

from connectomix.version import __version__

# ------ Set-up for ds005699 - 4 subjects, two groups, one task (no session or run)
ds = "ds005699"

bids_dir = os.path.join("/rawdata", ds)
fmriprep_dir = os.path.join("/derivatives", ds, "fmriprep")
output_dir = os.path.join("/derivatives", ds, f"connectomix-{__version__}-tests")

# ----- Test: single subject -----
# ds = "ds005625"
# bids_dir = os.path.join('/rawdata', ds)
# derivatives_dir = os.path.join('/derivatives', ds)
# output_dir = os.path.join(derivatives_dir, 'connectomix-roiToVoxel-dev')

