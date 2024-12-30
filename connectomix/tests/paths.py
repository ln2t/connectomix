import os
import importlib.resources

# ------ Set-up for ds005699 - 4 subjects, two groups, one task (no session or run)
ds = "ds005699"

bids_dir = ""
fmriprep_dir = ""
output_dir = ""

# ----- Test: single subject -----
ds = "ds005625"
bids_dir = os.path.join('/rawdata', ds)
derivatives_dir = os.path.join('/derivatives', ds)
output_dir = os.path.join(derivatives_dir, 'connectomix-roiToVoxel-dev')

example_seeds_for_roiToRoi = importlib.resources.path("connectomix.tests.seeds", "example_seeds_for_roiToRoi.tsv")
example_seeds_for_roiToVoxel = importlib.resources.path("connectomix.tests.seeds", "example_seeds_for_roiToVoxel.tsv")

precuneus_L_mask = importlib.resources.path("connectomix.tests.seeds", "AAL_Precuneus_L.nii.gz")
precentral_R_mask = importlib.resources.path("connectomix.tests.seeds", "AAL_Precuneus_L.nii.gz")