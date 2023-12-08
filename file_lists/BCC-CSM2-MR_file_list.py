"""Create BCC-CSM2-MR DCPP file lists"""

import os
import glob

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/BCC/BCC-CSM2-MR/dcppA-hindcast"
pr_file = "BCC-CSM2-MR_dcppA-hindcast_pr_files.txt"
try:
    os.remove(pr_file)
except OSError:
    pass

tos_file = "BCC-CSM2-MR_dcppA-hindcast_tos_files.txt"
try:
    os.remove(tos_file)
except OSError:
    pass

for year in np.arange(1961, 2014 + 1):
    infiles = glob.glob(f"{file_dir}/s{year}-r*i1p1f1/day/pr/gn/*/*.nc")
    infiles.sort()
    assert len(infiles) == 8, f"year {year} pr does not have 8 files"
    with open(pr_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")

for year in np.arange(1961, 2014 + 1):
    infiles = glob.glob(f"{file_dir}/s{year}-r*i1p1f1/Omon/tos/gn/*/*.nc")
    infiles.sort()
    assert len(infiles) == 8, f"year {year} tos does not have 8 files"
    with open(tos_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")
