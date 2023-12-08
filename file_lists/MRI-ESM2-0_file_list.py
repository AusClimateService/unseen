"""Create MRI-ESM2-0 DCPP file lists"""

import glob
import os

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/MRI/MRI-ESM2-0/dcppA-hindcast"
pr_file = "MRI-ESM2-0_dcppA-hindcast_pr_files.txt"
try:
    os.remove(pr_file)
except OSError:
    pass

tos_file = "MRI-ESM2-0_dcppA-hindcast_tos_files.txt"
try:
    os.remove(tos_file)
except OSError:
    pass

for year in np.arange(1960, 2019 + 1):
    infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/day/pr/gn/*/*.nc")
    infiles1.sort()
    infiles2 = glob.glob(f"{file_dir}/s{year}-r??i1p1f1/day/pr/gn/*/*.nc")
    infiles2.sort()
    infiles = infiles1 + infiles2
    assert len(infiles) == 10, f"year {year} pr does not have 10 files"
    with open(pr_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")

for year in np.arange(1960, 2019 + 1):
    infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/Omon/tos/gn/*/*.nc")
    infiles1.sort()
    infiles2 = glob.glob(f"{file_dir}/s{year}-r??i1p1f1/Omon/tos/gn/*/*.nc")
    infiles2.sort()
    infiles = infiles1 + infiles2
    assert len(infiles) == 10, f"year {year} tos does not have 10 files"
    with open(tos_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")
