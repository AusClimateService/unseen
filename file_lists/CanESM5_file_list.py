"""Create CanESM5 DCPP file lists"""

import glob
import os

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/CCCma/CanESM5/dcppA-hindcast"
pr_file = "CanESM5_dcppA-hindcast_pr_files.txt"
try:
    os.remove(pr_file)
except OSError:
    pass
tos_file = "CanESM5_dcppA-hindcast_tos_files.txt"
try:
    os.remove(tos_file)
except OSError:
    pass

for year in np.arange(1960, 2016 + 1):
    infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p2f1/day/pr/gn/*/*.nc")
    infiles1.sort()
    infiles2 = glob.glob(f"{file_dir}/s{year}-r??i1p2f1/day/pr/gn/*/*.nc")
    infiles2.sort()
    infiles = infiles1 + infiles2
    assert len(infiles) == 20, f"year {year} does not have 20 files"
    with open(pr_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")

for year in np.arange(1960, 2016 + 1):
    infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p2f1/Omon/tos/gn/*/*.nc")
    infiles1.sort()
    infiles2 = glob.glob(f"{file_dir}/s{year}-r1?i1p2f1/Omon/tos/gn/*/*.nc")
    infiles2.sort()
    infiles3 = glob.glob(f"{file_dir}/s{year}-r20i1p2f1/Omon/tos/gn/*/*.nc")
    infiles3.sort()
    infiles = infiles1 + infiles2 + infiles3
    assert len(infiles) == 20, f"year {year} does not have 20 files"
    with open(tos_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")
