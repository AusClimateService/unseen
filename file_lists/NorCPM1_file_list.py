"""Create NorCPM1 DCPP file lists"""

import glob
import os

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/NCC/NorCPM1/dcppA-hindcast"
pr_file = "NorCPM1_dcppA-hindcast_pr_files.txt"
try:
    os.remove(pr_file)
except OSError:
    pass

tos_file = "NorCPM1_dcppA-hindcast_tos_files.txt"
try:
    os.remove(tos_file)
except OSError:
    pass

for year in np.arange(1960, 2018 + 1):
    infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/day/pr/gn/*/*.nc")
    infiles1.sort()
    infiles2 = glob.glob(f"{file_dir}/s{year}-r??i1p1f1/day/pr/gn/*/*.nc")
    infiles2.sort()
    infiles3 = glob.glob(f"{file_dir}/s{year}-r?i2p1f1/day/pr/gn/*/*.nc")
    infiles3.sort()
    infiles4 = glob.glob(f"{file_dir}/s{year}-r??i2p1f1/day/pr/gn/*/*.nc")
    infiles4.sort()
    infiles = infiles1 + infiles2 + infiles3 + infiles4
    assert len(infiles) == 20, f"year {year} does not have 20 files"
    with open(pr_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")

for year in np.arange(1960, 2018 + 1):
    infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/Omon/tos/gn/*/*.nc")
    infiles1.sort()
    infiles2 = glob.glob(f"{file_dir}/s{year}-r??i1p1f1/Omon/tos/gn/*/*.nc")
    infiles2.sort()
    infiles3 = glob.glob(f"{file_dir}/s{year}-r?i2p1f1/Omon/tos/gn/*/*.nc")
    infiles3.sort()
    infiles4 = glob.glob(f"{file_dir}/s{year}-r??i2p1f1/Omon/tos/gn/*/*.nc")
    infiles4.sort()
    infiles = infiles1 + infiles2 + infiles3 + infiles4
    if not len(infiles) == 20:
        print(f"year {year} does not have 20 files")
    with open(tos_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")
