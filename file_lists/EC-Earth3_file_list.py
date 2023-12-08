"""Create EC-Earth3 DCPP file lists"""

import os

import glob
import numpy as np


file_dir = (
    "/g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppA-hindcast"
)
pr_file = "EC-Earth3_dcppA-hindcast_pr_files.txt"
try:
    os.remove(pr_file)
except OSError:
    pass

tos_file = "EC-Earth3_dcppA-hindcast_tos_files.txt"
try:
    os.remove(tos_file)
except OSError:
    pass

# From 2005-2010 there are no r1i4p1f1 files, so i4 has been left out
# 2018 is left out because it does not have i2 files
for year in np.arange(1960, 2017 + 1):
    infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/day/pr/gr/v202012??/*.nc")
    infiles1.sort()
    infiles2 = glob.glob(f"{file_dir}/s{year}-r10i1p1f1/day/pr/gr/v202012??/*.nc")
    infiles2.sort()
    infiles3 = glob.glob(f"{file_dir}/s{year}-r?i2p1f1/day/pr/gr/*/*.nc")
    infiles3.sort()
    infiles4 = glob.glob(f"{file_dir}/s{year}-r10i2p1f1/day/pr/gr/*/*.nc")
    infiles4.sort()
    infiles = infiles1 + infiles2 + infiles3 + infiles4
    assert len(infiles) == 165, f"year {year} pr does not have 165 (15*11) files"
    with open(pr_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")

for year in np.arange(1960, 2017 + 1):
    infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/Omon/tos/gn/v202012??/*.nc")
    infiles1.sort()
    infiles2 = glob.glob(f"{file_dir}/s{year}-r10i1p1f1/Omon/tos/gn/v202012??/*.nc")
    infiles2.sort()
    infiles3 = glob.glob(f"{file_dir}/s{year}-r?i2p1f1/Omon/tos/gn/v202009*/*.nc")
    infiles3.sort()
    infiles4 = glob.glob(f"{file_dir}/s{year}-r10i2p1f1/Omon/tos/gn/*/*.nc")
    infiles4.sort()
    infiles = infiles1 + infiles2 + infiles3 + infiles4
    ninfiles = len(infiles)
    assert ninfiles == 165, f"year {year} tos has {ninfiles} not 165 (15*11) files"
    with open(tos_file, "a") as outfile:
        for item in infiles:
            outfile.write(f"{item}\n")
