"""Create CAFE-f6 file lists"""

import os
import glob


pr_file = "CAFE_c5-d60-pX-f6_pr_files.txt"
try:
    os.remove(pr_file)
except OSError:
    pass

tos_file = "CAFE_c5-d60-pX-f6_tos_files.txt"
try:
    os.remove(tos_file)
except OSError:
    pass

infiles_1990s = glob.glob(
    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-199[5,6,7,8,9]*/atmos_isobaric_daily.zarr.zip"
)
infiles_1990s.sort()
infiles_2000s = glob.glob(
    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-2*/atmos_isobaric_daily.zarr.zip"
)
infiles_2000s.sort()
with open(pr_file, "a") as outfile:
    for item in infiles_1990s + infiles_2000s:
        outfile.write(f"{item}\n")

infiles_1990s = glob.glob(
    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-199[5,6,7,8,9]*/ocean_month.zarr.zip"
)
infiles_1990s.sort()
infiles_2000s = glob.glob(
    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-2*/ocean_month.zarr.zip"
)
infiles_2000s.sort()
with open(tos_file, "a") as outfile:
    for item in infiles_1990s + infiles_2000s:
        outfile.write(f"{item}\n")
