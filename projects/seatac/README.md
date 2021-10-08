Code for the following paper:

Risbey JS, Irving DB, Squire DT, Matear RJ, Monselesan DP, Richardson D & Tozer CR (submitted).
On the role of weather and sampling in assessing a record-breaking heat extreme.
*Environmental Research Letters*.

The forecast ensemble file was created by running the `Makefile` in the root directory of this repository:
```bash
$ make process-forecast CONFIG=config_seatac.mk
```

The `projects/seatac/Makefile` then defines the rules to use that ensemble file to
make most of the figures in the paper.
- Figure 3: `make plot-historgram CONFIG=config_seatac.mk`
- Figure 5: `make plot-reanalysis-hot-day CONFIG=config_seatac.mk`
- Figure 6: `make plot-model-hot-day CONFIG=config_seatac.mk`
- Figure 8: `make plot-sample-size-dist CONFIG=config_seatac.mk`
- Figure 10: `make plot-likelihoods CONFIG=config_seatac.mk`
- Figure 11: `make plot-return-periods CONFIG=config_seatac.mk`

Each output image file has the command history embedded in the image metadata.
It can be viewed by installing [exiftool](https://exiftool.org) (e.g. `$ conda install exiftool`)
and then running the following at the command line:
```bash
$ exiftool path/to/image.png
```



  
