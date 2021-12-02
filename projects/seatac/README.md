Code for the following paper:

Risbey JS, Irving DB, Squire DT, Matear RJ, Monselesan DP, Richardson D & Tozer CR (submitted).
On the role of weather and sampling in assessing a record-breaking heat extreme.
*Environmental Research Letters*.

The `Makefile` defines the rules to make the figures in the paper.
- Figure 2: `make plot-historgram CONFIG=config_seatac.mk`
- Figure 4: `make plot-reanalysis-hot-day CONFIG=config_seatac.mk`
- Figure 5: `make plot-model-hot-day CONFIG=config_seatac.mk`
  (see `projects/seatac/find_hottest_model_day.ipynb` for information about the hottest day)
- Figure 7: `make plot-sample-size-dist CONFIG=config_seatac.mk`
- Figure 8: `make plot-likelihoods CONFIG=config_seatac.mk`
- Figure 9: `make plot-return-periods CONFIG=config_seatac.mk`
- Figure 10: `make plot-annual-max CONFIG=config_seatac.mk`
- Figure 11: `make plot-distribution CONFIG=config_seatac.mk`

Each output image file has the command history embedded in the image metadata.
It can be viewed by installing [exiftool](https://exiftool.org) (e.g. `$ conda install exiftool`)
and then running the following at the command line:
```bash
$ exiftool path/to/image.png
```

The observations were daily maximum temperatures at Seattle Tacoma International Airport
from the GHCNv2 station dataset,
[downloaded](http://climexp.knmi.nl/gdcntmax.cgi?id=someone@somewhere&WMO=USW00024233&STATION=SEATTLE_TACOMA_INTL_AP,_WA&extraargs=)
from the KNMI climate explorer.




  
