Code for an analysis of rainfall across the wheatbelt in Australia.

## Background

The Australian Bureau of Agricultural and Resource Economics and Sciences (ABARES)
defines a set of Australian broadacre zones and regions.
(Download shapefiles [here](https://www.agriculture.gov.au/abares/research-topics/surveys/farm-survey-data).)

The last four very dry years across the "wheat-sheep" region
line up really well with the last four times Australia had to import grain
(1994-95, 2002-03, 2006-07, 2019-20; see
[ABC](https://www.abc.net.au/news/rural/2019-05-15/australia-approves-grain-imports/11113320),
[Guardian](https://www.theguardian.com/australia-news/2019/may/15/australia-to-import-wheat-for-first-time-in-12-years-as-drought-eats-into-grain-production)).

## Code

The XXX files were created by running the `Makefile` in the root directory of this repository:

TODO: Add Make commands.

The `projects/ag/Makefile` then defines the rules to use that ensemble file to
make most of the figures in the analysis.
- TODO: List them

Each output image file has the command history embedded in the image metadata.
It can be viewed by installing [exiftool](https://exiftool.org) (e.g. `$ conda install exiftool`)
and then running the following at the command line:
```bash
$ exiftool path/to/image.png
```



  
