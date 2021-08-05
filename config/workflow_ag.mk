# Configuration for ABARES wheat-sheep region analysis

VAR=pr
UNITS=${VAR}=mm/day

SHAPEFILE=/home/599/dbi599/unseen/shapefiles/wheat_sheep.zip
REGION_NAME=wheat-sheep
SHP_HEADER=region
SPATIAL_AGG=mean

TIME_FREQ=A-DEC
TIME_AGG=mean

IO_OPTIONS=--variables ${VAR} --spatial_coords -44 -11 113 154 --shapefile ${SHAPEFILE} --shp_header ${SHP_HEADER} --combine_shapes --spatial_agg ${SPATIAL_AGG} --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --complete_time_agg_periods --units ${UNITS}

FCST_DATA := $(sort $(wildcard /g/data/xv83/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-199[0,1,2,3]??01/ZARR/atmos_isobaric_daily.zarr.zip))
FCST_METADATA=config/dataset_cafe_daily.yml
FCST_ENSEMBLE_FILE=/g/data/xv83/dbi599/${VAR}_cafe-c5-d60-pX-f6_19900501-19931101_${TIME_FREQ}-${TIME_AGG}_${REGION_NAME}-${SPATIAL_AGG}.zarr.zip

OBS_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-csiro-awap/rain_day_19000101-20201202_cafe-grid.zarr/
OBS_METADATA=config/dataset_awap_daily.yml
OBS_FORECAST_FILE=/g/data/xv83/dbi599/${VAR}_awap_1900-2019_${TIME_FREQ}-${TIME_AGG}_${REGION_NAME}-${SPATIAL_AGG}.zarr.zip

DASK_CONFIG=config/dask_local.yml



