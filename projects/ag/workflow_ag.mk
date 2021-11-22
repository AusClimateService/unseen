# Configuration for ABARES wheat-sheep region analysis

VAR=pr
UNITS=${VAR}=mm/day

SHAPEFILE=/home/599/dbi599/unseen/projects/ag/wheat_sheep.zip
REGION=wheat-sheep
SUB_REGION=all
SHP_HEADER=region
SPATIAL_AGG=mean

TIME_FREQ=A-DEC
TIME_AGG=mean

BIAS_METHOD=additive
BASE_PERIOD=1990-01-01 2019-12-31

GENERAL_IO_OPTIONS=--variables ${VAR} --spatial_coords -44 -11 113 154 --units ${UNITS}
TIME_IO_OPTIONS=--time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --complete_time_agg_periods 
SPATIAL_IO_OPTIONS=--shapefile ${SHAPEFILE} --shp_header ${SHP_HEADER} --combine_shapes --spatial_agg ${SPATIAL_AGG}
FCST_IO_OPTIONS=${GENERAL_IO_OPTIONS} 
OBS_IO_OPTIONS=${FCST_IO_OPTIONS} --input_freq M

FCST_DATA := $(sort $(wildcard /g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-*/atmos_isobaric_daily.zarr.zip))
FCST_METADATA=config/dataset_cafe_daily.yml
FCST_CLIMATOLOGY=/g/data/xv83/dbi599/ag/data/${VAR}_cafe-c5-d60-pX-f6_1990-2019_annual-climatology.zarr.zip
FCST_ENSEMBLE_FILE=/g/data/xv83/dbi599/ag/data/${VAR}_cafe-c5-d60-pX-f6_19811101-20201101_${TIME_FREQ}-${TIME_AGG}_${REGION}-${SPATIAL_AGG}.zarr.zip

OBS_DATA=/g/data/ia39/agcd/post-processed/data/agcd_v2_precip_total_cafe-grid_monthly_1900-2020.zarr.zip
OBS_METADATA=config/dataset_agcd_monthly.yml
OBS_PROCESSED_FILE=/g/data/xv83/dbi599/ag/data/${VAR}_agcd_1900-2019_${TIME_FREQ}-${TIME_AGG}_${REGION}-${SPATIAL_AGG}.zarr.zip

FCST_BIAS_FILE=/g/data/xv83/dbi599/ag/data/${VAR}_cafe-c5-d60-pX-f6_19900501-20191101_${TIME_FREQ}-${TIME_AGG}_${REGION}-${SPATIAL_AGG}_bias-corrected-agcd-${BIAS_METHOD}.zarr.zip
SIMILARITY_FILE=/g/data/xv83/dbi599/ag/data/ks-test_${VAR}_cafe-c5-d60-pX-f6_19900501-20191101_${TIME_FREQ}-${TIME_AGG}_${REGION}-${SPATIAL_AGG}_bias-corrected-agcd-${BIAS_METHOD}.zarr.zip

INDEPENDENCE_PLOT=/g/data/xv83/dbi599/ag/figures/independence-test_${VAR}_cafe-c5-d60-pX-f6_19900501-20191101_${TIME_FREQ}-${TIME_AGG}_${REGION}-${SUB_REGION}-${SPATIAL_AGG}_bias-corrected-agcd-${BIAS_METHOD}.png
INDEPENDENCE_OPTIONS=--spatial_selection region=${SUB_REGION} --lead_time_increment 1

DASK_CONFIG=config/dask_local.yml



