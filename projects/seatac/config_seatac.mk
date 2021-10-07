# Configuration for SeaTac airport annual maximum temperture analysis

VAR=tasmax
UNITS=${VAR}=C
REGION_NAME=seatac
TIME_FREQ=A-DEC
TIME_AGG=max

CONFIG_DIR=/home/599/dbi599/unseen/config
SEATAC_DIR=/g/data/xv83/dbi599/seatac

IO_OPTIONS=--variables ${VAR} --spatial_coords 47.45 -122.31 --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --complete_time_agg_periods --units ${UNITS}

FCST_DATA := $(sort $(wildcard /g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-??[9,0,1,2]*/atmos_isobaric_daily.zarr.zip))
FCST_METADATA=${CONFIG_DIR}/dataset_cafe_daily.yml
FCST_ENSEMBLE_FILE=${SEATAC_DIR}/${VAR}_cafe-c5-d60-pX-f6_19900501-20201101_${TIME_FREQ}-${TIME_AGG}_${REGION_NAME}.zarr.zip

OBS_FILE=${SEATAC_DIR}/tasmax_GHCNv3_SeaTac_daily_1948-2021.nc
OBS_CONFIG=${CONFIG_DIR}/dataset_knmi_daily.yml

TXX_HISTOGRAM_PLOT=${SEATAC_DIR}/tasmax_histogram_seatac.pdf

DASK_CONFIG=${CONFIG_DIR}/dask_local.yml



