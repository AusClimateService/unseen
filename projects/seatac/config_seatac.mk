# Configuration for SeaTac airport annual maximum temperture analysis

VAR=tasmax
UNITS=${VAR}=C
LAT=47.45
LON=237.69
REGION_NAME=seatac
TIME_FREQ=A-DEC
TIME_AGG=max

CONFIG_DIR=/home/599/dbi599/unseen/config
SEATAC_DIR=/g/data/xv83/dbi599/seatac

DASK_CONFIG=${CONFIG_DIR}/dask_local.yml

FCST_IO_OPTIONS=--variables ${VAR} --spatial_coords ${LAT} ${LON} --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --complete_time_agg_periods --units ${UNITS}

FCST_DATA := $(sort $(wildcard /g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-??[9,0,1,2]*/atmos_isobaric_daily.zarr.zip))
FCST_METADATA=${CONFIG_DIR}/dataset_cafe_daily.yml
FCST_ENSEMBLE_FILE=${SEATAC_DIR}/${VAR}_cafe-c5-d60-pX-f6_19900501-20201101_${TIME_FREQ}-${TIME_AGG}_${REGION_NAME}.zarr.zip
FCST_CONFIG=${CONFIG_DIR}/dataset_cafe_daily.yml

OBS_FILE=${SEATAC_DIR}/tasmax_GHCNv3_SeaTac_daily_1948-2021.nc
OBS_CONFIG=${CONFIG_DIR}/dataset_knmi_daily.yml

REANALYSIS_HGT_FILE=${SEATAC_DIR}/h500_ERA5_hourly_2021-06-28.grib
REANALYSIS_TAS_FILE=${SEATAC_DIR}/tas_ERA5_hourly_2021-06-28.grib

TXX_HISTOGRAM_PLOT=${SEATAC_DIR}/tasmax_histogram_seatac.pdf
REANALYSIS_HOT_DAY_PLOT=${SEATAC_DIR}/seatac_2021-06-28_era5.pdf
MODEL_HOT_DAY_PLOT=${SEATAC_DIR}/seatac_hottest_day_cafe.pdf
TXX_SAMPLE_PLOT=${SEATAC_DIR}/tasmax_samples_seatac.pdf
TXX_LIKELIHOOD_PLOT=${SEATAC_DIR}/tasmax_likelihoods_seatac.pdf
TXX_RETURN_PERIODS_PLOT=${SEATAC_DIR}/model_return_periods_seatac.pdf
TXX_ANNUAL_MAX_PLOT=${SEATAC_DIR}/model_annual_max_seatac.pdf
TXX_ANNUAL_DIST_PLOT=${SEATAC_DIR}/model_annual_distribution_seatac.pdf
