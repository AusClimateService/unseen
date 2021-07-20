.PHONY: all process-obs process-forecast clean help settings

REGION=wheat-sheep
SHAPEFILE=/home/599/dbi599/unseen/shapefiles/wheat_sheep.zip
SHP_HEADER=region
SPATIAL_AGG=sum
VAR=pr
UNITS=--units ${VAR}=mm/day
OBS=awap
TIME_FREQ=A-DEC
TIME_AGG=sum

PYTHON=/g/data/e14/dbi599/miniconda3/envs/unseen/bin/python
DATA_DIR=/g/data/xv83/dbi599
FCST_DATA := $(sort $(wildcard /g/data/xv83/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-199[0,1,2,3]??01/ZARR/atmos_isobaric_month.zarr.zip))
FCST_METADATA=config/dataset_cafe.yml
DASK_CONFIG=config/dask_local.yml
OBS_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-csiro-awap/rain_month_19000101-20191201_cafe-grid.zarr/
OBS_METADATA=config/dataset_awap_monthly.yml

## process-obs : preprocessing of observational data
OBS_FORECAST_FILE=${DATA_DIR}/${VAR}_${OBS}_1900-2019_${TIME_FREQ}-${TIME_AGG}_${REGION}-${SPATIAL_AGG}.zarr.zip
process-obs : ${OBS_FORECAST_FILE}
${OBS_FORECAST_FILE} : ${OBS_DATA} ${OBS_METADATA}
	${PYTHON} cmdline_scripts/preprocess.py $< obs $@ --metadata_file $(word 2,$^) --variables ${VAR} --region ${SHAPEFILE} --shp_header ${SHP_HEADER} --spatial_agg ${SPATIAL_AGG} --no_leap_days --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} ${UNITS}

## process-forecast : preprocessing of CAFE forecast ensemble
FCST_ENSEMBLE_FILE=/g/data/xv83/dbi599/${VAR}_cafe-c5-d60-pX-f6_19900501-19931101_${TIME_FREQ}-${TIME_AGG}_${REGION}-${SPATIAL_AGG}.zarr.zip
process-forecast : ${FCST_ENSEMBLE_FILE}
${FCST_ENSEMBLE_FILE} : ${FCST_METADATA}
	${PYTHON} cmdline_scripts/preprocess.py ${FCST_DATA} forecast $@ --metadata_file $< --variables ${VAR} --region ${SHAPEFILE} --shp_header ${SHP_HEADER} --spatial_agg ${SPATIAL_AGG} --no_leap_days --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} ${UNITS} --output_chunks lead_time=50 --dask_config ${DASK_CONFIG}

## clean : remove all generated files
clean :
	rm ${OBS_FORECAST_FILE} ${FCST_ENSEMBLE_FILE}

## settings : show variables' values
settings :
	@echo REGION: ${REGION}
	@echo SPATIAL_AGG: ${SPATIAL_AGG}
	@echo VAR: ${VAR}
	@echo TIME_FREQ: ${TIME_FREQ}
	@echo TIME_AGG: ${TIME_AGG}
	@echo UNITS: ${UNITS}
	@echo OBS: ${OBS}

## help : show this message
help :
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

