.PHONY: all process-obs process-forecast bias-correction clean help settings

INIT_DATES=1990-11-01 1991-11-01 1992-11-01 1993-11-01
LEAD_TIME=3650
REGION=TAS-POINT
BIAS_METHOD=additive
VAR=pr
UNITS=mm/day
OBS=awap

PYTHON=/g/data/e14/dbi599/miniconda3/envs/unseen/bin/python
DATA_DIR=/g/data/xv83/dbi599
FCST_DATA := $(sort $(wildcard /g/data/xv83/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-199*1101/ZARR/atmos_isobaric_daily.zarr.zip))
FCST_METADATA=cafe.yml

ifeq (${OBS}, awap)
  OBS_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-csiro-awap/rain_day_19000101-20201202_cafe-grid.zarr/
  OBS_METADATA=awap.yml
else ifeq (${OBS}, jra55)
  OBS_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-jra55/surface_daily_cafe-grid.zarr/
  OBS_METADATA=jra.yml
endif

## process-obs : observation data in forecast (i.e. initial date / lead time) format
OBS_FORECAST_FILE=${DATA_DIR}/${VAR}_${OBS}_19901101-19931101_${LEAD_TIME}D_cafe-grid-${REGION}.zarr.zip
process-obs : ${OBS_FORECAST_FILE}
${OBS_FORECAST_FILE} : ${OBS_DATA} ${OBS_METADATA}
	${PYTHON} preprocess_obs.py $< $@ --metadata_file $(word 2,$^) --variables ${VAR} --n_lead_steps ${LEAD_TIME} --init_dates ${INIT_DATES} --no_leap_days --region ${REGION} --units ${VAR}=${UNITS}

## process-forecast : preprocessing of CAFE forecast ensemble
FCST_ENSEMBLE_FILE=/g/data/xv83/dbi599/${VAR}_cafe-c5-d60-pX-f6_19901101-19931101_${LEAD_TIME}D_cafe-grid-${REGION}.zarr.zip
process-forecast : ${FCST_ENSEMBLE_FILE}
${FCST_ENSEMBLE_FILE} : ${FCST_METADATA}
	${PYTHON} preprocess_forecasts.py ${FCST_DATA} $@ --metadata_file $< --variables ${VAR} --no_leap_days --region ${REGION} --units ${VAR}=${UNITS}

## bias-correction : bias corrected forecast data using observations
FCST_BIAS_FILE=/g/data/xv83/dbi599/${VAR}_cafe-c5-d60-pX-f6_${OBS}-${BIAS_METHOD}-correction_19901101-19931101_${LEAD_TIME}D_cafe-grid-${REGION}.zarr.zip
bias-correction : ${FCST_BIAS_FILE}
${FCST_BIAS_FILE} : ${FCST_ENSEMBLE_FILE} ${OBS_FORECAST_FILE}
	${PYTHON} bias_correction.py $< $(word 2,$^) pr ${BIAS_METHOD} $@

## clean : remove all generated files
clean :
	rm ${OBS_FORECAST_FILE} ${FCST_ENSEMBLE_FILE} ${FCST_BIAS_FILE}

## settings : show variables' values
settings :
	@echo INIT_DATES: ${INIT_DATES}
	@echo LEAD_TIME: ${LEAD_TIME}
	@echo REGION: ${REGION}
	@echo BIAS_METHOD: ${BIAS_METHOD}
	@echo VAR: ${VAR}
	@echo UNITS: ${UNITS}
	@echo OBS: ${OBS}

## help : show this message
help :
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

