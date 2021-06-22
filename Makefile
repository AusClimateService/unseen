.PHONY: all process-obs process-forecast bias-correction clean help settings

REGION=TAS-POINT
BIAS_METHOD=additive
VAR=pr
#pr ffdi
UNITS=--units ${VAR}=mm/day
# --units ${VAR}=mm/day
OBS=awap
# jra55 awap

PYTHON=/g/data/e14/dbi599/miniconda3/envs/unseen/bin/python
DATA_DIR=/g/data/xv83/dbi599
FCST_DATA := $(sort $(wildcard /g/data/xv83/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-199[1,2]1101/ZARR/atmos_isobaric_daily.zarr.zip))
FCST_METADATA=config/dataset_cafe.yml

ifeq (${OBS}, awap)
  OBS_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-csiro-awap/rain_day_19000101-20201202_cafe-grid.zarr/
  OBS_METADATA=config/dataset_awap.yml
else ifeq (${OBS}, jra55)
  OBS_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-jra55/surface_daily_cafe-grid.zarr/
  OBS_METADATA=config/dataset_jra55.yml
endif

## process-obs : preprocessing of observational data
OBS_FORECAST_FILE=${DATA_DIR}/${VAR}_${OBS}_cafe-grid-${REGION}.zarr.zip
process-obs : ${OBS_FORECAST_FILE}
${OBS_FORECAST_FILE} : ${OBS_DATA} ${OBS_METADATA}
	${PYTHON} unseen/preprocess.py $< obs $@ --metadata_file $(word 2,$^) --variables ${VAR} --no_leap_days --region ${REGION} ${UNITS}

## process-forecast : preprocessing of CAFE forecast ensemble
FCST_ENSEMBLE_FILE=/g/data/xv83/dbi599/${VAR}_cafe-c5-d60-pX-f6_19911101-19921101_3650D_cafe-grid-${REGION}.zarr.zip
process-forecast : ${FCST_ENSEMBLE_FILE}
${FCST_ENSEMBLE_FILE} : ${FCST_METADATA}
	${PYTHON} unseen/preprocess.py ${FCST_DATA} forecast $@ --metadata_file $< --variables ${VAR} --no_leap_days --region ${REGION} ${UNITS} --output_chunks lead_time=50  #--isel level=-1

## bias-correction : bias corrected forecast data using observations
FCST_BIAS_FILE=/g/data/xv83/dbi599/${VAR}_cafe-c5-d60-pX-f6_${OBS}-${BIAS_METHOD}-correction_19911101-19921101_3650D_cafe-grid-${REGION}.zarr.zip
bias-correction : ${FCST_BIAS_FILE}
${FCST_BIAS_FILE} : ${FCST_ENSEMBLE_FILE} ${OBS_FORECAST_FILE}
	${PYTHON} unseen/bias_correction.py $< $(word 2,$^) ${VAR} ${BIAS_METHOD} $@

## clean : remove all generated files
clean :
	rm ${OBS_FORECAST_FILE} ${FCST_ENSEMBLE_FILE} ${FCST_BIAS_FILE}

## settings : show variables' values
settings :
	@echo REGION: ${REGION}
	@echo BIAS_METHOD: ${BIAS_METHOD}
	@echo VAR: ${VAR}
	@echo UNITS: ${UNITS}
	@echo OBS: ${OBS}

## help : show this message
help :
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

