.PHONY: all process-obs process-forecast bias-correction clean help settings

include ${CONFIG}

SCRIPT_DIR=/home/599/dbi599/unseen/cmdline_scripts
PYTHON=/g/data/e14/dbi599/miniconda3/envs/unseen/bin/python

## process-obs : preprocessing of observational data
process-obs : ${OBS_FORECAST_FILE}
${OBS_FORECAST_FILE} : ${OBS_DATA} ${OBS_METADATA}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py $< obs $@ --metadata_file $(word 2,$^) ${IO_OPTIONS}

## process-forecast : preprocessing of CAFE forecast ensemble
process-forecast : ${FCST_ENSEMBLE_FILE}
${FCST_ENSEMBLE_FILE} : ${FCST_METADATA}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py ${FCST_DATA} forecast $@ --metadata_file $< ${IO_OPTIONS} --reset_times --output_chunks lead_time=50 --dask_config ${DASK_CONFIG}

## bias-correction : bias corrected forecast data using observations
bias-correction : ${FCST_BIAS_FILE}
${FCST_BIAS_FILE} : ${FCST_ENSEMBLE_FILE} ${OBS_FORECAST_FILE}
	${PYTHON} ${SCRIPT_DIR}/bias_correct.py $< $(word 2,$^) ${VAR} ${BIAS_METHOD} $@ --base_period ${BASE_PERIOD}

## clean : remove all generated files
clean :
	rm ${OBS_FORECAST_FILE} ${FCST_ENSEMBLE_FILE} ${FCST_BIAS_FILE}

## settings : show variables' values
settings :
	@echo REGION: ${REGION}
	@echo BIAS_METHOD: ${BIAS_METHOD}
	@echo VAR: ${VAR}
	@echo TIME_FREQ: ${TIME_FREQ}
	@echo TIME_AGG: ${TIME_AGG}
	@echo UNITS: ${UNITS}
	@echo OBS: ${OBS}

## help : show this message
help :
	@echo 'make [target] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

