.PHONY: all process-obs process-forecast bias-correction similarity-test independence-test clean help settings

include ${CONFIG}

SCRIPT_DIR=/home/599/dbi599/unseen/cmdline_scripts
PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python

## process-obs : preprocessing of observational data
process-obs : ${OBS_PROCESSED_FILE}
${OBS_PROCESSED_FILE} : ${OBS_DATA} ${OBS_METADATA}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py $< obs $@ --metadata_file $(word 2,$^) ${GENERAL_IO_OPTIONS} ${OBS_IO_OPTIONS}

## process-forecast : preprocessing of CAFE forecast ensemble
process-forecast : ${FCST_ENSEMBLE_FILE}
${FCST_ENSEMBLE_FILE} : ${FCST_METADATA}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py ${FCST_DATA} forecast $@ --metadata_file $< ${GENERAL_IO_OPTIONS} --reset_times --output_chunks lead_time=50 --dask_config ${DASK_CONFIG}

## bias-correction : bias corrected forecast data using observations
bias-correction : ${FCST_BIAS_FILE}
${FCST_BIAS_FILE} : ${FCST_ENSEMBLE_FILE} ${OBS_PROCESSED_FILE}
	${PYTHON} ${SCRIPT_DIR}/bias_correct.py $< $(word 2,$^) ${VAR} ${BIAS_METHOD} $@ --base_period ${BASE_PERIOD}

## similarity-test : similarity test between observations and bias corrected forecast
similarity-test : ${SIMILARITY_FILE}
${SIMILARITY_FILE} : ${FCST_BIAS_FILE} ${OBS_PROCESSED_FILE}
	${PYTHON} ${SCRIPT_DIR}/similarity_test.py $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD}

## independence-test : independence test for different lead times
independence-test : ${INDEPENDENCE_PLOT}
${INDEPENDENCE_PLOT} : ${FCST_BIAS_FILE}
	${PYTHON} ${SCRIPT_DIR}/independence_test.py $< ${VAR} $@ ${INDEPENDENCE_OPTIONS}

## all : run the whole analysis
all : ${INDEPENDENCE_PLOT} ${SIMILARITY_FILE}

## clean : remove all generated files
clean :
	rm ${OBS_FORECAST_FILE} ${FCST_ENSEMBLE_FILE} ${FCST_BIAS_FILE} ${SIMILARITY_FILE} ${INDEPENDENCE_PLOT}

## settings : show variables' values
settings :
	@echo REGION: ${REGION}
	@echo BIAS_METHOD: ${BIAS_METHOD}
	@echo BASE_PERIOD: ${BASE_PERIOD}
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

