.PHONY: all tasmax-obs tasmax-forecast tasmax-bias-correction calc-txx-obs calc-txx-forecast calc-txx-forecast-bias-corrected plot-histogram plot-reanalysis-hot-day plot-model-hot-day plot-sample-size-dist plot-likelihoods plot-return-periods plot-annual-max plot-distribution clean help

include ${CONFIG}

PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python
SCRIPT_DIR=/home/599/dbi599/unseen/cmdline_scripts
PLOT_PARAMS=${CONFIG_DIR}/plotparams_publication.yml

## tasmax-obs : preparation of observed tasmax data
tasmax-obs : ${OBS_TASMAX_FILE}
${OBS_TASMAX_FILE} : ${OBS_DATA} ${OBS_METADATA}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py $< obs $@ --metadata_file $(word 2,$^) --variables tasmax --units ${UNITS} --no_leap_day --input_freq D 

## tasmax-forecast : preparation of CAFE tasmax data
tasmax-forecast : ${FCST_TASMAX_FILE}
${FCST_TASMAX_FILE} : ${FCST_METADATA}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py ${FCST_DATA} forecast $@ --metadata_file $< --variables tasmax --units ${UNITS} --no_leap_day --input_freq D --spatial_coords ${LAT} ${LON} --output_chunks lead_time=50 --dask_config ${DASK_CONFIG}

## tasmax-bias-correction : bias correct tasmax data using observations
tasmax-bias-correction : ${FCST_TASMAX_BIAS_CORRECTED_FILE}
${FCST_TASMAX_BIAS_CORRECTED_FILE} : ${FCST_TASMAX_FILE} ${OBS_TASMAX_FILE}
	${PYTHON} ${SCRIPT_DIR}/bias_correct.py $< $(word 2,$^) tasmax ${BIAS_METHOD} $@ --base_period ${BASE_PERIOD}

## calc-txx-obs : calculate annual daily maximum temperature from observational data
calc-txx-obs : ${OBS_TXX_FILE}
${OBS_TXX_FILE} : ${OBS_TASMAX_FILE}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py $< obs $@ --variables tasmax --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --input_freq D 

## calc-txx-forecast : calculate annual daily maximum temperature from forecast data
calc-txx-forecast : ${FCST_TXX_FILE}
${FCST_TXX_FILE} : ${FCST_TASMAX_FILE}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py $< obs $@ --variables tasmax --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --complete_time_agg_periods --input_freq D 

## calc-txx-forecast-bias-corrected : calculate annual daily maximum temperature from bias corrected forecast data
calc-txx-forecast-bias-corrected : ${FCST_TXX_BIAS_CORRECTED_FILE}
${FCST_TXX_BIAS_CORRECTED_FILE} : ${FCST_TASMAX_BIAS_CORRECTED_FILE}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py $< obs $@ --variables tasmax --time_freq ${TIME_FREQ} --time_agg ${TIME_AGG} --complete_time_agg_periods --input_freq D 

## similarity-test : similarity test between observations and bias corrected forecast
#similarity-test : ${SIMILARITY_FILE}
#${SIMILARITY_FILE} : ${FCST_BIAS_FILE} ${OBS_PROCESSED_FILE}
#	${PYTHON} ${SCRIPT_DIR}/similarity_test.py $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD}

## independence-test : independence test for different lead times
#independence-test : ${INDEPENDENCE_PLOT}
#${INDEPENDENCE_PLOT} : ${FCST_BIAS_FILE}
#	${PYTHON} ${SCRIPT_DIR}/independence_test.py $< ${VAR} $@ ${INDEPENDENCE_OPTIONS}

## plot-histogram : plot TXx histogram
plot-histogram : ${TXX_HISTOGRAM_PLOT}
${TXX_HISTOGRAM_PLOT} : ${OBS_TXX_FILE} ${FCST_TXX_FILE} ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_seatac_TXx_histogram.py $< $(word 2,$^) $(word 3,$^) $@  --plotparams ${PLOT_PARAMS}

## plot-reanalysis-hot-day : plot reanalysis hottest day
plot-reanalysis-hot-day : ${REANALYSIS_HOT_DAY_PLOT}
${REANALYSIS_HOT_DAY_PLOT} : ${REANALYSIS_HGT_FILE} ${REANALYSIS_TAS_FILE}
	${PYTHON} plot_reanalysis_hottest_day.py $< $(word 2,$^) $@  --plotparams ${PLOT_PARAMS} --point ${LON} ${LAT}

## plot-model-hot-day : plot model hottest day
plot-model-hot-day : ${MODEL_HOT_DAY_PLOT}
${MODEL_HOT_DAY_PLOT} : ${FCST_HOT_DAY_DATA} ${FCST_METADATA}
	${PYTHON} plot_model_hottest_day.py $< $(word 2,$^) ${LAT} ${LON} ${FCST_HOT_DAY_YEAR} $@ --plotparams ${PLOT_PARAMS}

## plot-sample-size-dist : plot TXx sample size distribution
plot-sample-size-dist : ${TXX_SAMPLE_PLOT}
${TXX_SAMPLE_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_TXx_sample_size_distribution.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-likelihoods : plot TXx likelihoods
plot-likelihoods : ${TXX_LIKELIHOOD_PLOT}
${TXX_LIKELIHOOD_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_TXx_likelihoods.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-return-periods : plot TXx return periods
plot-return-periods : ${TXX_RETURN_PERIODS_PLOT}
${TXX_RETURN_PERIODS_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_TXx_return_periods.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-annual-max : plot maximum TXx by year
plot-annual-max : ${TXX_ANNUAL_MAX_PLOT}
${TXX_ANNUAL_MAX_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_maxTXx_by_year.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-distribution : plot TXx distribution by year
plot-distribution : ${TXX_ANNUAL_DIST_PLOT}
${TXX_ANNUAL_DIST_PLOT} : ${FCST_TXX_BIAS_CORRECTED_FILE}
	${PYTHON} plot_TXx_distribution.py $< $@ --plotparams ${PLOT_PARAMS}

## clean : remove all generated files
clean :
	rm ${TXX_HISTOGRAM_PLOT} ${REANALYSIS_HOT_DAY_PLOT} ${MODEL_HOT_DAY_PLOT} ${TXX_SAMPLE_PLOT} ${TXX_LIKELIHOOD_PLOT} ${TXX_RETURN_PERIODS_PLOT} ${TXX_ANNUAL_MAX_PLOT} ${TXX_ANNUAL_DIST_PLOT}

## help : show this message
help :
	@echo 'make [target] [-Bnf] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

