.PHONY: all plot-histogram plot-reanalysis-hot-day plot-model-hot-day plot-sample-size-dist plot-likelihoods plot-return-periods plot-annual-max plot-distribution clean help

include ${CONFIG}

PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python
PLOT_PARAMS=${CONFIG_DIR}/plotparams_publication.yml

## plot-histogram : plot TXx histogram
plot-histogram : ${TXX_HISTOGRAM_PLOT}
${TXX_HISTOGRAM_PLOT} : ${OBS_PROCESSED_FILE} ${FCST_ENSEMBLE_FILE} ${FCST_BIAS_FILE}
	${PYTHON} plot_seatac_TXx_histogram.py $< $(word 2,$^) $(word 3,$^) $@  --plotparams ${PLOT_PARAMS}

## plot-reanalysis-hot-day : plot reanalysis hottest day
plot-reanalysis-hot-day : ${REANALYSIS_HOT_DAY_PLOT}
${REANALYSIS_HOT_DAY_PLOT} : ${REANALYSIS_HGT_FILE} ${REANALYSIS_TAS_FILE}
	${PYTHON} plot_reanalysis_hottest_day.py $< $(word 2,$^) $@  --plotparams ${PLOT_PARAMS} --point ${LON} ${LAT}

## plot-model-hot-day : plot model hottest day
plot-model-hot-day : ${MODEL_HOT_DAY_PLOT}
${MODEL_HOT_DAY_PLOT} : ${FCST_HOT_DAY_DATA} ${FCST_METADATA}
	${PYTHON} plot_model_hottest_day.py $< $(word 2,$^) ${LAT} ${LON} $@ --plotparams ${PLOT_PARAMS}

## plot-sample-size-dist : plot TXx sample size distribution
plot-sample-size-dist : ${TXX_SAMPLE_PLOT}
${TXX_SAMPLE_PLOT} : ${FCST_BIAS_FILE}
	${PYTHON} plot_TXx_sample_size_distribution.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-likelihoods : plot TXx likelihoods
plot-likelihoods : ${TXX_LIKELIHOOD_PLOT}
${TXX_LIKELIHOOD_PLOT} : ${FCST_BIAS_FILE}
	${PYTHON} plot_TXx_likelihoods.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-return-periods : plot TXx return periods
plot-return-periods : ${TXX_RETURN_PERIODS_PLOT}
${TXX_RETURN_PERIODS_PLOT} : ${FCST_BIAS_FILE}
	${PYTHON} plot_TXx_return_periods.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-annual-max : plot maximum TXx by year
plot-annual-max : ${TXX_ANNUAL_MAX_PLOT}
${TXX_ANNUAL_MAX_PLOT} : ${FCST_BIAS_FILE}
	${PYTHON} plot_maxTXx_by_year.py $< $@ --plotparams ${PLOT_PARAMS}

## plot-distribution : plot TXx distribution by year
plot-distribution : ${TXX_ANNUAL_DIST_PLOT}
${TXX_ANNUAL_DIST_PLOT} : ${FCST_BIAS_FILE}
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

