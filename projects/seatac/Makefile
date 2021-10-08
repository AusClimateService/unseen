.PHONY: all plot-histogram plot-reanalysis-hot-day plot-model-hot-day clean help

include ${CONFIG}

PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python
PLOT_PARAMS=${CONFIG_DIR}/plotparams_publication.yml

## plot-histogram : plot TXx histogram
plot-histogram : ${TXX_HISTOGRAM_PLOT}
${TXX_HISTOGRAM_PLOT} : ${OBS_FILE} ${OBS_CONFIG} ${FCST_ENSEMBLE_FILE}
	${PYTHON} plot_seatac_TXx_histogram.py $< $(word 2,$^) $(word 3,$^) $@  --plotparams ${PLOT_PARAMS}

## plot-reanalysis-hot-day : plot reanalysis hottest day
plot-reanalysis-hot-day : ${REANALYSIS_HOT_DAY_PLOT}
${REANALYSIS_HOT_DAY_PLOT} : ${REANALYSIS_HGT_FILE} ${REANALYSIS_TAS_FILE}
	${PYTHON} plot_reanalysis_hottest_day.py $< $(word 2,$^) $@  --plotparams ${PLOT_PARAMS} --point ${LON} ${LAT}

## plot-model-hot-day : plot model hottest day
plot-model-hot-day : ${MODEL_HOT_DAY_PLOT}
${MODEL_HOT_DAY_PLOT} : ${FCST_CONFIG}
	${PYTHON} plot_model_hottest_day.py ${FCST_DATA} $< $@ --plotparams ${PLOT_PARAMS} --max_index 73506 61 --point ${LON} ${LAT}

## clean : remove all generated files
clean :
	rm ${TXX_HISTOGRAM_PLOT} ${REANALYSIS_HOT_DAY_PLOT} ${MODEL_HOT_DAY_PLOT}

## help : show this message
help :
	@echo 'make [target] [-Bnf] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

