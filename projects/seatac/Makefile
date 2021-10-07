.PHONY: all plot-histogram clean help

include ${CONFIG}

PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python

## plot-histogram : plot TXx histogram
plot-histogram : ${TXX_HISTOGRAM_PLOT}
${TXX_HISTOGRAM_PLOT} : ${OBS_FILE} ${OBS_CONFIG} ${FCST_ENSEMBLE_FILE}
	${PYTHON} plot_seatac_TXx_histogram.py $< $(word 2,$^) $(word 3,$^) $@  --plotparams ${CONFIG_DIR}/plotparams_publication.yml

## clean : remove all generated files
clean :
	rm ${TXX_HISTOGRAM_PLOT}

## help : show this message
help :
	@echo 'make [target] [-Bnf] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

