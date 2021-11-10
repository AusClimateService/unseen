.PHONY: all clean help

include ${CONFIG}

PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python
PLOT_PARAMS=${CONFIG_DIR}/plotparams_publication.yml
REGION=all
#REGION all south-west south-east north-east

## ag_analysis.ipynb : do the final analysis
ag_analysis.ipynb : ${OBS_PROCESSED_FILE} ${FCST_ENSEMBLE_FILE} ${FCST_BIAS_FILE} ${SIMILARITY_FILE} ${INDEPENDENCE_PLOT}
	papermill -p agcd_file $< -p cafe_file $(word 2,$^) -p cafe_bc_file $(word 3,$^) -p fidelity_file $(word 4,$^) -p independence_plot $(word 5,$^) -p region ${REGION} $@ $@

## clean : remove all generated files
clean :
	rm 

## help : show this message
help :
	@echo 'make [target] [-Bnf] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

