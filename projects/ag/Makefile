.PHONY: final_analysis help

include ${CONFIG}

PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python
PLOT_PARAMS=${CONFIG_DIR}/plotparams_publication.yml
REGION=all
#REGION all south-west south-east north-east

## fcst_climatology : calculate the forecast climatology
fcst_climatology : ${FCST_CLIMATOLOGY}
${FCST_CLIMATOLOGY} : ${FCST_METADATA}
	${PYTHON} ${SCRIPT_DIR}/preprocess.py ${FCST_DATA} forecast $@ --metadata_file $< ${CLIM_IO_OPTIONS} 
	
## define_regions : define the wheat-sheep regions
define_regions : define_regions.ipynb
define_regions.ipynb : abares.zip ${FCST_CLIMATOLOGY}
	papermill -p abares_shapefile $< -p agcd_file ${OBS_DATA} -p agcd_config ${OBS_METADATA} -p climatology $(word 2,$^) $@ $@

## final_analysis : do the final analysis
final_analysis : ag_analysis.ipynb
ag_analysis.ipynb :     
	papermill -p agcd_file ${OBS_PROCESSED_FILE} -p cafe_file ${FCST_ENSEMBLE_FILE} -p cafe_bc_file ${FCST_BIAS_FILE} -p fidelity_file ${SIMILARITY_FILE} -p independence_plot ${INDEPENDENCE_PLOT} -p region ${REGION} $@ $@

## help : show this message
help :
	@echo 'make [target] [-Bnf] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

