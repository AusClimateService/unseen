.PHONY: final_analysis help

include ${CONFIG}

PYTHON=/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python
PLOT_PARAMS=${CONFIG_DIR}/plotparams_publication.yml
	
## define_regions : define the wheat-sheep regions
define_regions : define_regions.ipynb
define_regions.ipynb : abares.zip ${FCST_CLIMATOLOGY}
	papermill -p abares_shapefile $< -p agcd_file ${OBS_DATA} -p agcd_config ${OBS_METADATA} $@ $@

## final_analysis : do the final analysis
final_analysis : ag_analysis_${SUB_REGION}.ipynb
ag_analysis_${SUB_REGION}.ipynb : ag_analysis.ipynb    
	papermill -p agcd_file ${OBS_PROCESSED_FILE} -p cafe_file ${FCST_ENSEMBLE_FILE} -p cafe_bc_file ${FCST_BIAS_FILE} -p fidelity_file ${SIMILARITY_FILE} -p independence_plot ${INDEPENDENCE_PLOT} -p region ${SUB_REGION} $< $@

## help : show this message
help :
	@echo 'make [target] [-Bnf] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

