.PHONY: all jra awap clean help settings

INIT_DATES=1989-11-01 1990-11-01 1991-11-01 1992-11-01
LEAD_TIME=3650
REGION=aus
PYTHON=/g/data/e14/dbi599/miniconda3/envs/unseen/bin/python
DATA_DIR=/g/data/xv83/dbi599
JRA_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-jra55/surface_daily_cafe-grid.zarr/
AWAP_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-csiro-awap/rain_day_19000101-20201202_cafe-grid.zarr/

## jra : JRA-55 precip data in forecast (i.e. initial date / lead time) format
JRA_PR_FORECAST_FILE=${DATA_DIR}/precip_jra55_19891101-19921101_${LEAD_TIME}D_cafe-grid-${REGION}.zarr.zip
jra : ${JRA_PR_FORECAST_FILE}
${JRA_PR_FORECAST_FILE} : ${JRA_DATA}
	${PYTHON} preprocess_obs.py $< TPRAT_GDS0_SFC precip $@ --n_lead_steps ${LEAD_TIME} --init_dates ${INIT_DATES} --dataset JRA-55 --no_leap_days --region ${REGION} --chunk_size 3000

## awap : AWAP precip data in forecast (i.e. initial date / lead time) format
AWAP_PR_FORECAST_FILE=${DATA_DIR}/precip_awap_19891101-19921101_3650D_cafe-grid.zarr.zip
awap : ${AWAP_PR_FORECAST_FILE}
${AWAP_PR_FORECAST_FILE} : ${AWAP_DATA}
	${PYTHON} preprocess_obs.py $< precip precip $@ --n_lead_steps ${LEAD_TIME} --init_dates ${INIT_DATES} --no_leap_days

## clean : remove all generated files
clean :
	rm ${JRA_PR_FORECAST_FILE} ${AWAP_PR_FORECAST_FILE}

## settings : show variables' values
settings :
	@echo INIT_DATES: $(INIT_DATES)
	@echo LEAD_TIME: $(LEAD_TIME)
	@echo REGION: $(REGION)

## help : show this message
help :
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

