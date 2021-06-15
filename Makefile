.PHONY: all jra awap cafe awap-bias clean help settings

INIT_DATES=1990-11-01 1991-11-01 1992-11-01 1993-11-01
LEAD_TIME=3650
REGION=TAS-POINT
BIAS_METHOD=additive
UNITS=mm/day

PYTHON=/g/data/e14/dbi599/miniconda3/envs/unseen/bin/python
DATA_DIR=/g/data/xv83/dbi599
JRA_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-jra55/surface_daily_cafe-grid.zarr/
JRA_METADATA=jra.yml
AWAP_DATA=/g/data/xv83/ds0092/data/csiro-dcfp-csiro-awap/rain_day_19000101-20201202_cafe-grid.zarr/
AWAP_METADATA=awap.yml
CAFE_DATA := $(sort $(wildcard /g/data/xv83/ds0092/CAFE/forecasts/f6/WIP/c5-d60-pX-f6-199*1101/ZARR/atmos_isobaric_daily.zarr.zip))
CAFE_METADATA=cafe.yml

## jra : JRA-55 precip data in forecast (i.e. initial date / lead time) format
JRA_PR_FORECAST_FILE=${DATA_DIR}/pr_jra55_19901101-19931101_${LEAD_TIME}D_cafe-grid-${REGION}.zarr.zip
jra : ${JRA_PR_FORECAST_FILE}
${JRA_PR_FORECAST_FILE} : ${JRA_DATA} ${JRA_METADATA}
	${PYTHON} preprocess_obs.py $< $@ --metadata_file $(word 2,$^) --variables pr --n_lead_steps ${LEAD_TIME} --init_dates ${INIT_DATES} --no_leap_days --region ${REGION} --chunk_size 3000 --units pr=${UNITS}

## awap : AWAP precip data in forecast (i.e. initial date / lead time) format
AWAP_PR_FORECAST_FILE=${DATA_DIR}/pr_awap_19901101-19931101_3650D_cafe-grid-${REGION}.zarr.zip
awap : ${AWAP_PR_FORECAST_FILE}
${AWAP_PR_FORECAST_FILE} : ${AWAP_DATA} ${AWAP_METADATA}
	${PYTHON} preprocess_obs.py $< $@ --metadata_file $(word 2,$^) --variables pr --n_lead_steps ${LEAD_TIME} --init_dates ${INIT_DATES} --no_leap_days --region ${REGION} --units pr=${UNITS}

## cafe : preprocessing of CAFE ensemble
CAFE_PR_ENSEMBLE_FILE=/g/data/xv83/dbi599/pr_cafe-c5-d60-pX-f6_19901101-19931101_3650D_cafe-grid-${REGION}.zarr.zip
cafe : ${CAFE_PR_ENSEMBLE_FILE}
${CAFE_PR_ENSEMBLE_FILE} : ${CAFE_METADATA}
	${PYTHON} preprocess_forecasts.py ${CAFE_DATA} $@ --metadata_file $< --variables pr --no_leap_days --region ${REGION} --units pr=${UNITS}

## awap-bias : bias corrected CAFE data using AWAP
CAFE_PR_BIAS_FILE=/g/data/xv83/dbi599/pr_cafe-c5-d60-pX-f6_awap-additive-correction_19901101-19931101_3650D_cafe-grid-${REGION}.zarr.zip
awap-bias : ${CAFE_PR_BIAS_FILE}
${CAFE_PR_BIAS_FILE} : ${CAFE_PR_ENSEMBLE_FILE} ${AWAP_PR_FORECAST_FILE}
	${PYTHON} bias_correction.py $< $(word 2,$^) pr ${BIAS_METHOD} $@

## clean : remove all generated files
clean :
	rm ${JRA_PR_FORECAST_FILE} ${AWAP_PR_FORECAST_FILE} ${CAFE_PR_ENSEMBLE_FILE} ${CAFE_PR_BIAS_FILE}

## settings : show variables' values
settings :
	@echo INIT_DATES: ${INIT_DATES}
	@echo LEAD_TIME: ${LEAD_TIME}
	@echo REGION: ${REGION}
	@echo BIAS_METHOD: ${BIAS_METHOD}

## help : show this message
help :
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

