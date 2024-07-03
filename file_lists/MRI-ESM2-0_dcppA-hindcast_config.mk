# Configuration files for Makefile workflows

MODEL=MRI-ESM2-0
EXPERIMENT=dcppA-hindcast
BASE_PERIOD=1965-01-01 2018-12-31
BASE_PERIOD_TEXT=196011-201911
TIME_PERIOD_TEXT=196011-201911
STABILITY_START_YEARS=1960 1970 1980 1990 2000 2010
MODEL_IO_OPTIONS=--n_ensemble_files 10
MODEL_NINO_OPTIONS=--n_ensemble_files 10 --lon_bnds 190 240 --lat_dim latitude --lon_dim longitude --agg_y_dim y --agg_x_dim x --anomaly ${BASE_PERIOD} --anomaly_freq month




