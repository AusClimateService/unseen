# Configuration file for Makefile workflows

MODEL=CMCC-CM2-SR5
EXPERIMENT=dcppA-hindcast
BASE_PERIOD=1960-01-01 2019-12-31
BASE_PERIOD_TEXT=196011-201911
TIME_PERIOD_TEXT=196011-201911
STABILITY_START_YEARS=1960 1970 1980 1990 2000 2010
MODEL_IO_OPTIONS=--n_ensemble_files 10
MODEL_NINO_OPTIONS=${MODEL_IO_OPTIONS} --lon_bnds 190 240 --lat_dim latitude --lon_dim longitude --agg_y_dim j --agg_x_dim i --anomaly ${BASE_PERIOD} --anomaly_freq month


