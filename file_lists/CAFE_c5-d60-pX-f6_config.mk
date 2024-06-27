# Configuration file for Makefile workflows

MODEL=CAFE
EXPERIMENT=c5-d60-pX-f6
BASE_PERIOD=1995-01-01 2020-12-31
BASE_PERIOD_TEXT=1995-2020
TIME_PERIOD_TEXT=19950501-20201101
STABILITY_START_YEARS=1995 2000 2005 2010 2015
MODEL_IO_OPTIONS=--metadata_file /home/599/dbi599/unseen/config/dataset_cafe_daily.yml
MODEL_NINO_OPTIONS=--lon_bnds -170 -120 --lat_dim geolat_t --lon_dim geolon_t --agg_x_dim xt_ocean --agg_y_dim yt_ocean --anomaly ${BASE_PERIOD} --anomaly_freq month --metadata_file /home/599/dbi599/unseen/config/dataset_cafe_monthly_ocean.yml


