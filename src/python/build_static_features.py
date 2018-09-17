import pandas as pd
import numpy as np
from sklearn import preprocessing
from src.python.space_configs import space_lightgbm, tune_model
from lightgbm import LGBMRegressor
from ml_metrics import rmse

# Dump cleaned data to parquets for later.
train_df = pd.read_parquet('input/cleaned/train.parquet.gzip')
test_df = pd.read_parquet('input/cleaned/test.parquet.gzip')







# label encode the categorical variables and convert the numerical variables to float
cat_cols = ["channelGrouping", "device.browser", "device.deviceCategory", "device.operatingSystem",
            "geoNetwork.city", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent",
            "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.gclId",
            "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", "trafficSource.referralPath", "trafficSource.source"]
num_cols = ["totals.hits", "totals.pageviews"]
for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)


# Dump cleaned data to parquets for later.
train_df.to_parquet('input/processed/train_static_features.parquet.gzip', compression='gzip')
test_df.to_parquet('input/processed/test_static_features.parquet.gzip', compression='gzip')