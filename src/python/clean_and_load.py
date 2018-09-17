"""
Script taken from SRK to load train and test data files and parse JSON cols to have a base data set to work from
"""

import os
import json
import pandas as pd
from pandas.io.json import json_normalize

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


train_df = load_df("./input/raw/train.csv")
test_df = load_df("./input/raw/test.csv")

# Drop cols
cols_to_drop = ['socialEngagementType',
                'device.browserSize',
                'device.browserVersion',
                'device.flashVersion',
                'device.language',
                'device.mobileDeviceBranding',
                'device.mobileDeviceInfo',
                'device.mobileDeviceMarketingName',
                'device.mobileDeviceModel',
                'device.mobileInputSelector',
                'device.operatingSystemVersion',
                'device.screenColors',
                'device.screenResolution',
                'geoNetwork.cityId',
                'geoNetwork.latitude',
                'geoNetwork.longitude',
                'geoNetwork.networkLocation',
                'totals.bounces',
                'totals.newVisits',
                'totals.visits',
                'trafficSource.adwordsClickInfo.criteriaParameters',
                'trafficSource.adwordsClickInfo.isVideoAd',
                'trafficSource.isTrueDirect',
                'sessionId']

train_df = train_df.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)

# Convert str date to timestamp
train_df['date'] = pd.to_datetime(train_df['date'], format='%Y%m%d')
test_df['date'] = pd.to_datetime(test_df['date'], format='%Y%m%d')

# Impute 0 for missing target values
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].fillna(0).astype(float)

# Dump cleaned data to parquets for later.
train_df.to_parquet('input/cleaned/train.parquet.gzip', compression='gzip')
test_df.to_parquet('input/cleaned/test.parquet.gzip', compression='gzip')
