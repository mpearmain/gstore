"""
Script taken from SRK to load train and test data files and parse JSON cols to have a base data set to work from
"""

import json
import os

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

# Impute 0 for missing target values
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].fillna(0).astype(float)

# Generate list of cols with unique values in train.
columns_to_remove = [col for col in train_df.columns if train_df[col].nunique() == 1]
print(f"Nb. of variables with unique value: {len(columns_to_remove)}")

# Some make sense as having NaN values as per organisers , only remove those.
for col in columns_to_remove:
    if set(['not available in demo dataset']) == set(train_df[col].unique()): continue
    print(col, train_df[col].dtypes, train_df[col].unique())

train_df['totals.bounces'] = train_df['totals.bounces'].fillna('0')
test_df['totals.bounces'] = test_df['totals.bounces'].fillna('0')
train_df['totals.newVisits'] = train_df['totals.newVisits'].fillna('0')
test_df['totals.newVisits'] = test_df['totals.newVisits'].fillna('0')
train_df['trafficSource.adwordsClickInfo.isVideoAd'] = train_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)
test_df['trafficSource.adwordsClickInfo.isVideoAd'] = test_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)
train_df['trafficSource.isTrueDirect'] = train_df['trafficSource.isTrueDirect'].fillna(False)
test_df['trafficSource.isTrueDirect'] = test_df['trafficSource.isTrueDirect'].fillna(False)

# Now remove variables with only a single class:
cols = [col for col in train_df.columns if train_df[col].nunique() > 1]
train_df = train_df[cols]

# Lets convert obvious things now.

train_df['visitStartTime'].astype(str).astype(int)
train_df['visitId'].astype(str).astype(int)
train_df['totals.transactionRevenue'].fillna(value = '0', inplace = True)
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].astype(int)
train_df['visitNumber'].astype(str).astype(int)
train_df['totals.hits'] = train_df['totals.hits'].astype(int)
train_df['totals.pageviews'].fillna(value = '0', inplace = True)
train_df['totals.pageviews'] = train_df['totals.pageviews'].astype(int)

test_df['visitStartTime'].astype(str).astype(int)
test_df['visitId'].astype(str).astype(int)
test_df['visitNumber'].astype(str).astype(int)
test_df['totals.hits'] = test_df['totals.hits'].astype(int)
test_df['totals.pageviews'].fillna(value = '0', inplace = True)
test_df['totals.pageviews'] = test_df['totals.pageviews'].astype(int)



# Remove the target col from test set.
cols.remove('totals.transactionRevenue')
test_df = test_df[cols]




# Should only be 'totals.transactionRevenue' different
print(set(list(train_df)) - set(list(test_df)))

# Dump cleaned data to parquets for later.
train_df.to_parquet('input/cleaned/train.parquet.gzip', compression='gzip')
test_df.to_parquet('input/cleaned/test.parquet.gzip', compression='gzip')
