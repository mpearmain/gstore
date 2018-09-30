import logging

import pandas as pd
from sklearn import preprocessing


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()

# Load different data sources.
train_df = pd.read_parquet('input/processed/train_static_features.parquet.gzip')
test_df = pd.read_parquet('input/processed/test_static_features.parquet.gzip')

# OHE Cols with small number of uniques (less than 20)
ohe_cols = ['trafficSource.campaign', 'channelGrouping',
            'trafficSource.adwordsClickInfo.page', 'trafficSource.medium',
            'geoNetwork.continent', 'trafficSource.keyword_groups',
            'device.deviceCategory', 'totals.bounces', 'totals.newVisits',
            'trafficSource.adwordsClickInfo.slot',
            'trafficSource.adwordsClickInfo.adNetworkType']

# Remove target col.
y_train = train_df['totals.transactionRevenue'].values
train_df = train_df.drop(['totals.transactionRevenue'], axis=1)

# Join datasets for rowise feature engineering.
trn_len = train_df.shape[0]
merged_df = pd.concat([train_df, test_df])

merged_df_one_hot = pd.get_dummies(merged_df[ohe_cols])
merged_df = pd.concat([merged_df, merged_df_one_hot], axis=1)
merged_df = merged_df.drop(ohe_cols, axis=1)

del merged_df_one_hot

# Split data back to original data sets.
train_df = merged_df[:trn_len]
test_df = merged_df[trn_len:]
del merged_df

train_df['totals.transactionRevenue'] = y_train


# High Cardinality Cols left.
high_card = ['geoNetwork.networkDomain',
             'trafficSource.adwordsClickInfo.gclId', "trafficSource.keyword",
             'source_country', 'trafficSource.referralPath', 'geoNetwork.city',
             'trafficSource.source', 'geoNetwork.region', 'geoNetwork.country',
             'browser_operatingSystem', 'geoNetwork.metro', 'browser_category',
             'device.browser', 'trafficSource.adContent',
             'geoNetwork.subContinent', 'device.operatingSystem']

for col in high_card:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

# Build out dev and valid


print(set(list(train_df)) - set(list(test_df)))

# Dump cleaned data to parquets for later.
train_df.to_parquet('input/processed/train_encoded_features.parquet.gzip', compression='gzip')
test_df.to_parquet('input/processed/test_encoded_features.parquet.gzip', compression='gzip')
