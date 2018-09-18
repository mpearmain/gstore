import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

"""
Calculate per fold features.
We already have a 'validation' set that mirrors the test set on Kaggle.
For each fold we want to have a fold for stacking, this means we drop another level of complexity. 
 
"""

def calculate_groupby_features(train, test, folds):
    # Run folds for Dev and Valid
    i = 0
    for train_index, test_index in kf.split(train):
        print(f"Running fold {i}")
        j = 0
        X_dev, X_valid = train[train_index], train[test_index]
        j+=1
        for value, df in [X_dev, X_valid]:
            print(f"Running dataset {value}")
            df['browser_category'] = df['device.browser'] + '_' + df['device.deviceCategory']
            df['browser_operatingSystem'] = df['device.browser'] + '_' + df['device.operatingSystem']
            df['source_country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']
            df['visitNumber'] = np.log1p(df['visitNumber'])
            df['totals.hits'] = np.log1p(df['totals.hits'])
            df['totals.pageviews'] = np.log1p(df['totals.pageviews'].fillna(0))
            df['sum_pageviews_per_network_domain'] = \
                df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
            df['count_pageviews_per_network_domain'] = \
                df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
            df['sum_hits_per_network_domain'] = \
                df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
            df['count_hits_per_network_domain'] = \
                df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
            df['mean_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('mean')
            df['sum_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('sum')
            # Network
            df['sum_pageviews_per_network_domain'] = \
                df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
            df['count_pageviews_per_network_domain'] = \
                df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
            df['mean_pageviews_per_network_domain'] = \
                df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
            df['sum_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
            df['count_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('count')
            df['mean_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')
            df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
            df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
            df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')
            df['sum_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('sum')
            df['count_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('count')
            df['mean_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('mean')
            df['user_pageviews_sum'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
            df['user_hits_sum'] = df.groupby('fullVisitorId')['totals.hits'].transform('sum')
            df['user_pageviews_count'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('count')
            df['user_hits_count'] = df.groupby('fullVisitorId')['totals.hits'].transform('count')
            df['user_pageviews_sum_to_mean'] = df['user_pageviews_sum'] / df['user_pageviews_sum'].mean()
            df['user_hits_sum_to_mean'] = df['user_hits_sum'] / df['user_hits_sum'].mean()
            df['user_pageviews_to_region'] = df['user_pageviews_sum'] / df['mean_pageviews_per_region']
            df['user_hits_to_region'] = df['user_hits_sum'] / df['mean_hits_per_region']

    # Finally caluculate for the test set.
    test['browser_category'] = test['device.browser'] + '_' + test['device.deviceCategory']
    test['browser_operatingSystem'] = test['device.browser'] + '_' + test['device.operatingSystem']
    test['source_country'] = test['trafficSource.source'] + '_' + test['geoNetwork.country']
    test['visitNumber'] = np.log1p(test['visitNumber'])
    test['totals.hits'] = np.log1p(test['totals.hits'].astype(int))
    test['totals.pageviews'] = np.log1p(test['totals.pageviews'].astype(float).fillna(0))
    test['sum_pageviews_per_network_domain'] = \
        test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    test['count_pageviews_per_network_domain'] = \
        test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    test['sum_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    test['count_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    test['mean_hits_per_day'] = test.groupby(['day'])['totals.hits'].transform('mean')
    test['sum_hits_per_day'] = test.groupby(['day'])['totals.hits'].transform('sum')
    test['sum_pageviews_per_network_domain'] = \
        test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    test['count_pageviews_per_network_domain'] = \
        test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    test['mean_pageviews_per_network_domain'] = \
        test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
    test['sum_pageviews_per_region'] = test.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
    test['count_pageviews_per_region'] = test.groupby('geoNetwork.region')['totals.pageviews'].transform('count')
    test['mean_pageviews_per_region'] = test.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')
    test['sum_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    test['count_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    test['mean_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')
    test['sum_hits_per_region'] = test.groupby('geoNetwork.region')['totals.hits'].transform('sum')
    test['count_hits_per_region'] = test.groupby('geoNetwork.region')['totals.hits'].transform('count')
    test['mean_hits_per_region'] = test.groupby('geoNetwork.region')['totals.hits'].transform('mean')
    test['user_pageviews_sum'] = test.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
    test['user_hits_sum'] = test.groupby('fullVisitorId')['totals.hits'].transform('sum')
    test['user_pageviews_count'] = test.groupby('fullVisitorId')['totals.pageviews'].transform('count')
    test['user_hits_count'] = test.groupby('fullVisitorId')['totals.hits'].transform('count')
    test['user_pageviews_sum_to_mean'] = test['user_pageviews_sum'] / test['user_pageviews_sum'].mean()
    test['user_hits_sum_to_mean'] = test['user_hits_sum'] / test['user_hits_sum'].mean()
    test['user_pageviews_to_region'] = test['user_pageviews_sum'] / test['mean_pageviews_per_region']
    test['user_hits_to_region'] = test['user_hits_sum'] / test['mean_hits_per_region']

    return train, test

## --------------------- Build encoding per fold -----------------------------##

# Load different data sources.
dev = pd.read_parquet('input/processed/dev_static_features.parquet.gzip')
valid = pd.read_parquet('input/processed/valid_static_features.parquet.gzip')
train_df = pd.read_parquet('input/processed/train_static_features.parquet.gzip')
test_df = pd.read_parquet('input/processed/test_static_features.parquet.gzip')

# Set the K-fold and run for Dev, Valid, then Train, test
kf = KFold(n_splits=5, random_state=42, shuffle=False)


# Build out dev and valid
dev, valid = calculate_groupby_features(dev, valid, kf)
train, test = calculate_groupby_features(train_df, test_df, kf)

print(set(list(train_df)) - set(list(test_df)))

"""
Here we create the dev, valid, split for CVs to use in modelling later.
These splits are based on 3 week validation 
"""

# Dump cleaned data to parquets for later.
dev.to_parquet('input/processed/dev_dynamic_features.parquet.gzip', compression='gzip')
valid.to_parquet('input/processed/valid_dynamic_features.parquet.gzip', compression='gzip')
train_df.to_parquet('input/processed/train_dynamic_features.parquet.gzip', compression='gzip')
test_df.to_parquet('input/processed/test_dynamic_features.parquet.gzip', compression='gzip')
