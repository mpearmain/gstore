import numpy as np
import pandas as pd

"""
Calculate per fold features.
We already have a 'validation' set that mirrors the test set on Kaggle.
For each fold we want to have a fold for stacking, this means we drop another level of complexity. 
 
"""


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(train_series=None,
                  test_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0,
                  aggr_func="mean"):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    train_series : training categorical feature as a pd.Series
    test_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """
    assert len(train_series) == len(target)
    assert train_series.name == test_series.name
    temp = pd.concat([train_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=train_series.name)[target.name].agg([aggr_func, "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages[aggr_func] * smoothing
    averages.drop([aggr_func, "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_train_series = pd.merge(
        train_series.to_frame(train_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=train_series.name,
        how='left')['average'].rename(train_series.name + '_' + aggr_func).fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_train_series.index = train_series.index
    ft_test_series = pd.merge(
        test_series.to_frame(test_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=test_series.name,
        how='left')['average'].rename(train_series.name + '_' + aggr_func).fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_test_series.index = test_series.index
    return add_noise(ft_train_series, noise_level), add_noise(ft_test_series, noise_level)


def calculate_groupby_features(df):

    # need thing like xdev.groupby('geoNetwork.region')['totals.pageviews'].count().rank()

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

    return df


## --------------------- Build encoding per fold -----------------------------##

# Load different data sources.
dev = pd.read_parquet('input/processed/dev_static_features.parquet.gzip')
valid = pd.read_parquet('input/processed/valid_static_features.parquet.gzip')
train_df = pd.read_parquet('input/processed/train_static_features.parquet.gzip')
test_df = pd.read_parquet('input/processed/test_static_features.parquet.gzip')

# Build out dev and valid
dev = calculate_groupby_features(dev)
valid = calculate_groupby_features(valid)
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
