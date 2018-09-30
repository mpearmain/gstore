import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import KFold

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()

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


def process_date_time(train, test):
    logger.info('Groupby date features')

    test['month_unique_user_count'] = train.groupby('month')['fullVisitorId'].nunique()
    data_df['day_unique_user_count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday_unique_user_count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')
    logger.info('Done with date')

    return data_df


def process_format(data_df):
    logger.info('Start format')
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        data_df[col] = data_df[col].astype(float)
    data_df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    data_df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    logger.info('Done with format')
    return data_df


def process_device(data_df):
    logger.info('Start device')
    data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
    data_df['browser_os'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    logger.info('Done with device')
    return data_df


def process_totals(data_df):
    logger.info('Start totals')
    data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
    data_df['totals_hits'] = np.log1p(data_df['totals_hits'])
    data_df['totals_pageviews'] = np.log1p(data_df['totals_pageviews'].fillna(0))
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
    data_df['mean_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('mean')
    data_df['sum_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('sum')
    data_df['max_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('max')
    data_df['min_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('min')
    logger.info('Done with totals')
    return data_df


def process_geo_network(data_df):
    logger.info('Start geoNetworks')
    data_df['sum_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')[
        'totals_pageviews'].transform('sum')
    data_df['count_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')[
        'totals_pageviews'].transform('count')
    data_df['mean_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')[
        'totals_pageviews'].transform('mean')
    data_df['max_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')[
        'totals_pageviews'].transform('max')
    data_df['sum_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    data_df['count_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform(
        'count')
    data_df['mean_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform(
        'mean')
    data_df['max_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('max')

    logger.info('Done with geoNetworks')
    return data_df


def process_traffic_source(data_df):
    logger.info('Start trafficSource')
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
    data_df['campaign_medium'] = data_df['trafficSource_campaign'] + '_' + data_df['trafficSource_medium']
    data_df['medium_hits_mean'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('mean')
    data_df['medium_hits_max'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('max')
    data_df['medium_hits_min'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('min')
    data_df['medium_hits_sum'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('sum')
    data_df['medium_hits_var'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('var')
    logger.info('Done with trafficSource')
    return data_df


def drop_convert_columns(train_df=None, test_df=None):
    logger.info('Start drop convert')
    cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
    train_df.drop(cols_to_drop, axis=1, inplace=True)
    test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)
    ###only one not null value
    train_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)
    ###converting columns format
    train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].astype(float)
    train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].fillna(0)
    train_df['totals_transactionRevenue'] = np.log1p(train_df['totals_transactionRevenue'])
    logger.info('Done with  drop convert')
    return train_df, test_df

## --------------------- Build encoding per fold -----------------------------##

# Load different data sources.
train_df = pd.read_parquet('input/processed/train_static_features.parquet.gzip')
test_df = pd.read_parquet('input/processed/test_static_features.parquet.gzip')

kf = KFold(10, shuffle=True, random_state=2601)

# Build out dev and valid

for train_index, test_index in kf.split(train_df):
    # First create the different datasets to encode.
    dev = .iloc[train_index]
    valid = X.iloc[test_index]

    # iterate through the combinations, first extract the cols from the tuple.
    for encode_row in encode_combinations:
        encode_cols = []
        [encode_cols.append(j) for j in encode_row]

        aggr_funcs = ["mean", "median"]

        meanDF = pd.DataFrame(encode_X.groupby(encode_cols)[target_col].aggregate(aggr_funcs))
        meanDF = meanDF.reset_index()

        global_mean = encode_X.groupby(encode_cols)[target_col].mean().mean()
        global_median = encode_X.groupby(encode_cols)[target_col].median().median()

        label = target_col
        for i in encode_cols:
            label = label + i
        print("Getting", label, "wise demand..")

        dfcols = [[col for col in encode_cols], [i + label for i in aggr_funcs]]
        meanDF.columns = [item for sublist in dfcols for item in sublist]

        # We only care about the values in this stack.
        encode_y = pd.merge(encode_y, meanDF, on=encode_cols, how="left")
        # fill any missing values (in y not in X)
        encode_y['mean' + label].fillna(global_mean, inplace=True)
        encode_y['median' + label].fillna(global_median, inplace=True)

        encode_y.index = testcv

        #  Create col if not present and add y
        for col in dfcols[1]:
            if col not in encode_train:
                encode_train[col] = np.nan
            encode_train.ix[encode_y.index, col] = encode_y[col]

# Repeat for all the data and create the test set values
# iterate through the combinations, first extract the cols from the tuple.
for encode_row in encode_combinations:
    encode_cols = []
    [encode_cols.append(j) for j in encode_row]

    aggr_funcs = ["mean", "median"]

    meanDF = pd.DataFrame(X.groupby(encode_cols)[target_col].aggregate(aggr_funcs))
    meanDF = meanDF.reset_index()

    global_mean = X.groupby(encode_cols)[target_col].mean().mean()
    global_median = X.groupby(encode_cols)[target_col].median().median()

    label = target_col
    for i in encode_cols:
        label = label + i
    print("Getting", label, "wise demand..")

    dfcols = [[col for col in encode_cols], [i + label for i in aggr_funcs]]
    meanDF.columns = [item for sublist in dfcols for item in sublist]

    # We only care about the values in this stack.
    y = pd.merge(y, meanDF, on=encode_cols, how="left")
    # fill any missing values (in y not in X)
    y['mean' + label].fillna(global_mean, inplace=True)
    y['median' + label].fillna(global_median, inplace=True)

    y.index = y_idx

    #  Create col if not present and add y
    for col in dfcols[1]:
        if col not in encode_test:
            encode_test[col] = np.nan
        encode_test.ix[encode_test.index, col] = y[col]







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
