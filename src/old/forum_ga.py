import json
import logging

import numpy as np
import pandas as pd
import xlearn as xl
from lightgbm import LGBMRegressor
from pandas.io.json import json_normalize
from pystacknet.pystacknet import StackNetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

IS_LOCAL = False
if (IS_LOCAL):
    PATH = "../input/google-analytics-customer-revenue-prediction/"
else:
    PATH = "./input/"

N_SPLITS = 3


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()
# the columns that will be parsed to extract the fields from the jsons
cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(train_series=None,
                  test_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1.,
                  noise_level=0.,
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


def read_parse_dataframe(file_name=None, nrows=None):
    logger.info('Start read parse')
    # full path for the data file
    path = PATH + file_name
    # read the data file, convert the columns in the list of columns to parse using json loader,
    # convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path,
                          converters={column: json.loads for column in cols_to_parse},
                          dtype={'fullVisitorId': 'str'},
                          nrows=nrows)
    # parse the json-type columns
    for col in cols_to_parse:
        # each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        # we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
    logger.info('Done with read parse')
    return data_df


def process_date_time(data_df):
    logger.info('Start date')

    data_df['date2'] = data_df['date']
    data_df['date'] = data_df['date'].astype(str)
    data_df["date"] = data_df["date"].apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:])
    data_df["date"] = pd.to_datetime(data_df["date"])
    data_df["year"] = data_df['date'].dt.year
    data_df["month"] = data_df['date'].dt.month
    data_df["day"] = data_df['date'].dt.day
    data_df["weekday"] = data_df['date'].dt.weekday
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['month_unique_user_count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
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


def process_categorical_columns(train_df=None, test_df=None):
    ## Categorical columns
    logger.info('Process categorical columns ...')
    num_cols = ['month_unique_user_count', 'day_unique_user_count', 'weekday_unique_user_count',
                'visitNumber', 'totals_hits', 'totals_pageviews',
                'mean_hits_per_day', 'sum_hits_per_day', 'min_hits_per_day', 'max_hits_per_day', 'var_hits_per_day',
                'mean_pageviews_per_day', 'sum_pageviews_per_day', 'min_pageviews_per_day', 'max_pageviews_per_day',
                'sum_pageviews_per_network_domain', 'count_pageviews_per_network_domain',
                'mean_pageviews_per_network_domain', 'max_pageviews_per_network_domain',
                'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'mean_hits_per_network_domain',
                'max_hits_per_network_domain',
                'medium_hits_mean', 'medium_hits_min', 'medium_hits_max', 'medium_hits_sum', 'medium_hits_var']

    not_used_cols = ["visitNumber", "date", "date2", "fullVisitorId", "sessionId",
                     "visitId", "visitStartTime", 'totals_transactionRevenue', 'trafficSource_referralPath']
    cat_cols = [col for col in train_df.columns if col not in num_cols and col not in not_used_cols]
    for col in cat_cols:
        logger.info('Process categorical columns:{}'.format(col))
        train_df[col + "mean"], test_df[col + "mean"] = target_encode(train_series=train_df[col],
                                                                      test_series=test_df[col],
                                                                      target=train_df['totals_transactionRevenue'],
                                                                      min_samples_leaf=100,
                                                                      smoothing=2,
                                                                      noise_level=0.01,
                                                                      aggr_func="mean")
        train_df[col + "median"], test_df[col + "median"] = target_encode(train_series=train_df[col],
                                                                          test_series=test_df[col],
                                                                          target=train_df['totals_transactionRevenue'],
                                                                          min_samples_leaf=100,
                                                                          smoothing=2,
                                                                          noise_level=0.01,
                                                                          aggr_func="median")
    train_df = train_df.drop(cat_cols, axis=1)
    test_df = test_df.drop(cat_cols, axis=1)

    return not_used_cols, train_df, test_df


def model(train_df=None, test_df=None, not_used_cols=None):
    logger.info('Start prepare model')
    train_df = train_df.sort_values('date')

    # format_str = '%Y%m%d'
    # train_df['date2'] = train_df['date2'].apply(lambda x: datetime.strptime(str(x), format_str))
    # split_date = '2017-05-31'
    # dev_df = train_df.loc[train_df['date2'] <= split_date]
    # valid_df = train_df.loc[train_df['date2'] > split_date]
    # dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
    # dev_df = dev_df.drop(not_used_cols, axis=1)
    # valid_y = np.log1p(valid_df["totals_transactionRevenue"].values)
    # valid_df = valid_df.drop(not_used_cols, axis=1)

    X = train_df.drop(not_used_cols, axis=1)
    y = train_df['totals_transactionRevenue']
    X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)

    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=20180917)

    ## Model
    logger.info('Start tuning model')
    # space = space_lightgbm()
    # clf = LGBMRegressor(n_jobs=-1, random_state=56, objective='regression', verbose=-1)
    # model = tune_model(dev_df, dev_y, valid_df, valid_y, clf,
    #                    space=space, metric=rmse, n_calls=50, min_func=forest_minimize)
    params = {"objective": "regression", "metric": "rmse", "max_depth": 8, "min_child_samples": 40, "reg_alpha": 0.4,
              "reg_lambda": 0.1, "num_leaves": 297, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.97}

    # Try Stacknet
    models = [
        ######## First level ########
        [RandomForestRegressor(n_estimators=500, random_state=1, n_jobs=-1),
         ExtraTreesRegressor(n_estimators=500, random_state=1, n_jobs=-1),

         LGBMRegressor(n_jobs=-1, random_state=56, objective='regression', max_depth=8, min_child_samples=40,
                       reg_alpha=0.4, reg_lambda=0.1, num_leaves=290, learning_rate=0.01, subsample=0.8,
                       colsample_bytree=0.9, n_estimators=520),
         xl.FMModel(task='reg', metric='rmse', block_size=800, lr=0.05, k=12, reg_lambda=0.05, init=0.1, fold=1,
                    epoch=50, stop_window=5, opt='ftrl', nthread=0, n_jobs=-1, alpha=0.05, beta=1,
                    lambda_1=0.1, lambda_2=0.1),
         Ridge(random_state=1)
         ],
        ######## Second level ########
        [RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)]
    ]

    model = StackNetRegressor(models, metric="rmse", folds=3, restacking=True, use_retraining=True,
                              random_state=12345, n_jobs=1, verbose=1)

    model.fit(X, y)
    prediction = model.predict(X_test)

    return prediction


def submission(test_df=None, prediction=None):
    # Submission
    logger.info('Prepare submission')
    submission = test_df[['fullVisitorId']].copy()
    submission.loc[:, 'PredictedLogRevenue'] = prediction
    submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x: 0.0 if x < 0 else x)
    submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
    grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
    grouped_test.to_csv('submission.csv', index=False)
    logger.info('Done with submission')


def main(sum_of_logs=False, nrows=None):
    # Feature processing
    # Load data
    train_df = read_parse_dataframe('raw/train.csv', nrows=nrows)
    train_df = process_date_time(train_df)
    test_df = read_parse_dataframe('raw/test.csv')
    test_df = process_date_time(test_df)

    # Drop columns
    train_df, test_df = drop_convert_columns(train_df, test_df)

    # Features engineering
    train_df = process_format(train_df)
    train_df = process_device(train_df)
    train_df = process_totals(train_df)
    train_df = process_geo_network(train_df)
    train_df = process_traffic_source(train_df)

    test_df = process_format(test_df)
    test_df = process_device(test_df)
    test_df = process_totals(test_df)
    test_df = process_geo_network(test_df)
    test_df = process_traffic_source(test_df)

    # Categorical columns
    not_used_cols, train_df, test_df = process_categorical_columns(train_df, test_df)

    # Model
    prediction = model(train_df, test_df, not_used_cols)

    # Submission
    submission(test_df, prediction)


if __name__ == "__main__":
    main(nrows=None)


