import time

import numpy as np
import pandas as pd


def add_new_category(x):
    """
    Aimed at 'trafficSource.keyword' to tidy things up a little
    """
    x = str(x).lower()
    if x == 'nan':
        return 'nan'
    x = ''.join(x.split())
    if r'provided' in x:
        return 'not_provided'
    if r'youtube' in x or r'you' in x or r'yo' in x or r'tub' in x or r'yout' in x or r'y o u' in x:
        return 'youtube'
    if r'google' in x or r'goo' in x or r'gle' in x:
        return 'google'
    else:
        return 'other'


# Dump cleaned data to parquets for later.
train_df = pd.read_parquet('input/cleaned/train.parquet.gzip')
test_df = pd.read_parquet('input/cleaned/test.parquet.gzip')

# Remove target col.
y_train = train_df['totals.transactionRevenue'].values
train_df = train_df.drop(['totals.transactionRevenue'], axis=1)

# Join datasets for rowise feature engineering.
trn_len = train_df.shape[0]
merged_df = pd.concat([train_df, test_df])

num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime"]
for col in num_cols:
    merged_df[col] = merged_df[col].astype(float)

merged_df['diff_visitId_time'] = merged_df['visitId'] - merged_df['visitStartTime']
merged_df['diff_visitId_time'] = (merged_df['diff_visitId_time'] != 0).astype(float)
merged_df['totals.hits'] = merged_df['totals.hits'].astype(float)

# Build Time based features.
merged_df['formated_date'] = pd.to_datetime(merged_df['date'], format='%Y%m%d')
merged_df['month'] = pd.DatetimeIndex(merged_df['formated_date']).month
merged_df['year'] = pd.DatetimeIndex(merged_df['formated_date']).year
merged_df['day'] = pd.DatetimeIndex(merged_df['formated_date']).day
merged_df['quarter'] = pd.DatetimeIndex(merged_df['formated_date']).quarter
merged_df['weekday'] = pd.DatetimeIndex(merged_df['formated_date']).weekday
merged_df['weekofyear'] = pd.DatetimeIndex(merged_df['formated_date']).weekofyear

merged_df['is_month_start'] = pd.DatetimeIndex(merged_df['formated_date']).is_month_start
merged_df['is_month_end'] = pd.DatetimeIndex(merged_df['formated_date']).is_month_end
merged_df['is_quarter_start'] = pd.DatetimeIndex(merged_df['formated_date']).is_quarter_start
merged_df['is_quarter_end'] = pd.DatetimeIndex(merged_df['formated_date']).is_quarter_end
merged_df['is_year_start'] = pd.DatetimeIndex(merged_df['formated_date']).is_year_start
merged_df['is_year_end'] = pd.DatetimeIndex(merged_df['formated_date']).is_year_end

merged_df['month_unique_user_count'] = merged_df.groupby('month')['fullVisitorId'].transform('nunique')
merged_df['day_unique_user_count'] = merged_df.groupby('day')['fullVisitorId'].transform('nunique')
merged_df['weekday_unique_user_count'] = merged_df.groupby('weekday')['fullVisitorId'].transform('nunique')

merged_df['visitStartTime'] = pd.to_datetime(merged_df['visitStartTime'], unit='s')
merged_df['hour'] = pd.DatetimeIndex(merged_df['visitStartTime']).hour
merged_df['minute'] = pd.DatetimeIndex(merged_df['visitStartTime']).minute

# Cleanup for keywords
merged_df['trafficSource.keyword'] = merged_df['trafficSource.keyword'].fillna('nan')
merged_df['trafficSource.keyword_groups'] = merged_df['trafficSource.keyword'].apply(add_new_category)

merged_df['browser_category'] = merged_df['device.browser'] + '_' + merged_df['device.deviceCategory']
merged_df['browser_operatingSystem'] = merged_df['device.browser'] + '_' + merged_df['device.operatingSystem']
merged_df['source_country'] = merged_df['trafficSource.source'] + '_' + merged_df['geoNetwork.country']
merged_df['log.visitNumber'] = np.log1p(merged_df['visitNumber'])
merged_df['log.totals.hits'] = np.log1p(merged_df['totals.hits'])
merged_df['totals.pageviews'] = merged_df['totals.pageviews'].astype(float).fillna(0)
merged_df['log.totals.pageviews'] = np.log1p(merged_df['totals.pageviews'])


merged_df["page_hits_ratio"] =  merged_df['visitNumber'] / (merged_df['totals.pageviews'] + 1)

# Drop old vars.
merged_df = merged_df.drop(['formated_date', 'visitId', 'sessionId', 'visitStartTime'], axis=1)

# Split data back to original data sets.
train_df = merged_df[:trn_len]
test_df = merged_df[trn_len:]
del merged_df

train_df['totals.transactionRevenue'] = y_train
print(set(list(train_df)) - set(list(test_df)))

train_df.to_parquet('input/processed/train_static_features.parquet.gzip', compression='gzip')
test_df.to_parquet('input/processed/test_static_features.parquet.gzip', compression='gzip')
