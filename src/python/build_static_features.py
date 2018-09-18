import time
from datetime import datetime

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

merged_df['diff_visitId_time'] = merged_df['visitId'] - merged_df['visitStartTime']
merged_df['diff_visitId_time'] = (merged_df['diff_visitId_time'] != 0).astype(int)
merged_df['totals.hits'] = merged_df['totals.hits'].astype(int)

# Build Time based features.
format_str = '%Y%m%d'
merged_df['formated_date'] = merged_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))
merged_df['month'] = merged_df['formated_date'].apply(lambda x: x.month)
merged_df['quarter_month'] = merged_df['formated_date'].apply(lambda x: x.day // 8)
merged_df['day'] = merged_df['formated_date'].apply(lambda x: x.day)
merged_df['weekday'] = merged_df['formated_date'].apply(lambda x: x.weekday())

merged_df['formated_visitStartTime'] = merged_df['visitStartTime'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
merged_df['formated_visitStartTime'] = pd.to_datetime(merged_df['formated_visitStartTime'])
merged_df['visit_hour'] = merged_df['formated_visitStartTime'].apply(lambda x: x.hour)

# Cleanup for keywords
merged_df['trafficSource.keyword'] = merged_df['trafficSource.keyword'].fillna('nan')
merged_df['trafficSource.keyword'] = merged_df['trafficSource.keyword'].apply(add_new_category)

# Drop old vars.
merged_df = merged_df.drop(['formated_date', 'visitId', 'sessionId', 'visitStartTime',
                            'formated_visitStartTime'], axis=1)

# Split data back to original data sets.
train_df = merged_df[:trn_len]
test_df = merged_df[trn_len:]
del merged_df

train_df['totals.transactionRevenue'] = y_train
print(set(list(train_df)) - set(list(test_df)))


# Here we create the dev, valid, split for CVs to use in modelling later.
# These splits are based on 3 week validation
# Split for train and validation based on date
train_df['date'] = train_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))
split_date = '2017-05-24'  # 3 weeks
xdev = train_df.loc[train_df['date'] <= split_date]
xvalid = train_df.loc[train_df['date'] > split_date]

# Dump cleaned data to parquets for later.
xdev.to_parquet('input/processed/dev_static_features.parquet.gzip', compression='gzip')
xvalid.to_parquet('input/processed/valid_static_features.parquet.gzip', compression='gzip')
train_df.to_parquet('input/processed/train_static_features.parquet.gzip', compression='gzip')
test_df.to_parquet('input/processed/test_static_features.parquet.gzip', compression='gzip')
