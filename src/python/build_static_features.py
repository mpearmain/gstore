import time
from datetime import datetime

import pandas as pd

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

# Drop old vars.
merged_df = merged_df.drop(['date', 'formated_date', 'visitId', 'sessionId', 'visitStartTime',
                            'formated_visitStartTime'], axis =1)

# Split data back to original data sets.
train_df = merged_df[:trn_len]
test_df = merged_df[trn_len:]

train_df['totals.transactionRevenue'] = y_train
print(set(list(train_df)) - set(list(test_df)))

# Dump cleaned data to parquets for later.
train_df.to_parquet('input/processed/train_static_features.parquet.gzip', compression='gzip')
test_df.to_parquet('input/processed/test_static_features.parquet.gzip', compression='gzip')
