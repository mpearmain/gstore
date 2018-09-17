# Dump cleaned data to parquets for later.
train_df = pd.read_parquet('input/processed/train_static_features.parquet.gzip')
test_df = pd.read_parquet('input/processed/test_static_features.parquet.gzip')

# Split for train and validation based on date
format_str = '%Y%m%d'
train_df['date'] = train_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))
split_date ='2017-05-31'
xtrain = train_df.loc[train_df['date'] <= split_date]
xvalid = train_df.loc[train_df['date'] > split_date]

# take the log(1+x) of the values for better accuracy.
xtrain_y = np.log1p(xtrain["totals.transactionRevenue"].values)
xtrain_id = xtrain["fullVisitorId"].values
xtrain = xtrain.drop(["totals.transactionRevenue", "fullVisitorId", "date"], axis=1)

xvalid_y = np.log1p(xvalid["totals.transactionRevenue"].values)
xvalid_id = xvalid["fullVisitorId"].values
xvalid = xvalid.drop(["totals.transactionRevenue", "fullVisitorId", "date"], axis=1)

y_train = np.log1p(train_df["totals.transactionRevenue"].values)
train_id = train_df["fullVisitorId"].values
train_df = train_df.drop(["totals.transactionRevenue", "fullVisitorId", "date"], axis=1)
