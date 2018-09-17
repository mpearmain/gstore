import pandas as pd
import numpy as np
from sklearn import preprocessing
from src.python.space_configs import space_lightgbm, tune_model
from lightgbm import LGBMRegressor
from ml_metrics import rmse

# Dump cleaned data to parquets for later.
train_df = pd.read_parquet('input/cleaned/train.parquet.gzip')
test_df = pd.read_parquet('input/cleaned/test.parquet.gzip')

# label encode the categorical variables and convert the numerical variables to float
cat_cols = ["channelGrouping", "device.browser", "device.deviceCategory", "device.operatingSystem",
            "geoNetwork.city", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent",
            "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.gclId",
            "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", "trafficSource.referralPath", "trafficSource.source"]
num_cols = ["totals.hits", "totals.pageviews"]
for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)

# Split for train and validation based on date
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


space = space_lightgbm()
clf = LGBMRegressor(n_jobs=-1, random_state=56, objective='regression', verbose=-1)
model = tune_model(xtrain, xtrain_y, xvalid, xvalid_y, clf,
                   space=space, metric=rmse, n_calls=50, min_func=gp_minimize)

full = model.fit(train_df, y_train)

test_id = test_df["fullVisitorId"].values
test_df = test_df.drop(["fullVisitorId", "date"], axis=1)

pred_test = model.predict(test_df)

sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("input/output/baseline_lgb.csv", index=False)