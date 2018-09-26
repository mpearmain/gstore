import pandas as pd
import numpy as np
from datetime import datetime
from src.python.space_configs import space_lightgbm, space_xlearn, tune_model
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from sklearn import preprocessing
from lightgbm import LGBMRegressor
from ml_metrics import rmse
import xlearn as xl


# Load different data sources.
dev_df = pd.read_parquet('input/processed/dev_dynamic_features.parquet.gzip')
dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
dev_df = dev_df.drop(["totals.transactionRevenue", 'date'], axis=1)

valid_df = pd.read_parquet('input/processed/valid_dynamic_features.parquet.gzip')
valid_y = np.log1p(valid_df["totals.transactionRevenue"].values)
valid_df = valid_df.drop(["totals.transactionRevenue", 'date'], axis=1)

train_df = pd.read_parquet('input/processed/train_dynamic_features.parquet.gzip')
train_y = np.log1p(train_df["totals.transactionRevenue"].values)
train_df = train_df.drop(["totals.transactionRevenue", 'date'], axis=1)

test_df = pd.read_parquet('input/processed/test_dynamic_features.parquet.gzip')
test_id = test_df["fullVisitorId"].values.astype(str)
test_df = test_df.drop(['date'], axis=1)

cat_cols = list(train_df.select_dtypes(['object']))

for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(dev_df[col].values.astype('str')) + list(valid_df[col].values.astype('str')))
    dev_df[col] = lbl.transform(list(dev_df[col].values.astype('str')))
    valid_df[col] = lbl.transform(list(valid_df[col].values.astype('str')))


for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

#
# space = space_lightgbm()
# clf = LGBMRegressor(n_jobs=-1, random_state=56, objective='regression', verbose=-1)
# model = tune_model(dev_df, dev_y, valid_df, valid_y, clf,
#                    space=space, metric=rmse, n_calls=100, min_func=forest_minimize)


space = space_xlearn()
reg = xl.FMModel(task='reg', metric='rmse', n_jobs=12, random_state=42, nthread=None)
model = tune_model(dev_df, dev_y, valid_df, valid_y, reg,
                   space=space, metric=rmse, n_calls=25, min_func=gp_minimize)



full = model.fit(train_df, train_y)
pred_test = full.predict(test_df)

sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("input/output/extra_features_FM.csv", index=False)