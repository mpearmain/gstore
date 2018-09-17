import pandas as pd
import numpy as np
from datetime import datetime
from src.python.space_configs import space_lightgbm, tune_model
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from lightgbm import LGBMRegressor
from ml_metrics import rmse


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