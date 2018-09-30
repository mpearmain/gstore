import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from src.python.space_configs import space_lightgbm, tune_model, gp_minimize, forest_minimize
from ml_metrics import rmse


train_df = pd.read_parquet('input/processed/train_encoded_features.parquet.gzip')
train_y = np.log1p(train_df["totals.transactionRevenue"])
y_clf = (train_df['totals.transactionRevenue'] > 0).astype(np.uint8)
train_id = train_df["fullVisitorId"]

train_df = train_df.drop(["totals.transactionRevenue", 'date'], axis=1)

test_df = pd.read_parquet('input/processed/test_encoded_features.parquet.gzip')
test_id = test_df["fullVisitorId"].values.astype(str)
test_df = test_df.drop(['date'], axis=1)

# Classify non-zero revenues

folds = GroupKFold(n_splits=10)

oof_clf_preds = np.zeros(train_df.shape[0])
sub_clf_preds = np.zeros(test_df.shape[0])

excluded_features = ['fullVisitorId']
train_features = [_f for _f in train_df.columns if _f not in excluded_features]

for fold_, (trn_, val_) in enumerate(folds.split(y_clf, y_clf, groups=train_df['fullVisitorId'])):
    trn_x, trn_y = train_df[train_features].iloc[trn_], y_clf.iloc[trn_]
    val_x, val_y = train_df[train_features].iloc[val_], y_clf.iloc[val_]

    clf = LGBMClassifier(n_jobs=-1,
                         n_estimators=10000,
                         random_state=56,
                         max_depth=8,
                         min_child_samples=40,
                         reg_alpha=0.4,
                         reg_lambda=0.1,
                         num_leaves=290,
                         learning_rate=0.01,
                         subsample=0.8,
                         colsample_bytree=0.9,
                         silent=True)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=25, verbose=25)

    oof_clf_preds[val_] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    print(roc_auc_score(val_y, oof_clf_preds[val_]))
    sub_clf_preds += clf.predict_proba(test_df[train_features],
                                       num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

roc_auc_score(y_clf, oof_clf_preds)

# Add PRedictions to data set
train_df['non_zero_proba'] = oof_clf_preds
test_df['non_zero_proba'] = sub_clf_preds

# Predict Revenues by folds.

for fold_, (trn_, val_) in enumerate(folds.split(train_y, train_y, groups=train_df['fullVisitorId'])):
    if fold_ == 0:
        trn_x, trn_y = train_df[train_features].iloc[trn_], train_y.iloc[trn_].fillna(0)
        val_x, val_y = train_df[train_features].iloc[val_], train_y.iloc[val_].fillna(0)
        print("Tuning LGBMRegressor")
        space = space_lightgbm()
        clf = LGBMRegressor(n_jobs=-1, random_state=56, objective='regression', verbose=-1)
        model = tune_model(trn_x, trn_y, val_x, val_y, clf,
                           space=space, metric=rmse, n_calls=50, min_func=forest_minimize)


oof_reg_preds = np.zeros(train_df.shape[0])
sub_reg_preds = np.zeros(test_df.shape[0])
importances = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds.split(train_y, train_y, groups=train_df['fullVisitorId'])):
    trn_x, trn_y = train_df[train_features].iloc[trn_], train_y.iloc[trn_].fillna(0)
    val_x, val_y = train_df[train_features].iloc[val_], train_y.iloc[val_].fillna(0)

    reg = model
    reg.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=50, verbose=50)
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')

    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(test_df[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / folds.n_splits


sub_df = pd.DataFrame({"fullVisitorId":test_id})
sub_df["PredictedLogRevenue"] = sub_reg_preds
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("input/output/sub_clf.csv", index=False)










