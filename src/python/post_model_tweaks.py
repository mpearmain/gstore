
import pandas as pd
# Read csv files

df1 = pd.read_csv('./input/output/simple_average_0.8.csv')
df2 = pd.read_csv('input/output/sub_clf.csv')

sub_df = pd.read_csv("./input/raw/sample_submission.csv")
p = 0.8

sub_df['PredictedLogRevenue'] = df1['PredictedLogRevenue'].values * p + df2['PredictedLogRevenue'].values * (1-p)

# Recode small values to zero
mask = sub_df['PredictedLogRevenue'] < 0.01
sub_df.loc[mask, 'PredictedLogRevenue'] = 0

sub_df.to_csv("input/output/simple_average_0.8_sub_clf_08.csv", index=False)