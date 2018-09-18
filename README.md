# Pipeline to run

1. `clean_and_load.py` to take raw dataset and do some basic cleanups to get into a tabular state to be able to work 
later.  No transformation of data have occured at this stage, it is simply removing and recoding features.
2. `build_static_features` a baseline script to build row-wise features suitable for all ML algos.

At this stage we also dump out dev, valid and train and test sets.
From this point on doing K-fold and groupby must be done on each fold correctly to prevent leakage.

3. `build_groupby_fold_features.py` Script for calculating means, sums, counts, etc by groups by fold

 
4. Build classification model for if a user is likely to purchase
5. `build_dirty_cat.py` for categorical variables encoding
6. `build_`  