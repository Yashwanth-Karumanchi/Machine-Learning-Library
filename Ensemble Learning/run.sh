#!/bin/sh

echo "Results for Adaboost"
python3 bank_adaboost.py

echo "Results for bagged tree"
python3 bagged_tree.py

echo "Results for biasvariance"
python3 bias_variance.py

echo "Results for randomforest"
python3 random_forest.py

echo "Results for bias variance for Random Tree vs Bagged Tree vs Single Tree"
python3 rt_vs_bt_vs_dt.py