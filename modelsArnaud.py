import sklearn as sk
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
import models


def main(args):
    data = pd.read_pickle('./CleanedData/Features1.pkl')

    X_train = data.drop(['log_target', 'target'], axis=1).values
    log_y_train = data.loc[:, 'log_target'].values

    y_train = data.loc[:, 'target'].values
    if args.model == 'LightGBM':
        lgb_train = lgb.Dataset(X_train, y_train)
        params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'rmse'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }
        print('Starting training...')
        cv_results = lgb.cv(
        params,
        lgb_train,
        num_boost_round=10000,
        nfold=10,
        early_stopping_rounds=10,
        stratified=False,
        verbose_eval=100
        )
        print('CV Score with LightGBM:' + str(cv_results['rmse-mean'][-1]))
    elif args.model == 'LinearRegression':
            model = models.ols(X_train, log_y_train)
            y_pred = np.exp(model.predict(X_test))
    elif args.model == 'RidgeRegression':
            pass
    elif args.model == 'NearestNeighbors':
            pass
    else:
            raise Exception('Model asked for has not been implemented')

    # eval
    #print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='LightGBM', help="ModelToSelect")
    args = parser.parse_args()
    main(args)

