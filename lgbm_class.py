import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
import argparse
import models
from tqdm import tqdm


def main(args):
    data = pd.read_pickle('./CleanedData/Features2.pkl')
    X = data.drop(['target'], axis=1).values
    Y = data.loc[:, 'target'].values
    sampled_rows = np.random.rand(X.shape[0]) < 0.9
    X_train = X[sampled_rows]
    X_test = X[~sampled_rows]
    Y_train = Y[sampled_rows]
    Y_test = Y[~sampled_rows]

    if args.model == 'LightGBM':
        lgb_train = lgb.Dataset(X_train, Y_train)
        lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
        params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }
        print('Starting training...')
        gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)
        print('Saving model...')
        # save model to file
        gbm.save_model('./savedModels/bestClassificationModel.txt')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='LightGBM', help="ModelToSelect")
    parser.add_argument('--dataset', type=str, default='Features1', help="DatasetToSelect")
    args = parser.parse_args()
    main(args)
