import sklearn as sk
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import random_projection
from sklearn import neighbors
import argparse
import models
from tqdm import tqdm
import visualization


def main(args):
    data = pd.read_pickle('./CleanedData/' + args.dataset + '.pkl')
    if args.dataset == 'dataset_train':
        # Data Preprocessing
        data.drop(columns=['Unnamed: 0'], inplace=True)
        data = data.assign(target=data.order_products_value / data.order_items_qty)
        data = data.assign(log_target=np.log(data.target))
        data.drop(columns=['order_products_value', 'order_items_qty'], inplace=True)
    X = data.drop(['log_target', 'target'], axis=1)
    log_Y = data.loc[:, 'log_target'].values
    Y = data.loc[:, 'target'].values
    sampled_rows = np.random.rand(X.shape[0]) < 0.9
    X_train = X[sampled_rows]
    X_test = X[~sampled_rows]
    log_Y_train = log_Y[sampled_rows]
    log_Y_test = log_Y[~sampled_rows]
    """
    BEST MODEL - LIGHT GBM
    """
    if args.model == 'LightGBM':
        lgb_train = lgb.Dataset(X_train, log_Y_train)
        lgb_eval = lgb.Dataset(X_test, log_Y_test, reference=lgb_train)
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
        gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)
        print('Saving model...')
        # save model to file
        gbm.save_model('./savedModels/bestRegressionModel.txt')
    """
    LINEAR REGRESSION MODEL
    """
    '''
    elif args.model == 'LinearRegression':
        model = models.linear_regression('ols')
        print(- sum(cross_val_score(model, X_train, log_y_train, cv=10, scoring='neg_mean_squared_error'))/10)
    """
    K NEAREST NEIGHBOURS
    """
    elif args.model == 'NearestNeighbors':
        scores = []
        transformer = random_projection.GaussianRandomProjection(n_components = 5)
        X_new = transformer.fit_transform(X_train)
        print(X_new.shape)
        for k_value in tqdm(range(5, 100, 5)):
            model = neighbors.KNeighborsRegressor(n_neighbors=k_value, weights='uniform')
            scores.append(- sum(cross_val_score(model, X_train, log_y_train, cv=10, scoring='neg_mean_squared_error'))/10)
        plt.figure(1)
        plt.plot(range(5, 100, 5), scores)
        plt.xlabel("k")
        plt.ylabel("RMSE")
        plt.savefig('./Saved_Plots/NNplotRegression.png')
        plt.close()
    """
    BASELINE
    """
    elif args.model == 'BaselineMean':
        print(np.mean((log_y_train - log_y_train.mean()) ** 2) ** 0.5)
    else:
        raise Exception('Model asked for has not been implemented')
    '''
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='LightGBM', help="ModelToSelect")
    parser.add_argument('--dataset', type=str, default='Features1', help="DatasetToSelect")
    args = parser.parse_args()
    main(args)

