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
    data = pd.read_pickle('./CleanedData/args.dataset.pkl')
    if args.dataset == 'BaseFeatures':
        # Data Preprocessing
        data.drop(columns=['Unnamed: 0'], inplace=True)
        data = data.assign(target=data.order_products_value / data.order_items_qty)
        data = data.assign(log_target=np.log(data.target))
        data.drop(columns=['order_products_value', 'order_items_qty'], inplace=True)

    X_train = data.drop(['target'], axis=1).values
    y_train = data.loc[:, 'target'].values

    if args.model == 'LightGBM':
        lgb_train = lgb.Dataset(X_train, y_train)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'classification',
            'metric': {'f1'},
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
        model = models.linear_regression('ols')
        print(- sum(cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')) / 10)
    elif args.model == 'NearestNeighbors':
        scores = []
        for k_value in tqdm(range(5, 100, 5)):
            model = neighbors.KNeighborsRegressor(n_neighbors=k_value, weights='uniform')
            scores.append(
                - sum(cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')) / 10)
        plt.figure(1)
        plt.plot(range(5, 100, 5), scores)
        plt.xlabel("k")
        plt.ylabel("f1")
        plt.savefig('./Saved_Plots/NNplotRegression.png')
        plt.close()
    else:
        raise Exception('Model asked for has not been implemented')

    # eval
    # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='LightGBM', help="ModelToSelect")
    parser.add_argument('--dataset', type=str, default='Features1', help="DatasetToSelect")
    args = parser.parse_args()
    main(args)
