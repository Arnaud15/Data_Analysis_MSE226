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

    mask = np.random.rand(len(data)) < 0.8
    train_data = data[mask]
    test_data = data[~mask]

    X_train = train_data.drop(['log_target'], axis=1).values
    log_y_train = train_data.loc[:, 'log_target'].values

    X_test = test_data.drop(['log_target'], axis=1).values
    log_y_test = test_data.loc[:, 'log_target'].values
    y_test = test_data.loc[:, 'target'].values

    if args.model == 'LightGBM':
        lgb_train = lgb.Dataset(X_train, log_y_train)
        lgb_eval = lgb.Dataset(X_test, log_y_test, reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        print('Starting training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=20)
        print('Saving model...')
        # save model to file
        gbm.save_model('./savedModels/model.txt')

        print('Starting predicting...')
        # predict
        y_pred = np.exp(gbm.predict(X_test, num_iteration=gbm.best_iteration))
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
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='LightGBM', help="ModelToSelect")
    args = parser.parse_args()
    main(args)
def main(args):
    data = pd.read_pickle('./CleanedData/Features1.pkl')

    mask = np.random.rand(len(data)) < 0.8
    train_data = data[mask]
    test_data = data[~mask]

    X_train = train_data.drop(['target', 'log_target'],  axis=1).values
    log_y_train = train_data.loc[:,'log_target'].values

    X_test = test_data.drop(['target', 'log_target'], axis=1).values
    log_y_test = test_data.loc[:,'log_target'].values
    y_test = test_data.loc[:, 'target'].values

    if args.model == 'LightGBM':
        lgb_train = lgb.Dataset(X_train, log_y_train)
        lgb_eval = lgb.Dataset(X_test, log_y_test, reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

	print('Starting training...')
	# train
	gbm = lgb.train(params,
			lgb_train,
			num_boost_round=10000,
			valid_sets=lgb_eval,
			early_stopping_rounds=20)
	print('Saving model...')
	# save model to file
	gbm.save_model('./savedModels/model.txt')

	print('Starting predicting...')
	# predict
	y_pred = np.exp(gbm.predict(X_test, num_iteration=gbm.best_iteration))
	elif args.model == 'LinearRegression':
		model = models.ols(X_train, log_y_train)
		y_pred = np.exp(model.predict(X_test))
	elif args.model == 'RidgeRegression':
		pass
	elif args.model == 'NearestNeighbors':
		pass
	elif args.model == 'BaseLineFull':
		original_data = pd.read_pickle('./CleanedData/dataset_train.pkl')
		original_data.drop(columns=['Unnamed: 0'], inplace=True)
		original_data = original_data.assign(target= original_data.order_products_value / original_data.order_items_qty)
		original_data = original_data.assign(log_target= np.log(original_data.target))
		original_data.drop(columns=['order_products_value', 'order_items_qty'], inplace=True)
	else:
		raise Exception('Model asked for has not been implemented')

	# eval
	print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
	return

if __name__=="__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model',  type=str, default='LightGBM', help="ModelToSelect")
	args = parser.parse_args()
	main(args)


        print('Starting predicting...')
        # predict
        y_pred = np.exp(gbm.predict(X_test, num_iteration=gbm.best_iteration))
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
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='LightGBM', help="ModelToSelect")
    args = parser.parse_args()
    main(args)