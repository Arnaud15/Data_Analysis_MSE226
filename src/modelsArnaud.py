import sklearn as sk
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def plot_performances(parameter_name, error_name, errors, parameters):
	return

def main():
	train_data = pd.read_pickle('./CleanedData/dataset_train.pkl')
	test_data = pd.read_pickle('./CleanedData/dataset_test.pkl')

	X_train = train_data.drop('order_products_value', axis=1)
	y_train = train_data.loc[:,'order_products_value']
	X_test = test_data.drop('order_products_value', axis=1)
	y_test = test_data.loc[:, 'order_products_value']

	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

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
                early_stopping_rounds=5)
	print('Saving model...')
	# save model to file
	gbm.save_model('./savedModels/model.txt')

	print('Starting predicting...')
	# predict
	y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

	# eval
	print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
	return

if __name__=="__main__":
	main()




