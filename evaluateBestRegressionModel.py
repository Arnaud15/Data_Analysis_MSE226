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


def main():
    data = pd.read_pickle('./CleanedData/Features1_test.pkl')
    X_test = data.drop(['log_target', 'target'], axis=1)
    log_Y_test = data.loc[:, 'log_target'].values
    Y_test = data.loc[:, 'target'].values
    """
    BEST MODEL - LIGHT GBM
    """
    bst = lgb.Booster(model_file='./savedModels/bestRegressionModel.txt')
    log_Y_pred = bst.predict(X_test)
    Y_pred = np.exp(log_Y_pred)

    rmse_model = np.mean( (Y_pred - Y_test) ** 2) ** 0.5

    print(rmse_model)
    return

if __name__ == "__main__":
    main()