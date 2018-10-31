import pandas as pd
from src.utils import evaluation_cv
from src.models import linear_regression
from sklearn.model_selection import cross_val_score


def load_data(target):
    train_data = pd.read_pickle("../CleanedData/dataset_train.pkl")
    test_data = pd.read_pickle("../CleanedData/dataset_test.pkl")
    X_train = (train_data.drop(target, axis=1)).values
    y_train = (train_data.loc[:, target]).values
    #y_train = y_train.reshape(len(y_train), 1)
    X_test = (test_data.drop(target, axis=1)).values
    y_test = (test_data.loc[:, target]).values
    #y_test = y_test.reshape(len(y_test), 1)
    print(X_train.shape, y_train.shape, type(X_train), type(y_train))
    return X_train, y_train, X_test, y_test


def main_regression():
    X_train, y_train, X_test, y_test = load_data("order_freight_value")


def main_classification():
    X_train, y_train, X_test, y_test = load_data("review_score")


if __name__ == '__main__':
    models = {"data": "dataset_train",
              "method": "ols",
              "type": "ridge",
              # Linear regression parameters
              "alpha": [0.5],
              "l1_ratio": [0.5],
              "penalty_log": 'l2'}
    X_train, y_train, X_test, y_test = load_data("order_freight_value")
    reg = linear_regression("ols")
    print(cross_val_score(reg, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    # pass
