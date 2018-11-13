import sklearn.linear_model

def linear_regression(X, y, type, alpha=0, l1_ratio=0, fit_intercept=True, normalize=False):
    dic = {}




def ols(X, y, fit_intercept=True, normalize=False):
    reg = sklearn.linear_model.LinearRegression(fit_intercept, normalize)
    reg.fit(X, y)
    return reg


def ols_ridge(X, y, alpha=.5, fit_intercept=True, normalize=False):
    reg = sklearn.linear_model.Ridge(alpha)
    reg.fit(X, y)


'''

reg = linear_model.Lasso(alpha = 0.1)
>>> reg.fit([[0, 0], [1, 1]], [0, 1])

 X, y = make_regression(n_features=2, random_state=0)
>>> regr = ElasticNet(random_state=0)
>>> regr.fit(X, y)
''' 
