import sklearn as sk


def ols_regression(X, y, fit_intercept=True, normalize=False):
    reg = sk.linear_model.LinearRegression(fit_intercept, normalize)
    reg.fit(X, y)




