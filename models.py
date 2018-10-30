import sklearn as sk


def ols(X, y, fit_intercept=True, normalize=False):
    reg = sk.linear_model.LinearRegression(fit_intercept, normalize)
    reg.fit(X, y)
    return reg


def ols_ridge(X, y, alpha=.5, fit_intercept=True, normalize=False):
    reg = sk.linear_model.Ridge(alpha)
    reg.fit(X, y)






