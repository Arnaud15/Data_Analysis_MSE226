import sklearn.linear_model as skl


def linear_regression(type, penalty_log='l2', alpha=0, l1_ratio=0, fit_intercept=True, normalize=False):
    dic = {}
    dic["ols"] = skl.LinearRegression(fit_intercept, normalize)
    dic['ridge'] = skl.Ridge(alpha, fit_intercept, normalize)
    dic["lasso"] = skl.Lasso(alpha, fit_intercept, normalize)
    dic["elasticNet"] = skl.ElasticNet(alpha, l1_ratio, fit_intercept, normalize, random_state=0)
    dic["logistic"] = skl.LogisticRegression(penalty=penalty_log, C=alpha, fit_intercept=fit_intercept)
    reg = dic[type]
    return reg