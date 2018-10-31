import sklearn.model_selection as skm


def evaluation_cv(mod, xtr, ytr, cv=10):
    return skm.cross_val_score(mod, xtr, ytr, cv)


def grid_search():
    pass