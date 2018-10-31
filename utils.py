import sklearn as sk


def evaluate_cv(X, y, model, cv=10):
    scores = sk.model_selection.cross_val_score(model, X, y, cv)
    return scores


def grid_search