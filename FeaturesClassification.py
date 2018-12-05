import pandas as pd
import numpy as np
import sklearn as sk
import visualization as vis
import utils
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


def lookup_best_alpha(augmented_data, y):
    selected_alpha = utils.lassoChoice(augmented_data.values, y)
    print(str(selected_alpha) + ' is the alpha parameter retained for Lasso')
    return selected_alpha


def lasso_pruning(alpha_parameter, augmented_data, y):
    features = list(augmented_data.columns)
    selected_lasso = sk.linear_model.Lasso(alpha=alpha_parameter)
    selected_lasso.fit(augmented_data.values, y)
    weights = selected_lasso.coef_
    features_to_drop = []
    for i, feature in enumerate(features):
        if weights[i] == 0:
            features_to_drop.append(feature)
    print(str(len(features_to_drop)) + ' features dropped out of ' + str(len(features)))
    return features_to_drop


def prune_features(features_to_drop, original_data):
    original_data.drop(columns=features_to_drop, inplace=True)
    return original_data


def clean():
    # 1. Localization features
    all_interactions_1 = ['order_freight_value', 'delivery_delay', 'product_weight_g']

    interactions = [(all_interactions_1[i], all_interactions_1[j]) \
                    for i in range(len(all_interactions_1)) for j in range(i + 1, len(all_interactions_1))]
    powers1 = [('delivery_delay', 2), ('product_weight_g', 2)]
    for feature, power in powers1:
        data = utils.polynomial_feature(data, feature, power)
    one_side_interactions_1 = zip(['delivery_delay'] * 4, ['lat', 'lng', 'seller_lat', 'seller_lng'])
    interactions += one_side_interactions_1
    for term1, term2 in interactions:
        data = utils.interaction_terms(data, term1, term2, term1 + '_' + term2)

    # 2. Feedback features
    data.loc[:, 'review_time'] = np.log(data.loc[:, 'review_time'])
    one_side_interactions_2 = zip(['review_time'] * 6, ['review_comment_message', 'review_score_1', \
                                                        'review_score_2', 'review_score_3', 'review_score_4',
                                                        'review_score_5'])
    for term1, term2 in one_side_interactions_2:
        data = utils.interaction_terms(data, term1, term2, term1 + '_' + term2)

    # 3. Product description
    one_side_interactions_3 = zip(['product_description_lenght'] * 3,
                                  ['product_photos_qty_0', 'product_photos_qty_1', 'product_photos_qty_2'])
    for term1, term2 in one_side_interactions_3:
        data = utils.interaction_terms(data, term1, term2, term1 + '_' + term2)

    # 4. Product dimensions
    powers4 = zip(['product_length_cm', 'product_height_cm', 'product_width_cm'], [2] * 3)
    for feature, power in powers4:
        data = utils.polynomial_feature(data, feature, power)
    one_side_interactions_4 = zip(['product_weight_g'] * 3,
                                  ['product_length_cm', 'product_height_cm', 'product_width_cm'])
    for term1, term2 in one_side_interactions_4:
        data = utils.interaction_terms(data, term1, term2, term1 + '_' + term2)

    # 5. Speed of processing
    data = utils.interaction_terms(data, 'approval_time', 'delivery_delay', 'approval_time' + '_' + 'delivery_delay')
    print(data.columns)

    # Saving
    data.to_pickle('./CleanedData/FeaturesClassification.pkl')
    # Try out with features suppression
    return


if __name__ == "__main__":
    data_clean = pd.read_pickle('./CleanedData/Features1.pkl')
    data_clean.drop(columns=["target"], inplace=True)
    data_clean = data_clean.drop("review_time_review_score_1", axis=1)
    data_clean = data_clean.drop("review_time_review_score_2", axis=1)
    data_clean = data_clean.drop("review_time_review_score_3", axis=1)
    data_clean = data_clean.drop("review_time_review_score_4", axis=1)
    data_clean = data_clean.drop("review_time_review_score_5", axis=1)

    data = pd.read_pickle('./CleanedData/dataset_train.pkl')
    data.drop(columns=['Unnamed: 0'], inplace=True)

    x = data.loc[:, ["review_score_1", "review_score_2", "review_score_3",
                     "review_score_4", "review_score_5"]].stack()
    x_clean = data_clean.loc[:, ["review_score_1", "review_score_2", "review_score_3",
                                 "review_score_4", "review_score_5"]].stack()
    review = pd.Series(pd.Categorical(x[x != 0].index.get_level_values(1)))
    review_clean = pd.Series(pd.Categorical(x_clean[x_clean != 0].index.get_level_values(1)))
    target = []
    target_clean = []
    for i in range(len(review)):
        n = review[i][-1]
        if int(n) >= 4:
            target.append(0)
        else:
            target.append(1)
    for i in range(len(review_clean)):
        n = review_clean[i][-1]
        if int(n) >= 4:
            target_clean.append(0)
        else:
            target_clean.append(1)

    data = data.assign(target=target)
    data_clean = data_clean.assign(target=target_clean)
    data.drop(columns=["review_score_1", "review_score_2", "review_score_3", "review_score_4", "review_score_5"],
              inplace=True)
    data_clean.drop(columns=["review_score_1", "review_score_2", "review_score_3", "review_score_4", "review_score_5"],
                    inplace=True)
    data_clean.to_pickle("Features2.pkl")

    X_train_simple = (data.loc[:, ["delivery_delay", "review_comment_message", "order_freight_value", "order_items_qty",
                                   "lng", "comment__True", "product_weight_g"]]).values
    y_train_simple = (data.loc[:, "target"]).values

    X_train = (data.drop("target", axis=1)).values
    y_train = (data.loc[:, "target"]).values
    X_train_c = (data_clean.drop("target", axis=1)).values
    y_train_c = (data_clean.loc[:, "target"]).values

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_c = min_max_scaler.fit_transform(X_train_c)
    X_train_c = preprocessing.scale(X_train_c)

    # K = range(1, 51)
    # F1_NN, F1_NN_clean, F1_NN_simple = [], [], []
    # for k in K:
    #     clf = KNeighborsClassifier(n_neighbors=k)
    #     clf_c = KNeighborsClassifier(n_neighbors=k)
    #     # F1_NN.append(sum(cross_val_score(clf, X_train, y_train, cv=10, scoring='f1')) / 10)
    #     F1_NN_clean.append(sum(cross_val_score(clf_c, X_train_c, y_train_c, cv=10, scoring='f1')) / 10)
    #     # F1_NN_simple.append(sum(cross_val_score(clf_c, X_train_simple, y_train_simple, cv=10, scoring='f1')) / 10)
    #
    # plt.figure(0)
    # plt.plot(K, F1_NN_clean)
    # plt.show()

    # clf = KNeighborsClassifier(n_neighbors=11)
    # clf_c = KNeighborsClassifier(n_neighbors=3)
    # print(sum(cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc')) / 10)
    # print(sum(cross_val_score(clf_c, X_train_c, y_train_c, cv=10, scoring='roc_auc')) / 10)
    # clf_s = KNeighborsClassifier(n_neighbors=33)
    # print(sum(cross_val_score(clf_s, X_train_simple, y_train_simple, cv=10, scoring='roc_auc')) / 10)
    # print(data_clean.info())

    # clf = LogisticRegression(penalty='l1', solver='liblinear', C=1, max_iter=100000)
    # for c in [0.01, 0.1, 1, 10, 100, 1000]:
    #     clf_c = LogisticRegression(penalty='l1', solver='saga', C=c, max_iter=1000)
    # clf_s = LogisticRegression(penalty='l1', solver='warn', C=1)
    # print(sum(cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc')) / 10)
    #     print(sum(cross_val_score(clf_c, X_train_c, y_train_c, cv=10, scoring='roc_auc')) / 10)
    # print(sum(cross_val_score(clf_s, X_train_simple, y_train_simple, cv=10, scoring='roc_auc')) / 10)

    # print(data.info())
    # print(data_clean.info())
    # clf = GaussianNB()
    # clf_c = GaussianNB()
    # clf_s = GaussianNB()
    # print(sum(cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc')) / 10)
    # print(sum(cross_val_score(clf_c, X_train_c, y_train_c, cv=10, scoring='roc_auc')) / 10)
    # print(sum(cross_val_score(clf_s, X_train_simple, y_train_simple, cv=10, scoring='roc_auc')) / 10)
    #
