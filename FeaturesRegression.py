import pandas as pd
import numpy as np
import sklearn as sk
import visualization as vis
import utils
import seaborn as sb
import matplotlib.pyplot as plt


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


def main():
    # Data Import
    data = pd.read_pickle('./CleanedData/dataset_train.pkl')
    # Data Preprocessing
    data.drop(columns=['Unnamed: 0'], inplace=True)
    data = data.assign(target=data.order_products_value / data.order_items_qty)
    data = data.assign(log_target=np.log(data.target))
    data.drop(columns=['order_products_value', 'order_items_qty'], inplace=True)

    # 1. Localization features
    all_interactions_1 = ['order_freight_value', 'delivery_delay', 'product_weight_g']

    interactions = [(all_interactions_1[i], all_interactions_1[j]) \
                    for i in range(len(all_interactions_1)) for j in range(i + 1, len(all_interactions_1))]
    powers1 = [('delivery_delay', 2), ('product_weight_g', 2)]
    for feature, power in powers1:
        data = utils.polynomial_feature(data, feature, power)
    '''
    one_side_interactions_1 = zip(['delivery_delay'] * 4, ['lat', 'lng', 'seller_lat', 'seller_lng'])
    interactions += one_side_interactions_1
    for term1, term2 in interactions:
        data = utils.interaction_terms(data, term1, term2, term1 + '_' + term2)
    '''
    # 2. Feedback features
    data.loc[:, 'review_time'] = np.log(data.loc[:, 'review_time'])
    one_side_interactions_2 = zip(['review_time'] * 6, ['review_comment_message', 'review_score_1',
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
    print(data.info())

    # Saving
    data.to_pickle('./CleanedData/Features2.pkl')
    # Try out with features suppression
    return


if __name__ == "__main__":
    main()
