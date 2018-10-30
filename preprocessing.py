import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', False)


def load_csv_save_binary():
    geo = pd.read_csv('./data/geolocation_olist_public_dataset.csv')
    main = pd.read_csv('./data/olist_public_dataset_v2.csv')
    custom = pd.read_csv('./data/olist_public_dataset_v2_customers.csv')
    payments = pd.read_csv('./data/olist_public_dataset_v2_payments.csv')
    labeled = pd.read_csv('./data/olist_classified_public_dataset.csv')
    measures = pd.read_csv('./data/product_measures_olist_public_dataset_.csv')
    sellers = pd.read_csv('./data/sellers_olist_public_dataset_.csv')
    geo.to_pickle("./data/geolocation.pkl")
    main.to_pickle("./data/main_olist.pkl")
    custom.to_pickle("./data/customers.pkl")
    payments.to_pickle("./data/payments.pkl")
    labeled.to_pickle("./data/classified_olis.pkl")
    measures.to_pickle("./data/product_measures.pkl")
    sellers.to_pickle("./data/sellers.pkl")


def load_binary():
    geo = pd.read_pickle("./data/geolocation.pkl")
    main = pd.read_pickle("./data/main_olist.pkl")
    custom = pd.read_pickle("./data/customers.pkl")
    payments = pd.read_pickle("./data/payments.pkl")
    labeled = pd.read_pickle("./data/classified_olis.pkl")
    measures = pd.read_pickle("./data/product_measures.pkl")
    sellers = pd.read_pickle("./data/sellers.pkl")
    return main, labeled, payments, custom, geo, measures, sellers


def cleaning():
    main, labeled, payments, custom, geo, measures, sellers = load_binary()

    # Get one pair of coordinates for each zip code
    geo = geo.loc[:, ['zip_code_prefix', 'lat', 'lng']].groupby('zip_code_prefix').mean()
    if geo.isnull().values.any():
        print("Missing values in the geolocation table.")
    else:
        print("OK ! No missing values in the geolocation table.")

    # Convert time format
    time_columns = ['order_purchase_timestamp', 'order_aproved_at',
                    'order_estimated_delivery_date', 'order_delivered_customer_date',
                    'review_answer_timestamp', 'review_creation_date']
    main.loc[:, time_columns] = pd.to_datetime(main.loc[:, time_columns].stack()).unstack()
    main.sort_values(by='order_purchase_timestamp')
    main = main.groupby('order_id').last()

    # Create new duration variables
    approval_time = (main.loc[:, 'order_aproved_at'] - main.loc[:, 'order_purchase_timestamp']).astype('timedelta64[s]')
    main = main.assign(approval_time=approval_time)

    delivery_delay = (main.loc[:, 'order_delivered_customer_date'] - main.loc[:, 'order_estimated_delivery_date']).astype('timedelta64[s]')
    main = main.assign(delivery_delay=delivery_delay)

    review_time = (main.loc[:, 'review_answer_timestamp'] - main.loc[:, 'review_creation_date']).astype('timedelta64[s]')
    main = main.assign(review_time=review_time)

    # Deal with missing values
    main.dropna(axis=0, how='any', subset=['order_aproved_at'], inplace=True)
    main.loc[:, 'delivery_delay'] = main.loc[:, 'delivery_delay'].fillna(0)

    # Merge main dataset with geo table and remove rows with missing coordinates
    main = pd.merge(main, geo, left_on='customer_zip_code_prefix', right_index=True, how='left', sort=False)
    main.dropna(axis=0, how='any', subset=['lat'], inplace=True)

    # Remove useless covariates
    columns_to_remove = ['order_status', 'order_purchase_timestamp', 'order_aproved_at',
                         'order_estimated_delivery_date', 'order_delivered_customer_date',
                         'customer_id', 'product_name_lenght', 'review_id', 'review_creation_date',
                         'review_answer_timestamp', 'product_id']
    main.drop(labels=columns_to_remove, axis=1, inplace=True)

    # Modify string covariates
    main.loc[:, 'review_comment_title'] = main.loc[:, 'review_comment_title'].str.len() > 0
    main.loc[:, 'review_comment_message'] = main.loc[:, 'review_comment_message'].str.len()
    main.loc[:, 'review_comment_message'] = main.loc[:, 'review_comment_message'].fillna(0)

    return main


def correlation_matrix():
    main = cleaning()

    main.drop(labels=['customer_zip_code_prefix'], axis=1, inplace=True)

    corr = main.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(15, 11))
    cmap = sb.diverging_palette(220, 10, as_cmap=True)
    sb.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.show()


def histogram(column, kde=True):
    main = cleaning()
    sb.distplot(main.loc[:, column], kde=kde)
    plt.show()


def scatter_plots(c1, c2):
    main = cleaning()
    sb.jointplot(main.loc[:, c1], main.loc[:, c2])
    plt.show()


def box_plots(c1, c2):
    main = cleaning()
    sb.boxplot(main.loc[:, c1], main.loc[:, c2])
    plt.show()


if __name__ == "__main__":
    load_csv_save_binary()
    # load_binary()
    # cleaning()
    # correlation_matrix()
    # histogram("order_freight_value")
    # scatter_plots("order_freight_value", "order_products_value")
    # histogram("review_score", False)
    # box_plots("review_score", "review_comment_message")
    # box_plots("review_score", "delivery_delay")