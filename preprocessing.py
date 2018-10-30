import pandas as pd
import numpy as np

pd.set_option('display.expand_frame_repr', False)


def load_csv_save_binary(data_names):
    csv = '.csv'
    direc = './data/'
    pkl = '.pkl'
    for name in data_names:
        csv_file = pd.read_csv(direc + name + csv)
        csv_file.to_pickle(direc + name + pkl)


def load_binary(data_names):
    direc = './data/'
    pkl = '.pkl'
    res = []
    for name in data_names:
        res.append(pd.read_pickle(direc + name + pkl))
    return res


def train_test_split(data, train_size=0.8):
    msk = np.random.rand(len(data)) < train_size
    return data[msk], data[~msk]


def cleaning():
    data_names = ['geolocation_olist_public_dataset', 'olist_public_dataset_v2', 'olist_public_dataset_v2_customers',
                  'payments_olist_public_dataset', 'olist_classified_public_dataset',
                  'product_measures_olist_public_dataset_', 'sellers_olist_public_dataset_']
    [geo, main, custom, payments, labeled, measures, sellers] = load_binary(data_names)

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

    delivery_delay = (
                main.loc[:, 'order_delivered_customer_date'] - main.loc[:, 'order_estimated_delivery_date']).astype(
        'timedelta64[s]')
    main = main.assign(delivery_delay=delivery_delay)

    review_time = (main.loc[:, 'review_answer_timestamp'] - main.loc[:, 'review_creation_date']).astype(
        'timedelta64[s]')
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


if __name__ == "__main__":
    data = ['geolocation_olist_public_dataset', 'olist_public_dataset_v2', 'olist_public_dataset_v2_customers',
            'payments_olist_public_dataset', 'olist_classified_public_dataset',
            'product_measures_olist_public_dataset_', 'sellers_olist_public_dataset_']
    # load_csv_save_binary(data)
    # load_csv_save_binary()
    # load_binary()
    # cleaning()
    pass