import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from preprocessing import cleaning


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


if __name__ == '__main__':
    # correlation_matrix()
    # histogram("order_freight_value")
    # scatter_plots("order_freight_value", "order_products_value")
    # histogram("review_score", False)
    # box_plots("review_score", "review_comment_message")
    # box_plots("review_score", "delivery_delay")
    # histogram('product_photos_qty', False)
    # box_plots("product_photos_qty", "order_products_value")
    histogram("order_products_value")
    # pass