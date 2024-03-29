"""
Metrics for the trimkmeans algorithm
These are wrapper functions that remove the trimmed cluster marked by the highest label integer
"""

from numpy import var
from sklearn import metrics


def sum_variances(data, labels, cluster):
    """
    :param data: list of datapoints with trimmed points removed
    :param labels: list of labels generated by trimkmeans.predict() with trimmed points removed
    :param cluster: amount of clusters in the dataset
    :return: sum of variances in each cluster
    """
    # combine data and label so the label can be matched to the point
    zipped = zip(data, labels)
    zipped_list = list(zipped)
    sum_var = 0
    for i in range(cluster):
        sum_var += var([x[0] for x in zipped_list if x[1] == i])
    return sum_var


def score_even_distribution(data, labels, cluster):
    """
    :param data: list of datapoints with trimmed points removed
    :param labels: list of labels generated by trimkmeans.predict() with trimmed points removed
    :param cluster: amount of clusters in the dataset
    :return: product of count of points in each cluster
    """
    # combine data and label so the label can be matched to the point
    zipped = zip(data, labels)
    zipped_list = list(zipped)
    prod_cnt = 0
    for i in range(cluster):
        prod_cnt *= len([x[0] for x in zipped_list if x[1] == i])
    return prod_cnt


def trimmed_kmeans_metric_unsupervised(x_train, labels, metric):
    """
    remove all the trimmed points (label = n_centers) since they would interfere with the Silhouette score
    :param x_train: list of datapoints
    :param labels: list of labels generated by trimkmeans.predict()
    :param metric: name of the metric which should be evaluated from ['silhouette_score', 'sed', 'sv']
    :return:
    """
    # combine data and label so they can be deleted together with the order still in tact
    zipped = zip(x_train, labels)
    zipped_list = list(zipped)
    # the highest label is for the trimmed points
    centers = max(labels)
    removed_trimmed = [i for i in zipped_list if i[1] != centers]
    x_train_removed_trimmed, classification_removed_trimmed = list(zip(*removed_trimmed))
    if metric == 'silhouette_score':
        return metrics.silhouette_score(x_train_removed_trimmed, classification_removed_trimmed, metric='euclidean')
    if metric == 'sed':
        return score_even_distribution(x_train_removed_trimmed, classification_removed_trimmed, centers)
    if metric == 'sv':
        return sum_variances(x_train_removed_trimmed, classification_removed_trimmed, centers)

    raise ValueError("metric must be either 'silhouette_score', 'sed' or 'sv")


#
def trimmed_kmeans_metric_supervised(true_labels, labels, metric):
    """
        remove all the trimmed points labels (label = n_centers) since they would interfier with the supervised metric
        :param true_labels: list of true labels of a dataset, for example generated by sklearn.dataset.make_blobs()
        :param labels: list of labels generated by trimkmeans.predict()
        :param metric: name of the metric which should be evaluated from ['rand_score','completeness_score']
        :return:
        """
    # combine data and label so they can be deleted together with the order still in tact
    zipped = zip(true_labels, labels)
    zipped_list = list(zipped)
    # the highest label is for the trimmed points
    centers = max(labels)
    removed_trimmed = [i for i in zipped_list if i[1] != centers]
    true_label_removed_trimmed, classification_removed_trimmed = list(zip(*removed_trimmed))
    if metric == 'rand_score':
        return metrics.rand_score(true_label_removed_trimmed, classification_removed_trimmed)
    if metric == 'completeness_score':
        return metrics.completeness_score(true_label_removed_trimmed, classification_removed_trimmed)
    raise ValueError("metric must be either 'rand_score' or 'completeness_score'")
