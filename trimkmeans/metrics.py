from sklearn import metrics


# remove all the trimmed points (label = n_centers) since they would interfier with the Silhouette score
def trimmed_kmeans_metric_unsupervised(x_train, labels, metric):
    # combine data and label so they can be deleted together with the order still in tact
    zipped = zip(x_train, labels)
    zipped_list = list(zipped)
    # the highest label is for the trimmed points
    centers = max(labels)
    removed_trimmed = [i for i in zipped_list if i[1] != centers]
    x_train_removed_trimmed, classification_removed_trimmed = list(zip(*removed_trimmed))
    if metric == 'silhouette_score':
        return metrics.silhouette_score(x_train_removed_trimmed, classification_removed_trimmed, metric='euclidean')
    else:
        raise ValueError("metric must be 'silhouette_score'")


# remove all the trimmed points labels (label = n_centers) since they would interfier with the supervised metric
def trimmed_kmeans_metric_supervised(true_labels, labels, metric):
    # combine data and label so they can be deleted together with the order still in tact
    zipped = zip(true_labels, labels)
    zipped_list = list(zipped)
    # the highest label is for the trimmed points
    centers = max(labels)
    removed_trimmed = [i for i in zipped_list if i[1] != centers]
    true_label_removed_trimmed, classification_removed_trimmed = list(zip(*removed_trimmed))
    if metric == 'rand_score':
        return metrics.rand_score(true_label_removed_trimmed, classification_removed_trimmed)
    elif metric == 'completeness_score':
        return metrics.completeness_score(true_label_removed_trimmed, classification_removed_trimmed)
    else:
        raise ValueError("metric must be either 'rand_score' or 'completeness_score'")
