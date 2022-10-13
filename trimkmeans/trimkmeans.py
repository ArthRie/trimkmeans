'''
the example in R is a single function which takes an object of class 'tkm' as a parameter
to be more in line with the KMeans implementation in Pythons SKlearn library the following implementation
is realized as a class
'''

import math
import random

import numpy as np
from sklearn.preprocessing import StandardScaler


def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


def last_or_inf(x):
    """
    :param x: list of euclidean distances
    :return: returns either the last value or infinity if the list is empty
    """
    try:
        return x[-1]
    except IndexError:
        return math.inf


class TrimKMeans:

    # replaced parameters countmode and printcrit with sklearns verbose level
    # replaced R's run and maxit Parameters with sklearns n_init and max_iter
    def __init__(self, n_clusters=8, trim=0.1, scaling=False, n_init=10, max_iter=300, verbose=0, random_state=None):
        self.n_clusters = n_clusters
        self.trim = trim
        self.scaling = scaling
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.opt_cutoff_ranges = None
        self.cluster_centers_ = None
        self.crit_val = math.inf

    def fit(self, X_train):
        if self.scaling:
            X_train = StandardScaler().fit_transform(X_train)
        if self.random_state:
            random.seed(self.random_state)
        for run in range(self.n_init):
            # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
            # then the rest are initialized w/ probabilities proportional to their distances to the first
            # Pick a random point from train data for first centroid
            centroids = [random.choice(X_train)]
            for _ in range(self.n_clusters - 1):
                # Calculate distances from points to the centroids
                dists = np.sum([euclidean(centroid, X_train) for centroid in centroids], axis=0)
                # Normalize the distances
                dists /= np.sum(dists)
                # Choose remaining points based on their distances
                new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
                centroids += [X_train[new_centroid_idx]]
            # Iterate, adjusting centroids until converged or until passed max_iter
            iteration = 0
            prev_centroids = None
            while np.not_equal(centroids, prev_centroids).any() and iteration < self.max_iter:
                # Sort each datapoint, assigning to nearest centroid
                sorted_points = [[] for _ in range(self.n_clusters)]
                sorted_dists = [[] for _ in range(self.n_clusters)]
                for x in X_train:
                    dists = euclidean(x, centroids)
                    centroid_idx = np.argmin(dists)
                    sorted_points[centroid_idx].append(x)
                    # save the distance for each point for trimming later
                    sorted_dists[centroid_idx].append(dists[centroid_idx])
                # sort the points in  each cluster based on their distances
                for idx, cluster in enumerate(sorted_points):
                    if len(cluster) > 0:
                        # combine the lists, sort them based on distances and seperate them again
                        # sort based on the first element of the tuple which is the distance
                        points_and_dists = [[x, y] for y, x in
                                            sorted(zip(sorted_dists[idx], cluster), key=lambda point: point[0])]
                        points_and_dists = list(zip(*points_and_dists))
                        sorted_dists[idx] = points_and_dists[1]
                        sorted_points[idx] = points_and_dists[0]
                # trim the n points
                for _ in range(0, math.floor(self.trim * len(X_train))):
                    # find the cluster with the max value
                    max_dist_cluster = np.argmax([last_or_inf(x) for x in sorted_dists])
                    # remove  the farthest point
                    sorted_points[max_dist_cluster] = sorted_points[max_dist_cluster][:-1]
                    sorted_dists[max_dist_cluster] = sorted_dists[max_dist_cluster][:-1]
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
                prev_centroids = centroids
                centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
                # safe the cutoff range which is the last point in a cluster not cut off
                cutoff_ranges = [last_or_inf(x) for x in sorted_dists]
                for i, centroid in enumerate(centroids):
                    if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                        centroids[i] = prev_centroids[i]
                iteration += 1
            # calculate the sum of all the distances
            new_crit_val = sum([sum(x) for x in sorted_dists])
            if self.verbose >= 1:
                print(f"Iteration {run} criterion value {new_crit_val}")
            if new_crit_val < self.crit_val:
                self.crit_val = new_crit_val
                self.cluster_centers_ = centroids
                self.opt_cutoff_ranges = cutoff_ranges

    def predict(self, X):
        '''

        :param X: list of datapoints to be clustered
        :returns: cluster centroids, cluster label for the datapoints and cutoff ranges for each cluster
        '''
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.cluster_centers_)
            centroid_idx = np.argmin(dists)
            # check if distance is smaller than cutoff of that cluster
            # if not, label n_clusters is given
            if self.opt_cutoff_ranges[centroid_idx] < dists[centroid_idx]:
                centroid_idxs.append(self.n_clusters)
            else:
                centroid_idxs.append(centroid_idx)
        if self.verbose >= 1:
            print(f"trimmed k-means: trim= {self.trim} , n_clusters= {self.n_clusters}")
            print(f"Classification (trimmed points are indicated by  {self.n_clusters} ):")
            print(centroid_idxs)
        return centroid_idxs
