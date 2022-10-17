"""
the example in R is a single function which takes an object of class 'tkm' as a parameter
to be more in line with the KMeans implementation in Pythons SKlearn library the following implementation
is realized as a class
"""
import heapq
import random
from math import inf, floor

import numpy as np
from sklearn.preprocessing import StandardScaler


def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class TrimKMeans:
    """
    Optimizes the k-means algorithm by trimming a portion of the points.
    The implementation follows the algorithm of the r package but is modeled after the scikit-learn
    kmeans clustering code.
    The centroids are initialized using the "k-means++" method, where a random datapoint is selected as the first,
    then the rest are initialized with probabilities proportional to their distances to the first
    """

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
        self.crit_val = -1 * inf

    class ClusterPoint:
        """
        Inner Class used to group points together with the distance and cluster that was calculated by the algorithm
        """

        def __init__(self, points):
            self.points = points
            self.cluster = None
            self.dist = None

        # enable comparison for heapq
        def __lt__(self, cp2):
            return self.dist < cp2.dist

        # for debugging
        def __repr__(self):
            return f"Cluster: {self.cluster}, Distance: {self.dist}"

    def fit(self, x_train):
        """computes trimmed k means clustering
        :param x_train: list of datapoints
        """
        if self.scaling:
            x_train = StandardScaler().fit_transform(x_train)
        if self.random_state:
            random.seed(self.random_state)
        for run in range(self.n_init):
            # Pick a random point from train data for first centroid
            centroids = [random.choice(x_train)]
            for _ in range(self.n_clusters - 1):
                # Calculate distances from points to the centroids
                dists = np.sum((euclidean(centroid, x_train) for centroid in centroids), axis=0)
                # Normalize the distances
                dists /= np.sum(dists)
                # Choose remaining points based on their distances
                new_centroid_idx, = np.random.choice(range(len(x_train)), size=1, p=dists)
                centroids += [x_train[new_centroid_idx]]
            # Iterate, adjusting centroids until converged or until passed max_iter
            iteration = 0
            prev_centroids = None
            # heapq for points
            sorted_points = []
            while np.not_equal(centroids, prev_centroids).any() and iteration < self.max_iter:

                sorted_points = []
                # Sort each datapoint, assigning to nearest centroid
                for x in x_train:
                    c_p = self.ClusterPoint(points=x)
                    # save the distance for each point for trimming later
                    # distance is negated because heapq is implemented as a min stack
                    dists = -1 * euclidean(x, centroids)
                    c_p.dist = max(dists)
                    c_p.cluster = np.argmax(dists)
                    heapq.heappush(sorted_points, c_p)

                # trim the n points
                _ = [heapq.heappop(sorted_points) for _ in range(0, floor(self.trim * len(x_train)))]
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
                # copy list by value[:]
                prev_centroids = centroids[:]
                for i in range(self.n_clusters):
                    centroids[i] = np.mean([c_p.points for c_p in sorted_points if c_p.cluster == i], axis=0)
                for i, centroid in enumerate(centroids):
                    if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                        centroids[i] = prev_centroids[i]
                iteration += 1
            # calculate the sum of all the distances
            new_crit_val = sum((c_p.dist for c_p in sorted_points))
            if self.verbose >= 1:
                print(f"Iteration {run} criterion value {new_crit_val}")
            if new_crit_val > self.crit_val:
                # safe the cutoff range which is the last point in a cluster not cut off
                self.opt_cutoff_ranges = [None] * self.n_clusters
                while any(x is None for x in self.opt_cutoff_ranges):
                    c_p = heapq.heappop(sorted_points)
                    if not self.opt_cutoff_ranges[c_p.cluster]:
                        self.opt_cutoff_ranges[c_p.cluster] = c_p.dist
                self.crit_val = new_crit_val
                self.cluster_centers_ = centroids

    def predict(self, X):
        """
        :param X: list of datapoints to be clustered
        :returns: cluster centroids, cluster label for the datapoints and cutoff ranges for each cluster
        """
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.cluster_centers_)
            centroid_idx = np.argmin(dists)
            # check if distance is smaller than cutoff of that cluster
            # if not, label n_clusters is given
            if self.opt_cutoff_ranges[centroid_idx] * -1 < dists[centroid_idx]:
                centroid_idxs.append(self.n_clusters)
            else:
                centroid_idxs.append(centroid_idx)
        if self.verbose >= 1:
            print(f"trimmed k-means: trim= {self.trim} , n_clusters= {self.n_clusters}")
            print(f"Classification (trimmed points are indicated by  {self.n_clusters} ):")
        return centroid_idxs
