"""
the example in R is a single function which takes an object of class 'tkm' as a parameter
to be more in line with the KMeans implementation in Pythons SKlearn library the following implementation
is realized as a class
"""
import random
import warnings
from math import inf, floor

import numpy as np
from sklearn.preprocessing import StandardScaler


def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    # list conversion to prevent deprecation warning Calling np.sum(generator) is deprecated
    gen = ((point - data) ** 2)
    return np.sqrt(np.sum(gen, axis=1))


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
    def __init__(self,
                 n_clusters=8,
                 trim=0.1,
                 scaling=False,
                 n_init=10,
                 max_iter=100,
                 verbose=0,
                 random_state=None,
                 init='k-means++'):
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
        self.init = init

    class ClusterPoint:
        """
        Inner Class used to group points together with the distance and cluster that was calculated by the algorithm
        """
        # https://stackoverflow.com/questions/53388451/how-to-speed-up-python-instance-initialization-for-millions-of-objects
        # 107.801 -> 92.025 cumtime for 10000 points
        __slots__ = ('points', 'cluster', 'dist')

        def __init__(self, points=None):
            self.points = points
            self.cluster = None
            self.dist = None

        # enable comparison for sort
        def __lt__(self, cp2):
            return self.dist < cp2.dist

        # for debugging
        def __repr__(self):
            return f"Point: {self.points}, Cluster: {self.cluster}, Distance: {self.dist}"

        # for testing
        def __eq__(self, cp2):
            if self.cluster == cp2.cluster and self.dist == cp2.dist:
                return np.array_equal(self.points, cp2.points)
            return False

    def __create_points(self, x_train, centroids):
        """
        Inner function used to group the transformation of raw point data to instances of inner class ClusterPoint
        :param x_train: list of datapoints
        :param centroids: centers of kmeans cluster
        :return: heapq of sorted ClusterPoint instances
        """
        sorted_points = np.array([self.ClusterPoint() for _ in range(len(x_train))])
        # Sort each datapoint, assigning to nearest centroid
        for idx, val in enumerate(x_train):
            sorted_points[idx].points = val
            # save the distance for each point for trimming later
            # distance is negated because heapq is implemented as a min stack
            dists = -1 * euclidean(val, centroids)
            sorted_points[idx].dist = max(dists)
            sorted_points[idx].cluster = np.argmax(dists)

        # trim the n points
        sorted_points.sort()
        for x in range(0, floor(self.trim * len(sorted_points))):
            sorted_points[x].cluster = len(centroids)

        return sorted_points

    def __calculate_cutoff(self, sorted_points):
        """
        # set the member variable cutoff range which is the last point in a cluster not cut off
        :param sorted_points: result of the __create_points() method
        :return:
        """
        self.opt_cutoff_ranges = [None] * self.n_clusters

        for i in range(self.n_clusters):
            # get max dist per cluster
            try:
                self.opt_cutoff_ranges[i] = min(x.dist for x in sorted_points if x.cluster == i)
            except ValueError:
                self.opt_cutoff_ranges[i] = -1 * inf

    def __compare_iterations(self, sorted_points, centroids, run):
        """
        Encapsulates the comparison between the different results of two iterations of trimmed_kmeans
        :param sorted_points: heapq of ClusterPoints
        :param centroids: centroids of the clusters
        :param run: number of iteration for verbose printing
        """
        # calculate the sum of all the distances
        new_crit_val = sum((c_p.dist for c_p in sorted_points))
        if self.verbose >= 1:
            print(f"Iteration {run} criterion value {new_crit_val}")
        if new_crit_val > self.crit_val:
            self.__calculate_cutoff(sorted_points)
            self.crit_val = new_crit_val
            self.cluster_centers_ = centroids

    def __check_data(self, data):
        """
        throws ValueError if the input data has the wrong shape
        :param self:
        :param data: list of data points to fit the trimmed kmeans to
        :return:
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(
                "Please convert input data to numpy array"
            )
        # If input is scalar raise error
        if len(data) == 0:
            raise ValueError(
                "Expected 2D array, array is empty"
            )
        # If input is 1D raise error
        if len(data[0]) == 0:
            raise ValueError(
                "Expected 2D array, got 1D array instead"
            )
        # If input n_rows is smaller than n_cluster raise Error
        trimmed_len = len(data) - floor(self.trim * len(data))
        if trimmed_len < self.n_clusters:
            raise ValueError(
                f"n_samples-trim={trimmed_len} should be >= n_clusters={self.n_clusters}."
            )

    def fit(self, x_train):
        """computes trimmed k means clustering
        https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
        :param x_train: list of datapoints
        """
        self.__check_data(x_train)
        if self.scaling:
            x_train = StandardScaler().fit_transform(x_train)
        if self.random_state:
            random.seed(self.random_state)

        for run in range(self.n_init):
            if isinstance(self.init, str) and self.init == "k-means++":
                # Pick a random point from train data for first centroid
                centroids = [random.choice(x_train)]
                for _ in range(self.n_clusters - 1):
                    # Calculate distances from points to the centroids
                    dists = sum((euclidean(centroid, x_train) for centroid in centroids))
                    # Normalize the distances
                    dists /= np.sum(dists)
                    # Choose remaining points based on their distances
                    new_centroid_idx, = np.random.choice(range(len(x_train)), size=1, p=dists)
                    centroids += [x_train[new_centroid_idx]]
            elif isinstance(self.init, str) and self.init == "random":
                # Randomly select centroid start points, uniformly distributed across the domain of the dataset
                min_point, max_point = np.min(x_train, axis=0), np.max(x_train, axis=0)
                centroids = [np.random.uniform(min_point, max_point) for _ in range(self.n_clusters)]
            elif isinstance(self.init, str):
                raise ValueError(f"{self.init} is not a recognized method, try 'k-means++', 'random' or passing"
                                 f"an inital centroid ndarray")
            else:
                # calculation for set centroids, only one iteration
                if self.n_init > 1:
                    warnings.warn("If initial mean vectors are specified, only one run will be calculated")
                centroids = self.init

            # Iterate, adjusting centroids until converged or until passed max_iter
            iteration = 0
            prev_centroids = None
            # heapq for points
            sorted_points = []
            while np.not_equal(centroids, prev_centroids).any() and iteration < self.max_iter:

                sorted_points = self.__create_points(x_train, centroids)
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
                # copy list by value[:]
                prev_centroids = centroids[:]
                for i in range(self.n_clusters):
                    with warnings.catch_warnings():
                        # clusters can be empty, the corresponding RuntimeWarning is suppressed here
                        # to prevent screen clutter
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        centroids[i] = np.mean([c_p.points for c_p in sorted_points if c_p.cluster == i], axis=0)
                for i, centroid in enumerate(centroids):
                    if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                        centroids[i] = prev_centroids[i]
                iteration += 1

            self.__compare_iterations(sorted_points, centroids, run)

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
