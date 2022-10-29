"""
unittests which use rpy2 and therefore need to be run on a local environment because
'Windows R has changed its registry structure, and it made the current rpy2's R install path detection obsolete'
https://stackoverflow.com/questions/72356173
/when-using-rpy-robjects-a-message-comes-up-unable-to-determine-r-home-winerr
"""

import os
import unittest

import numpy as np

# run this line before importing any rpy2 modules
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"  # change as needed
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

# Windows R has changed its registry structure and it made the current rpy2's R install path detection obsolete
# https://stackoverflow.com/questions/72356173
# /when-using-rpy-robjects-a-message-comes-up-unable-to-determine-r-home-winerr
# run this line before importing any rpy2 modules
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"  # change as needed
from rpy2.robjects.packages import importr
# R vector of strings
from sklearn.datasets import make_blobs

from trimkmeans.trimkmeans import TrimKMeans


class LocalTests(unittest.TestCase):
    """
    class for tests run on a local environment
    """

    def test_compare_to_r(self):
        """
        Compares the results to the r versions result on the same centroids
        :return: None
        """
        make_blob_data, _, make_blob_centers = make_blobs(n_samples=100,
                                                          centers=3,
                                                          random_state=42,
                                                          return_centers=True)
        trimkmeans = TrimKMeans(n_clusters=3, n_init=10, init=make_blob_centers)
        trimkmeans.fit(make_blob_data)
        py_labels = trimkmeans.predict(make_blob_data)
        trimcluster = importr('trimcluster')

        # https://rpy2.github.io/doc/latest/html/numpy.html
        # Create a converter that starts with rpy2's default converter
        # to which the numpy conversion rules are added.
        np_cv_rules = default_converter + numpy2ri.converter
        with localconverter(np_cv_rules) as cv:
            # Anything here and until the `with` block is exited
            # will use our numpy converter whenever objects are
            # passed to R or are returned by R while calling
            # rpy2.robjects functions.
            tkm1 = trimcluster.trimkmeans(data=make_blob_data, k=3, trim=0.1, runs=10, points=make_blob_centers)
            r_labels = tkm1['classification']
        # r labels are indexed at 1 so this is adjusted for comparison
        adjusted_py_labels = np.array(py_labels) + 1
        adjusted_r_labels = np.array(r_labels).astype(int)
        trim_indices_py = [i for i, x in enumerate(adjusted_py_labels) if x == 4]
        trim_indices_r = [i for i, x in enumerate(adjusted_r_labels) if x == 4]
        # cluster labels are given randomly, therefore only the index of the trimmed points is compared
        self.assertIsNone(np.testing.assert_array_equal(trim_indices_py, trim_indices_r))


if __name__ == '__main__':
    unittest.main()
