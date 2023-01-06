import cProfile
import pstats

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from src.trimkmeans.trimkmeans import TrimKMeans

if __name__ == "__main__":
    # Create a dataset of 2D distributions
    CENTERS = 5
    X_train, true_labels = make_blobs(n_samples=10000, centers=CENTERS, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)
    # Fit centroids to dataset
    trimkmeans = TrimKMeans(n_clusters=CENTERS, verbose=1)
    profiler = cProfile.Profile()
    profiler.enable()
    trimkmeans.fit(X_train)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
