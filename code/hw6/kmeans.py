import numpy as np
from matplotlib import pyplot as plt

# don't change the seed
RNG = np.random.default_rng(3)
normal = RNG.multivariate_normal

t, n = 3, 2500
spread = np.identity(2)
X = normal([-t, t], np.identity(2), n)
X = np.vstack((X, normal([t, t], spread, n)))
X = np.vstack((X, normal([t, -t], spread, n)))
X = np.vstack((X, normal([-t, -t], spread, n))) 


def k_means(X, K, epsilon=(10**-5)):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, K, replace=False)
    centroids = X[indices]
    error = 1 # number irrelevant as long as it's greater than epsilon

    while error >= epsilon:
        distances = np.linalg.norm(X[:, None] - centroids, axis = 2)
        labels = np.argmin(distances, axis = 1)
        
        old_centroids = centroids.copy()

        for j in range(K):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                centroids[j] = np.mean(points_in_cluster, axis = 0)
        error = np.linalg.norm(centroids - old_centroids)
    return centroids, labels

for k in range(1, 5):
    centroids, labels = k_means(X, k)
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=1, alpha=0.3)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100, label='Centroids')
    plt.title(f'K-Means Clustering (K={k})')
    plt.legend()
    plt.show()