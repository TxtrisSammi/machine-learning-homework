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
plt.title('Unlabelled Data Clusters')
plt.xlabel('x--axis')
plt.ylabel('y--axis')
plt.plot(X[:, 0], X[:, 1], 'k.', alpha=0.3)
plt.tight_layout()
plt.show()

