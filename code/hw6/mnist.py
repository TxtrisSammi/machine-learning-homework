import numpy as np
from matplotlib import pyplot as plt

PATH = './media/t10k-images-idx3-ubyte'
RNG = np.random.default_rng(3)

with open(PATH, 'rb') as f:
    arr = bytearray(f.read())
    magic_number = int.from_bytes(arr[:4], byteorder='big')
    num_examples = int.from_bytes(arr[4:8], byteorder='big')
    rows = int.from_bytes(arr[8:12], byteorder='big')
    cols = int.from_bytes(arr[12:16], byteorder='big')
    data = np.array(arr[16:], dtype=np.int64).reshape(num_examples, rows, cols)

X = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

ep, K = 0.0001, 10
means = X[RNG.integers(0, X.shape[0], K)]  # initial means

# <!-- remove from here
fig = plt.figure(figsize=(12, 12))
fig.suptitle('First 100 images arranged as a 10 by 10 table.')
for i, cj in enumerate(X[:100]):
    ax = fig.add_subplot(10, 10, i+1)
    ax.imshow(cj.reshape(28, 28), cmap='grey')
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

plt.title('Data Point 324: Label?')
plt.imshow(data[324], cmap='gray')
plt.tight_layout()
plt.show()
# till here, and implement K-means -->

while error > 3.5:
    distances = np.sqrt(((X[:, np.newaxis, :] - means)**2).sum(axis=2))
    labels = np.argmin(distances, axis=1)

# im just gonna go back to my naming conventions for my own ease
    centroids = np.zeros(means.shape())
    
    for j in range(K):
        points_in_cluster = X[labels == j]
        if len(points_in_cluster) > 0:
            centroids[j] = np.mean(points_in_cluster, axis = 0)        











rows, columns = 2, 5
fig = plt.figure(figsize=(15, 7))
fig.suptitle('Converged centeroids.')
for i, cj in enumerate(means):
    ax = fig.add_subplot(rows, columns, i+1)
    ax.imshow(cj.reshape(28, 28), cmap='grey')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
