import numpy as np
from matplotlib import pyplot as plt

PATH = '../../media/anderson.tsv'
classes = ['setosa', 'versicolor', 'virginica']
data = np.genfromtxt(PATH, delimiter = '\t', dtype = str)
features = data[0]
Xy = data[1:].astype(np.float64)
X = Xy[:, :-1]
y = Xy[:, -1].astype(np.int64)

# plotting
# if __name__ = '__main__':
# 	plt.title('Iris Data')
#	plt.xlabel(features[0])
#	plt.ylabel(features[1])
#	handles, _ = plt.scatter(X[:, 0], x[:, 1], c=y).legend_elements()
#	plt.legend(handles, classes)
#	plt.tight_layout()
#	plt.show()

X = np.expand_dims(X, axis = -1)

training_idx = np.concatenate((
	np.arange(35),
	50 + np.arange(35),
	100 + np.arange(35)
))

X_ = np.delete(X, training_idx, axis = 0)
y_ = np.delete(y, training_idx, axis = 0)
X = X[training_idx]
y = y[training_idx]

# one-hot encoding the training labels
I = np.identity(len(np.unique(y)))
y = np.array([I[int(t)] for t in y])

# row vectors to column vectors
y = np.expand_dims(y, axis=-1)

print(I)
print(y)
