import numpy as np

#       class  sex age fare
keep = [0,     3,  4,  8]
keep += [1] # keeping labels
encoding = {'female': 0, 'male': 1, 'C': 0, 'Q': 1, 'S': 2}
data = np.loadtxt('../media/titanic.csv', skiprows=1, delimiter=',', dtype=object)
data = np.vstack([data[:, j] for j in keep]).T
data = data[np.sum(data == '', axis=1) == 0]  # removing rows with empty values
for k in encoding:
    data[data == k] = encoding[k]
data = np.ceil(data.astype(np.float64))
arg_test = np.argwhere(data[:, -1] == 1).flatten()[:50]
arg_test = np.hstack((arg_test, np.argwhere(data[:, -1] == 0).flatten()[:50]))
data_test = data[arg_test]
data = np.delete(data, arg_test, axis=0)
X, y = data[:, :-1], data[:, -1]
X_, y_ = data_test[:, :-1], data_test[:, -1]