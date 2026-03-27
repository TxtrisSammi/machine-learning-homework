import numpy as np 
from matplotlib import pyplot as plt
PATH = '../../media/BostonHousing.csv'


Xy = np.genfromtxt(
	PATH,
	delimiter=',',
	skip_header=1,
	dtype=np.float64,
	converters = {3: lambda s: float(s[1:-1])}
)

Xy = np.column_stack((Xy[:, 12], Xy[:, 13], np.ones(shape=Xy.shape[0]), Xy[:, -1]))
test = Xy[-100:]
train = Xy[:-100]
X, y = train[:, :-1], train[:, -1]
_X, _y = test[:, :-1], test[:, -1]


lmbArr = np.arange(-9, 20)
RMSEArr = []

for lmb in lmbArr:
	beta = np.linalg.inv(X.T @ X + lmb * np.identity(X.shape[1])) @ X.T @ y 
	J = (_X @ beta - _y).T @ (_X @ beta - _y)
	RMSE = np.sqrt(J / _X.shape[0])
	RMSEArr.append(RMSE)
# Prints RMSE at lambda = 0; created after observing the graph
print(RMSEArr[9])

plt.plot(lmbArr, RMSEArr)
plt.title("Ridge Regression")
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.xticks(lmbArr)
plt.tight_layout()
plt.show()
