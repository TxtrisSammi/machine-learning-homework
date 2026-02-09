import numpy as np
from data import X, y, X_, y_

men = X[:, 1] == 1
women = X[:, 1] == 0

total = (sum(men) + sum(women))

menPercent = (sum(y[men] == 1) / total) * 100
womenPercent = (sum(y[women] == 1) / total) * 100

print(menPercent)
print(womenPercent)