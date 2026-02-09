import numpy as np
from data import X, y, X_, y_

class NaiveBayes:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        labels = np.unique(self.y)
        prob = [self.pr_y_given_x(y, x) for y in labels]
        return np.argmax(prob)

    def pr_y_given_x(self, y, x):
        loggies = np.zeros(x.shape)
        for i, xi in enumerate(x):
            loggies[i] = np.log(self.pr_xi_given_y(xi, i, y))
        return np.log(self.pr_y(y)) + np.sum(loggies)

    def pr_xi_given_y(self, xi, i, y):
        given_y = self.X[self.y == y]
        count = np.sum(given_y[:, i] == xi)
        prob = count / given_y.shape[0]
        return prob if prob > 0 else 2**(-32)

    def pr_y(self, y):
        count = np.sum(self.y == y)
        total = self.y.shape[0]
        return count / total

model = NaiveBayes(X, y)
if __name__ == '__main__':
    preds = [model.predict(x) for x in X_]
    accuracy = np.sum(preds == y_) / y_.shape[0]
    print(f'Accuracy: {accuracy}')
