import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import plot_decision_region
class Adaline(object):

    def __init__(self, eta=0.01, n_iter=50, shuffle = True , random_state=0):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X , y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.rave().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def activation(self, X):
        return X
        pass
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m + 1)
        self.w_initialized = True
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    # standaryczacja danych
    x_std = np.copy(X)
    x_std[:, 0] = (X[:, 0] - X[:, 0].mean() / X[:, 0].std())
    x_std[:, 1] = (X[:, 1] - X[:, 1].mean() / X[:, 1].std())

    ada = Adaline(n_iter=15, eta=0.01, random_state=1)
    ada.fit(x_std, y)
    plot_decision_region(x_std, y, classifier=ada)
    plt.title('Adaline - Stochastyczny spadek wzdłuż gradientu')
    plt.xlabel('Długość działki [standaryzowana]')
    plt.ylabel( 'Długość płatka [standaryzowana]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='x')
    plt.xlabel('Epoki')
    plt.ylabel('Średni koszt')
    plt.show()