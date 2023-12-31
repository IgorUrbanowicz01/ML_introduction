import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_region(X, y, classifier, resoloution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resoloution),
                           np.arange(x2_min, x2_max, resoloution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolors='black')
class Adaline(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors_ = []
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def activation(self, X):
        return X
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    adal = Adaline(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(adal.cost_)+1), np.log10(adal.cost_), marker='x')
    ax[0].set_xlabel('Epoki')
    ax[0].set_ylabel('Log (suma kwadratów błędów)')
    ax[0].set_title('Adaline - Współczynnik uczenia 0.01')
    adal2 = Adaline(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(adal2.cost_) + 1), adal2.cost_, marker='x')
    ax[1].set_xlabel('Epoki')
    ax[1].set_ylabel('suma kwadratów błędów')
    ax[1].set_title('Adaline - Współczynnik uczenia 0.0001')
    plt.show()

    plt.clf()

    x_std = np.copy(X)
    x_std[:,0] = (X[:,0] - X[:,0].mean() / X[:, 0].std())
    x_std[:,1] = (X[:,1] - X[:,1].mean() / X[:, 1].std())

    adal3 = Adaline(n_iter=15, eta=0.01)
    adal3.fit(x_std, y)
    plot_decision_region(x_std, y, classifier=adal3)
    plt.title('Adaline - Gradient Prosty')
    plt.xlabel('Długość działki [standaryzacja]')
    plt.ylabel('Długość płatka [standaryzacja]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.plot(range(1,len(adal3.cost_) + 1), adal3.cost_, marker='o')
    plt.xlabel('Epoki')
    plt.ylabel('Suma kwadratów błędów')
    plt.show()