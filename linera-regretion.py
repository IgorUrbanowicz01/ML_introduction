import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor, LinearRegression

class LinearRegressionGD:

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

def lin_ragplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolors='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None

if __name__ == '__main__':

    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                     'python-machine-learning-book-2nd-edition'
                     '/master/code/ch10/housing.data.txt',
                     header=None,
                     sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    cols = ['LSTAT', 'INDUS', 'NOX', 'CRIM', 'MEDV', 'B']
    sns.pairplot(df[cols], height=2.5)
    plt.tight_layout()
    plt.show()

    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True,
                     square=True, fmt='.2f',
                     annot_kws={'size':15},
                     xticklabels=cols,
                     yticklabels=cols)
    plt.show()

    X = df[['RM']].values
    y = df['MEDV'].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)
    sns.reset_orig()
    plt.plot(range(1, lr.n_iter+1), lr.cost_)
    plt.xlabel('Epos')
    plt.ylabel('Sum of square errors')
    plt.show()

    lin_ragplot(X_std, y_std, lr)
    plt.xlabel('Avrage rooms per Hause [RM] (standardize)')
    plt.ylabel('Avrage price per Hause [MEDV] (standardize)')
    plt.show()

    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100, min_samples=50,
                             loss='absolute_loss', residual_threshold=5.0,
                             random_state=0)
    ransac.fit(X, y)
