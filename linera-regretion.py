import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
                             loss='absolute_error', residual_threshold=5.0,
                             random_state=0)
    ransac.fit(X, y)
    inliner_mask = ransac.inlier_mask_
    outliner_mask = np.logical_not(inliner_mask)
    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inliner_mask], y[inliner_mask],
                c = 'steelblue', edgecolors='white',
                marker='o', label='Ptr. inline')
    plt.scatter(X[outliner_mask], y[outliner_mask],
                c='limegreen', edgecolors='black', lw=2,
                marker='s', label='Ptr. outline')
    plt.plot(line_X, line_y_ransac, color='black', lw=2)
    plt.ylabel('Price int thousand dollars')
    plt.xlabel('Average amount of rooms')
    plt.legend(loc='upper right')
    plt.show()

    print('tilt: %.3f' % ransac.estimator_.coef_[0])
    print('Cross point: %.3f' % ransac.estimator_.intercept_)

    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    X_train, X_test, y_tain, y_test = train_test_split(X, y,
                                                       train_size=0.3, random_state=0)
    slr = LinearRegression()
    slr.fit(X_train, y_tain)

    y_train_predict = slr.predict(X_train)
    y_test_predict = slr.predict(X_test)

    plt.scatter(y_train_predict, y_train_predict - y_tain,
                c='steelbule', marker='o', edgecolors='white',
                label='Training data')
    plt.scatter(y_test_predict, y_test_predict - y_tain,
                c='limegreen', marker='s', edgecolors='white',
                label='Testing data')
    plt.xlabel('Expected values')
    plt.ylabel('Residual values')
    plt.legend(loc='upper right')



    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
    plt.xlim([-10, 50])
    plt.show()

    print('MSE for train data: %.f3, test data: %.3f' % (
        mean_squared_error(y_tain, y_train_predict), mean_squared_error(y_test, y_test_predict)))
    print('R^2 for train data: %.f3, test data: %.3f' % (
        r2_score(y_tain, y_train_predict), r2_score(y_test, y_test_predict)))
