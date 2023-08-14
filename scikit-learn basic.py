import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import plot_decision_regions

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print("Etykiet klas: ", np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    print('Licba etykiet: w zbiorze y: ', np.bincount(y))
    print('Licba etykiet: w zbiorze y_train: ', np.bincount(y_train))
    print('Licba etykiet: w zbiorze y_test: ', np.bincount(y_test))
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print('Nieporawidło sklasyfikowane próbki: %d' % (y_test != y_pred).sum())
    print('Dokładnośc: %.2f' % accuracy_score(y_test, y_pred))
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std, y=y_combined,
                          classifier=ppn, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [standaryzowana]')
    plt.ylabel('Szerokość płatka [standaryzowana]')
    plt.legend(loc='upper left')
    plt.show()