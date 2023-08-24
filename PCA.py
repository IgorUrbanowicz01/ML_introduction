from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_decision_regions

if __name__ == '__main__':

    sc = StandardScaler()
    df_wine = pd.read_csv('https://archive.ics.'
                          'uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=0, stratify=y)
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print(eigen_vals)

    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in
               sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='Pojedyńcza wariacja wyjaśniona')
    plt.step(range(1,14), cum_var_exp, alpha=0.5, where='mid',  label='Łączna wariacja wyjaśniona')
    plt.ylabel('Współczynnik wariacji wyjaśnionej')
    plt.xlabel('Indeks głównej składowej')
    plt.legend(loc='best')
    plt.show()

    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key= lambda k: k[0], reverse=True)

    print(eigen_pairs)

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))

    print(w)

    X_train_std_pca = X_train_std.dot(w)

    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_std_pca[y_train==l, 0],
                    X_train_std_pca[y_train==l, 1],
                    c=c, label=l, marker=m)
    plt.ylabel('GS 2')
    plt.xlabel('GS 1')
    plt.legend(loc='lower left')
    plt.show()

    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.ylabel('GS 2')
    plt.xlabel('GS 1')
    plt.legend(loc='lower left')
    plt.show()

    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.ylabel('GS 2')
    plt.xlabel('GS 1')
    plt.legend(loc='lower left')