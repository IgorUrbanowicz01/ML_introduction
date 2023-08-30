import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, validation_curve, GridSearchCV


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    le = LabelEncoder()
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

    parameter_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    parameter_range2 = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'svc__C': parameter_range2,
                   'svc__kernel': ['linear']},
                  {'svc__C': parameter_range2,
                   'svc__gamma': parameter_range2,
                   'svc__kernel': ['rfb']}
                 ]
    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(random_state=1))

    pipe_lr2 = make_pipeline(StandardScaler(),
                             LogisticRegression(penalty='l2', random_state=1, max_iter=1000))
    pipe_lr.fit(X_train, y_train)
    pipe_scv = make_pipeline(StandardScaler(), SVC(random_state=1))
    gs = GridSearchCV(estimator=pipe_scv, param_grid=param_grid,
                      scoring='accuracy', cv=10, n_jobs=-1)
    y_pred = pipe_lr.predict(X_test)

    print('Dokładność testu: %.3f' % pipe_lr.score(X_test, y_test))

    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Podzbiór: %d, Rozkład klasy: %s, Dokładność: %.3f' % (k-1, np.bincount(y_train[train]), score))

    scores = cross_val_score(estimator=pipe_lr,
                             X=X_train,
                             y=y_train,
                             cv=10, n_jobs=1)
    print('Wynik dokładności sprawdzania: %s' % scores)

    train_size, train_scores, test_scores = learning_curve(estimator=pipe_lr2, X=X_train, y=y_train,
                                                           train_sizes=np.linspace(0.1, 1.0, 10),
                                                           cv=10, n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_size, train_mean, color='blue',
             marker='o', markersize=5,
             label='Dokładność uczenia')
    plt.fill_between(train_size, train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    plt.plot(train_size, test_mean, color='green',
             linestyle='--',marker='s', markersize=5,
             label='Dokładność walidacji')
    plt.fill_between(train_size, test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Liczba próbek uczących')
    plt.ylabel('Dokładność')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.show()

    train_scores, test_scores = validation_curve(estimator=pipe_lr2,
                                 X=X_train, y=y_train, param_name='logisticregression__C',
                                 param_range=parameter_range, cv=10)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)


    plt.plot(parameter_range, train_mean, color='blue',
             marker='o', markersize=5,
             label='Dokładność uczenia')
    plt.fill_between(parameter_range, train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    plt.plot(parameter_range, test_mean, color='green',
             linestyle='--', marker='s', markersize=5,
             label='Dokładność walidacji')
    plt.fill_between(parameter_range, test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Dokładność')
    plt.ylim([0.8, 1.03])
    plt.show()

    gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)