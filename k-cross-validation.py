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
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib import cm
from scipy import interp

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
                   'svc__kernel': ['rbf']}
                 ]
    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(random_state=1))

    pipe_lr2 = make_pipeline(StandardScaler(),
                             LogisticRegression(penalty='l2', random_state=1, max_iter=1000))
    pipe_lr3 = make_pipeline(StandardScaler(),
                             LogisticRegression(penalty='l2', random_state=1, C=100.0))
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

    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print('Dokładność testu: %.3f' % clf.score(X_test, y_test))

    pipe_scv.fit(X=X_train, y=y_train)
    y_pred = pipe_scv.predict(X=X_test)
    confMat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confMat)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confMat, cmap=cm.Blues, alpha=0.3)
    for i in range(confMat.shape[0]):
        for j in range(confMat.shape[1]):
            ax.text(x=j, y=i,
                    s=confMat[i,j],
                    va='center', ha='center')
    plt.xlabel('Przewidywana etykieta')
    plt.ylabel('Rzeczywista etykieta')
    plt.show()

    X_train2 = X_train[:, [4, 14]]
    cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
    fig = plt.figure(figsize=(7, 5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas = pipe_lr3.fit(X_train2[train], y_train[train],
                              ).predict_proba(X_train2[test])
        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='Podzbiór nr %d (obszar = %0.2f)' % (i+1, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Losowae zgadywanie')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Uśredniona krzywa ROC (obszar = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':',
             color='black', label='Doskonała skuteczność')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Odsetki fałszywie pozytywnych')
    plt.ylabel('Odsetki prawdziwie pozytywnych')
    plt.legend(loc='lower right')
    plt.show()