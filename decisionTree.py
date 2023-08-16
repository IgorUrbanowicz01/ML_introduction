import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import loadIris, plot_decision_regions

def gini(p):
    return p *(1 - p) + (1 - p)*(1 - (1-p))
def entropy(p):
    return -p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
    return  1 - np.max([p, (1 - p)])

if __name__ == '__main__':
    X_train_std, X_test_std, y_train, y_test = loadIris()
    x = np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e*0.5 if e else None for e in ent]
    err = [error(p) for p in x]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c in zip([ent, sc_ent,
                              gini(x), err],['Entropia','Entropia [skalowana]'
                              'Wskaźnik Giniego','Błąd klasyfikacji'],
                             ['-', '-', '--', '-'],['red', 'blue', 'lightgreen', 'grey', 'cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=5, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='red', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='red', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(p=1)')
    plt.ylabel('Wskaźnik zanieczyszczenia')
    plt.show()

    forest = RandomForestClassifier(criterion='gini', n_estimators=25,
                                    random_state=1, n_jobs=2)
    forest.fit(X_train_std, y_train)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined_std, y_combined,classifier=forest, test_idx=range(105,150))
    plt.xlabel('Długość płatka')
    plt.ylabel('Szerokość płatka')
    plt.legend(loc='upper left')
    plt.show()