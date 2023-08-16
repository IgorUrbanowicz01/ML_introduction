import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from utils import plot_decision_regions, loadIris
import numpy as np

if __name__ == '__main__':
    X_train_std, X_test_std, y_train, y_test = loadIris()
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    knn.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx= range(105, 150))
    plt.xlabel('Długość płatka')
    plt.ylabel('Szerokość płatka')
    plt.legend(loc='upper left')
    plt.show()