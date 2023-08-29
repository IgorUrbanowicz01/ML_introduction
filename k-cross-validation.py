import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    le = LabelEncoder()
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(random_state=1))
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)

    print('Dokładność testu: %.3f' % pipe_lr.score(X_test, y_test))
