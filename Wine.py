import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    stdsc = StandardScaler()
    lr = LogisticRegression(penalty=None, C=1.0)
    df_wine = pd.read_csv('https://archive.ics.'
                          'uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Etykiety klass', 'Alkohol', 'Kwas jabłkowy', 'Popiół',
                       'Zasadowość popiołu', 'Magnez', 'Fenol', 'Flawonidy',
                       'Febole nieflawonidowe', 'Proantocyjaniny',
                       'Intensywność koloru', 'Odcień', 'Transmitacja 280/515 nm',
                       'Prolina']
    print(df_wine.head())

    X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_train)

    lr.fit(X_train_std, y_train)
    print(lr.intercept_)