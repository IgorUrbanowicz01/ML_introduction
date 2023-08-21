import pandas as pd
import numpy as np
from  sklearn.preprocessing import LabelEncoder, OneHotEncoder

if __name__ == '__main__':
    ohe = OneHotEncoder(categories=[0])
    class_le = LabelEncoder()
    df = pd.DataFrame([
        ['Green', 'M', 10.1, 'clas1'],
        ['Red', 'L', 13.5, 'clas2'],
        ['Blue', 'XL', 15.3, 'clas1']]
    )
    df.columns = ['Color', 'Size', 'Price', 'Class']
    print(df)
    size_mapping = {
        'XL':3,
        'L':2,
        'M':1
    }
    df['Size'] = df['Size'].map(size_mapping)
    print(df)
    class_mapping = {label:idx for idx, label in
                     enumerate(np.unique(df['Class']))}
    df['Class'] = df['Class'].map(class_mapping)
    print(df)
    y = class_le.fit_transform(df['Class'].values)
    print(y)



