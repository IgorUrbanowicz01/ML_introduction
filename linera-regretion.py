import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                     'python-machine-learning-book-2nd-edition'
                     '/master/code/ch10/housing.data.txt',
                     header=None,
                     sep='\s+')
