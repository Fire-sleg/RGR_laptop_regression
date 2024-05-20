import pandas as pd
from sklearn.model_selection import train_test_split
def separate_data():
    ds = pd.read_csv('C:/D/Final Project/laptop regressor/data/laptop_price.csv', encoding='cp1251')

    train, test = train_test_split(ds, train_size=0.8, test_size=0.2)

    train.to_csv('C:/D/Final Project/laptop regressor/data/train.csv', index=False)
    test.to_csv('C:/D/Final Project/laptop regressor/data/test.csv', index=False)

