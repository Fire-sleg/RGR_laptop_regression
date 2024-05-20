import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing_data import preprocess_testing_data

def test_model(file_name: str = 'new_input.csv', model_name: str = 'ridge'):
    # loading data
    data = pd.read_csv('C:/D/Final Project/laptop regressor/data/' + file_name)

    # preprocessing data
    data = preprocess_testing_data(data)

    # split data
    X = data.drop(columns=['Price_euros'])
    Y = data['Price_euros']

    # testing model
    with open(f'C:/D/Final Project/laptop regressor/models/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    df = pd.DataFrame({'Y': Y, 'Predictions': predictions})
    # saving predictions
    pd.DataFrame(df).to_csv('C:/D/Final Project/laptop regressor/data/predictions.csv', index=False)

    #metrics
    mse = mean_squared_error(Y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y, predictions)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R²: {r2}')