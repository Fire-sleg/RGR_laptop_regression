import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from preprocessing_data import preprocess_train_data
import model_best_hyperparams

def train_model(file_name: str = 'train.csv', model_name: str = 'ridge'):
    # loading data
    data = pd.read_csv('C:/D/Final Project/laptop regressor/data/' + file_name)

    # preprocessing data
    data = preprocess_train_data(data)

    # split data
    X = data.drop(columns=['Price_euros'])
    Y = data['Price_euros']

    # models
    models = {
        'ridge': Ridge(**model_best_hyperparams.params_ridge),
    }

    # training model
    model = models[model_name]
    model.fit(X, Y)

    # saving model
    with open(f'C:/D/Final Project/laptop regressor/models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)