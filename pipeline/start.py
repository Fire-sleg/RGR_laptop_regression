import pandas as pd
import pickle
from train_model import train_model
from test_model import test_model
from separate_data import separate_data

separate_data()
model = 'ridge'
print(model)
train_model(file_name="train.csv",model_name=model)
test_model(file_name='test.csv',model_name=model)