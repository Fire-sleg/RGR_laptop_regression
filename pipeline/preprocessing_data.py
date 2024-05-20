import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')

def preprocess_train_data(ds: pd.DataFrame) -> pd.DataFrame:

    # Перейменувати колонку Ram на Ram(GB)
    ds = ds.rename(columns={'Ram': 'Ram(GB)'})

    # Замінити значення в колонці Ram(GB) на числове значення без тексту "GB"
    ds['Ram(GB)'] = ds['Ram(GB)'].str.replace('GB', '').astype(int)

    # Перейменувати колонку Weight на Weight(kg)
    ds = ds.rename(columns={'Weight': 'Weight(kg)'})

    # Замінити значення в колонці Weight(kg) на числове значення без тексту "kg"
    ds['Weight(kg)'] = ds['Weight(kg)'].str.replace('kg', '').astype(float)


    ds = ds.drop(['laptop_ID', 'Product'], axis = 1)

    # we create and train the encoder

    encoder = OneHotEncoder(categories='auto',
                        drop='first', # to return k-1, use drop=false to return k dummies
                        handle_unknown='ignore') # helps deal with rare labels

    one_hot_cols = ['Company','TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']

    encoder.fit(ds[one_hot_cols])

    with open('C:/D/Final Project/laptop regressor/pipeline/settings/ohe_encoder.pkl', 'wb') as pick:
        pickle.dump(encoder, pick)

    tmp = encoder.transform(ds[one_hot_cols])
    ohe_output = pd.DataFrame(tmp.toarray())
    ohe_output.columns = encoder.get_feature_names_out(one_hot_cols)

    ds = ds.drop(one_hot_cols, axis=1)
    ds = pd.concat([ds, ohe_output], axis=1)


    Ram_upper_limit, Ram_lower_limit = find_skewed_boundaries(ds, 'Ram(GB)', 4)
    outliers_Ram = np.where(ds['Ram(GB)'] > Ram_upper_limit, True,
                       np.where(ds['Ram(GB)'] < Ram_lower_limit, True, False))
    ds = ds.loc[~outliers_Ram, ]


    cols_to_scale = ['Inches', 'Ram(GB)', 'Weight(kg)']

    scaler = MinMaxScaler()
    ds[cols_to_scale] = scaler.fit_transform(ds[cols_to_scale])

    with open('C:/D/Final Project/laptop regressor/pipeline/settings/scaler.pkl', 'wb') as pick:
        pickle.dump(scaler, pick)

    ds.to_csv('C:/D/Final Project/laptop regressor/data/data_train.csv', index=False)

    return ds

def preprocess_testing_data(ds: pd.DataFrame) -> pd.DataFrame:
    # Перейменувати колонку Ram на Ram(GB)
    ds = ds.rename(columns={'Ram': 'Ram(GB)'})

    # Замінити значення в колонці Ram(GB) на числове значення без тексту "GB"
    ds['Ram(GB)'] = ds['Ram(GB)'].str.replace('GB', '').astype(int)

    # Перейменувати колонку Weight на Weight(kg)
    ds = ds.rename(columns={'Weight': 'Weight(kg)'})

    # Замінити значення в колонці Weight(kg) на числове значення без тексту "kg"
    ds['Weight(kg)'] = ds['Weight(kg)'].str.replace('kg', '').astype(float)


    ds = ds.drop(['laptop_ID', 'Product'], axis = 1)
    
    
    with open('C:/D/Final Project/laptop regressor/pipeline/settings/ohe_encoder.pkl', 'rb') as pick:
        encoder: OneHotEncoder = pickle.load(pick)

    one_hot_cols = ['Company','TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']

    tmp = encoder.transform(ds[one_hot_cols])
    ohe_output = pd.DataFrame(tmp.toarray())
    ohe_output.columns = encoder.get_feature_names_out(one_hot_cols)

    ds = ds.drop(one_hot_cols, axis=1)
    ds = pd.concat([ds, ohe_output], axis=1)


    Ram_upper_limit, Ram_lower_limit = find_skewed_boundaries(ds, 'Ram(GB)', 4)
    outliers_Ram = np.where(ds['Ram(GB)'] > Ram_upper_limit, True,
                       np.where(ds['Ram(GB)'] < Ram_lower_limit, True, False))
    ds = ds.loc[~outliers_Ram, ]


    cols_to_scale = ['Inches', 'Ram(GB)', 'Weight(kg)']


    with open('C:/D/Final Project/laptop regressor/pipeline/settings/scaler.pkl', 'rb') as pick:
        scaler: MinMaxScaler = pickle.load(pick)
    
    ds[cols_to_scale] = scaler.fit_transform(ds[cols_to_scale])
    ds[cols_to_scale].describe()
    ds.to_csv('C:/D/Final Project/laptop regressor/data/data_test.csv', index=False)
    return ds



def find_skewed_boundaries(ds, variable, distance):

    IQR = ds[variable].quantile(0.75) - ds[variable].quantile(0.25)

    lower_boundary = ds[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = ds[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary