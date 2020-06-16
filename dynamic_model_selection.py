from numpy.random import seed
from tensorflow import set_random_seed
from math import sqrt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import array
from keras.models import InputLayer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import Input, Model
from keras import regularizers
from keras import optimizers
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
impor os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--observations', default=None)
parser.add_argument("--true_values", default=None)
parser.add_argument('--forecasts_folder', default=None)

args = parser.parse_args()

seed(42)
set_random_seed(42)

def smape(a, b):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()

df_series = pd.read_csv(args.observations)
df_series = df_series.drop(['Unnamed: 0'],axis=1)

fh = 90

#Remove observations to be forecasted
first = True
c = 0
data_series = df_series.values
for i in range(len(data_series)):   
    
    if i % 1000 == 0:
        print("Preparing row {}".format(i))
    
    ts = data_series[i,:]
    ts = ts[~np.isnan(ts)]
    
    if len(ts)>fh+3:
        c += 1
    
        ts = ts[:-fh]

        ts_row = pd.DataFrame(ts).T

        if first == True:
            df_obs = ts_row
        else:
            df_obs = pd.concat([df_obs,ts_row])
        
        first = False

df_obs = df_obs.set_index(np.arange(c))

# Separate true values
obs_matr = np.zeros((len(data_series),fh))
for i in range(len(data_series)):   
    
    if i % 1000 == 0:
        print("Preparing row {}".format(i))
    
    ts = data_series[i,:]
    ts = ts[~np.isnan(ts)]
    
    if len(ts)>fh+3:
    
        ts = ts[-fh:]
        obs_matr[i,:] = ts

# Perform dynamic model selection
test_percentage = 0.25
total = df_obs.shape[0]
train_samples = int(np.ceil(total) * (1-test_percentage))
test_samples = total - train_samples

models = ['ses','holt','damped','theta','comb','rf']
steps = np.arange(0,fh+1,10)

for j in range(len(steps)-1):

    print('FH = {}'.format(steps[j+1]))

    min_smape = 20
    min_model = ''

    b = steps[j]
    e = steps[j+1]

    for m in models:

        print(m)
        
        filename = 'forecasts_'+m
        
        df_preds = pd.read_csv(os.path.join(args.forecasts_folder,filename))
        df_preds = df_preds.drop(['Unnamed: 0'],axis=1)
        df_preds = df_preds.iloc[:,b:e]
            
        df_all = pd.concat([df_obs,df_preds],axis=1,join='inner')
        
        data_all = df_all.values
        max_l = df_all.shape[1]
        
        for i in range(len(data_all)):   

            ts = data_all[i,:]
            ts = ts[~np.isnan(ts)]

            if len(ts)<max_l:
                diff = max_l - len(ts)
                padd = np.zeros((1,diff)).flatten()
                ts = np.hstack((ts,padd))

            if i == 0:
                X_mat = ts
            else:
                X_mat = np.vstack((X_mat,ts))

        min_max_scaler = preprocessing.MinMaxScaler()
        X_mat = min_max_scaler.fit_transform(X_mat)
        
        smape_arr = np.zeros(obs_matr.shape[0])
        preds = df_preds.values
        for i in range(obs_matr.shape[0]):
            obs_s = obs_matr[i,b:e]
            pred_s = preds[i,:]
            smape_arr[i] = smape(obs_s,pred_s)

        smape_col = pd.DataFrame(smape_arr)
        smape_col = np.log(smape_col)

        train_X = X_mat[:train_samples]
        train_y = smape_arr[:train_samples]
        test_X = X_mat[train_samples:]
        test_y = smape_arr[train_samples:]
        
        mc_samples = 100
        train_Xr = train_X.reshape(train_X.shape[0], 1, train_X.shape[1], 1)
        input_shape = (1,train_X.shape[1],1)

        inp = Input(shape=input_shape)
        x = Conv2D(6, (1,5), activation='relu', input_shape=input_shape, padding='same',kernel_regularizer=regularizers.l2(1.e-4))(inp)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(1,2))(x)
        x = Dropout(0.5)(x, training=True)
        x = Conv2D(16, (1,5), activation='relu', input_shape=input_shape, padding='same',kernel_regularizer=regularizers.l2(1.e-4))(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(1,2))(x)
        x = Dropout(0.5)(x, training=True)
        x = AveragePooling2D(pool_size=(1,2))(x)
        x = Flatten()(x)
        x = Dense(120, activation='relu')(x)
        x = Dropout(0.5)(x, training=True)
        x = Dense(84, activation='relu')(x)
        #x = Dropout(0.6)(x, training=True)
        x = Dense(1)(x)
        model = Model(inp, x, name='lenet-all')
        
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='mse')
        # fit model
        model.fit(train_Xr, train_y, epochs=1000, verbose=0)

        test_Xr = test_X.reshape(test_X.shape[0], 1, test_X.shape[1], 1)

        for i in range(mc_samples):
            yhat_s = model.predict(test_Xr, verbose=0)
            yhat_s = np.exp(yhat_s)
            row_yhat = pd.DataFrame(yhat_s)

            if i == 0:
                df_yhat = row_yhat
            else:
                df_yhat = pd.concat([df_yhat,row_yhat],axis=1)
                
        yhat = df_yhat.mean(axis=1)
        pred_smape_col = pd.DataFrame(yhat)
        
        if m == 'ses':
            pred_smape_df = pred_smape_col
        else:
            pred_smape_df = pd.concat([pred_smape_df,pred_smape_col],axis=1)

    pred_smape_df.columns = models
    min_model_col = pred_smape_df.idxmin(axis=1)

    if j == 0:
        min_model_df = min_model_col
    else:
        min_model_df = pd.concat([min_model_df,min_model_col],axis=1)

min_model_df.columns = steps[1:]

