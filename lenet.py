from numpy.random import seed
from tensorflow import set_random_seed
from math import sqrt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import array
from keras import Input, Model
from keras.models import InputLayer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

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

# Observations
df_series = pd.read_csv('./M4-train/Yearly-train.csv')
df_series = df_series.drop(['V1'],axis=1)

# True forecasts
df_obs = pd.read_csv('./M4-test/Yearly-test.csv')
df_obs = df_obs.drop(['V1'],axis=1)

# Forecats given by comb monitored model
df_preds = pd.read_csv('comb-preds-yearly.csv')
df_preds = df_preds.drop(['Unnamed: 0'],axis=1).T
df_preds.index = df_obs.index

series_matr = df_series.values
obs_matr = df_obs.values
preds = df_preds.values 

# Evaluate sMAPE between true values and forecasts
smape_arr = np.zeros(obs_matr.shape[0])
for i in range(len(obs_matr)):
    smape_arr[i] = smape(obs_matr[i],preds[i])

df_all = pd.concat([df_series,df_preds],axis=1,join='inner')
max_l = df_all.shape[1]
data_all = df_all.values

# Padding time-series
for i in range(len(data_all)):   

    if i % 1000 == 0:
        print("Padding row {}".format(i))

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

smape_arr = np.log(smape_arr)

test_percentage = 0.25
total = X_mat.shape[0]
train_samples = int(np.ceil(total) * (1-test_percentage))
test_samples = total - train_samples
train_X = X_mat[:train_samples]
train_y = smape_arr[:train_samples]
test_X = X_mat[train_samples:]
test_y = smape_arr[train_samples:]


# Run LeNet monitoring model
# reshape from [samples, timesteps] into [samples, timesteps, features]
train_Xr = train_X.reshape(train_X.shape[0], 1, train_X.shape[1], 1)
print(train_Xr.shape)
input_shape = (1,train_X.shape[1],1)
# define model
model = Sequential()
model.add(Conv2D(6, (1,5), activation='relu', input_shape=input_shape, padding='same',kernel_regularizer=regularizers.l2(1.e-4)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(1,2)))
model.add(Dropout(0.5))
model.add(Conv2D(16, (1,5), activation='relu', input_shape=input_shape, padding='same',kernel_regularizer=regularizers.l2(1.e-4)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(1,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='mse')
# fit model
model.fit(train_Xr, train_y, epochs=1000, verbose=2, batch_size=32)

test_Xr = test_X.reshape(test_X.shape[0], 1, test_X.shape[1], 1)
yhat = model.predict(test_Xr, verbose=0)

