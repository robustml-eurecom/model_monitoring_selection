from numpy.random import seed
from tensorflow import set_random_seed
from math import sqrt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import array
import GPy
from scipy.cluster.vq import kmeans
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--observations', default=None)
parser.add_argument("--true_values", default=None)
parser.add_argument("--forecasts", default=None)

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

# Observations
df_series = pd.read_csv(args.observations)
df_series = df_series.drop(['V1'],axis=1)

# True forecasts
df_obs = pd.read_csv(args.true_values)
df_obs = df_obs.drop(['V1'],axis=1)

# Forecats given by comb monitored model
df_preds = pd.read_csv(args.forecasts)
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


# Run GPs monitoring model
train_y = np.reshape(train_y,(-1,1))
test_y = np.reshape(test_y,(-1,1))

k  = GPy.kern.RBF(train_X.shape[1],active_dims=np.arange(train_X.shape[1]),variance=0.001,lengthscale=np.ones(train_X.shape[1]),ARD=True)
M = 650
Z = 1.0 * kmeans(train_X, M)[0]

m = GPy.models.sparse_gp_minibatch.SparseGPMiniBatch(train_X, train_y, Z=Z, kernel=k, likelihood = GPy.likelihoods.Gaussian(),batchsize=128)
print(m)
m.Z.fix()

m.optimize(max_iters=400, messages=0)

yhat, cov = m.predict_noiseless(test_X)


