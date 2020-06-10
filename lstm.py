from numpy.random import seed
from tensorflow import set_random_seed
from math import sqrt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import array
from __future__ import print_function
import tensorflow as tf
import random
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

class SequenceData(object):
    def __init__(self, n_samples=1000, test=False):
        self.data = []
        self.labels = []
        self.seqlen = []
        max_seq_len = max_l
        
        for i in range(n_samples):
            
            if test == False:
                o = series_matr[i, :]
                p = preds[i,:]
                smape = smape_arr[i]
            else:
                o = series_matr[train_samples+i, :]
                p = preds[train_samples+i,:]
                smape = smape_arr[train_samples+i]
            ts = np.hstack((o,p))
            ts = ts[~np.isnan(ts)]            
            
            s = [[ts[j]] for j in range(len(ts))]
            len_s = len(s)

            self.seqlen.append(len_s)
            
            if len_s<max_seq_len:
                #Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len_s)]
            
            s_arr = np.asarray(s).flatten()
            s_arr = np.interp(s_arr, (s_arr.min(), s_arr.max()), (0, 1))
            s = [[s_arr[j]] for j in range(len(s_arr))]
            
            self.data.append(s)
            self.labels.append([smape])

        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

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

smape_arr = np.log(smape_arr)

test_percentage = 0.25
total = X_mat.shape[0]
train_samples = int(np.ceil(total) * (1-test_percentage))
test_samples = total - train_samples

# Run LSTM monitoring model
tf.reset_default_graph()

learning_rate = 0.01
training_steps = 10000
batch_size = 32
display_step = 200

seq_max_len = max_l
n_hidden = 32
dim_output = 1

trainset = SequenceData(n_samples=train_samples,test=False)
testset = SequenceData(n_samples=test_samples,test=True)

x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, dim_output])
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, dim_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([dim_output]))
}


def dynamicRNN(x, seqlen, weights, biases):

    x = tf.unstack(x, seq_max_len, 1)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
 
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

with tf.name_scope('mse'):
    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope('prediction_error'):
    pred_err = tf.reduce_mean(tf.metrics.mean_absolute_error(labels=tf.math.exp(y), predictions=tf.math.exp(pred)))
resid = tf.abs(tf.subtract(tf.math.exp(y),tf.math.exp(pred)))

init_l = tf.local_variables_initializer()
init = tf.global_variables_initializer()

# Start training
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # Run the initializer
    sess.run(init)
    sess.run(init_l)
    
    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0 or step == 1:

            pred_error, loss = sess.run([pred_err,cost], feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", MAE= " + \
                  "{:.5f}".format(pred_error))

    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    
    predictions = sess.run(pred, feed_dict={x: test_data, y: test_label,
                                  seqlen: test_seqlen})
