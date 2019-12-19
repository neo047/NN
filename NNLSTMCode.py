#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:14:50 2019

@author: Akhil
"""

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame
from pandas import concat
import numpy as np
from numpy import concatenate
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.optimizers import Adam
#for saving to csv
from os import listdir
import csv
import pandas
from matplotlib import pyplot
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error
import time

def NN(Neurons = 100, Neuron2 = 125, Neuron3 = 125, Neuron4 = 100, epochs = 50,batchsize = 512,
       breg = 0.01, dropout = 0.1, layer = 2, lr = 0.001, n_lookahead = 24,recurrent = 'hard_sigmoid'):
    global Pickled_save, train_X,train_y,test_y,test_X
    n_lookahead = 24
    fea = '40_Features'
    RS = False
    f_bias = 0
    bs = 1
    print('Started')
    print(Neurons,Neuron2,Neuron3,Neuron4,batchsize,breg,dropout,layer,n_lookahead)
    #
    model=Sequential()
    if layer == 1:
        model.add(LSTM(Neurons,recurrent_activation = 'hard_sigmoid', input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
               dropout = dropout, return_sequences=False, unit_forget_bias = 1))
    elif layer == 2:
        model.add(LSTM(Neurons,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
               dropout = dropout, return_sequences=True,unit_forget_bias = 1))
        model.add(LSTM(Neuron2))
    elif layer == 3:
        model.add(LSTM(Neurons,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
                return_sequences=True,unit_forget_bias = 1))
        model.add(LSTM(Neuron2, return_sequences = True,dropout = dropout))
        model.add(LSTM(Neuron3,dropout = dropout))
    elif layer == 4:
        model.add(LSTM(Neurons,input_shape=(train_X.shape[1], train_X.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),
                return_sequences=True,recurrent_activation = recurrent, return_state = RS,unit_forget_bias = f_bias, use_bias = bs))
        model.add(LSTM(Neuron2, dropout = dropout, return_sequences = True,recurrent_activation = recurrent,return_state = RS,unit_forget_bias = f_bias, use_bias = bs))
        model.add(LSTM(Neuron3, dropout = dropout, return_sequences = True,recurrent_activation = recurrent,return_state = RS,unit_forget_bias = f_bias, use_bias = bs))
        model.add(LSTM(Neuron4,recurrent_activation = recurrent, use_bias = bs))
    model.add(Dense(n_lookahead))
    adam = Adam(lr = lr,decay = 0.001)
    model.compile(loss='mean_squared_error', optimizer = adam)
    start = time.time()
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batchsize,
            validation_split = 0.1, verbose=1, shuffle=True)  
    end = time.time()
    print('The time elapsed for 10 epochs in seconds is', end-start)
    yhat = model.predict(test_X)
    #
    model.get_config()
    k = model.get_weights()
    model.summary() 
    GG = "Model.pickle"
    pickle_out = open(GG,"wb")
    pickle.dump(k,pickle_out)
    #
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #
    return yhat, loss, val_loss


train_X = np.load('trainX.npy')
train_y = np.load('trainY.npy')
test_X = np.load('testX.npy')
test_y = np.load('testy.npy')


NNTraining0 = NN(Neurons = 100, 
                Neuron2 = 90, 
                Neuron3 = 80,
                Neuron4 = 70,
                epochs = 1,
                batchsize = 72, 
                breg = 0.04, 
                dropout = 0.1, 
                layer = 4, 
                lr = 0.001,
                recurrent = 'hard_sigmoid')

Forecast = NNTraining0[0]
#