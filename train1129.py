import tensorflow as tf
import numpy
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from numpy import array
from numpy import hstack

from preprocessing import Preprocessing
from data_logger import Datalogger
from model.multivariate.mlp import MIMLP_Attr01, MLP_Attr01
from model.multivariate.lstm import MILSTM
from utils.visualization import plot_data

from sklearn.metrics import mean_squared_error

from contextlib import redirect_stdout


get_model_from_name = {
    "MLP"       : MIMLP_Attr01,
    "MIMLP"     : MLP_Attr01,
    "MILSTM"    : MILSTM,
}


def model_input(X, y):
    X1 = X[:, :, 0]
    X2 = X[:, :, 1]
    X3 = X[:, :, 2]
    X4 = X[:, :, 3]
    X5 = X[:, :, 4]
    X6 = X[:, :, 5]
    X7 = X[:, :, 6]
    #X8 = X[:, :, 7]
    #X9 = X[:, :, 8]
    return [X1, X2, X3, X4, X5], y


class PrintLogs(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs

    def set_params(self, params):
        params['epochs'] = 0

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d' % (epoch + 1, self.epochs), end='')


def denormalize(sequence):
    data = array(sequence)
    scaler = StandardScaler()
    if len(data.shape) == 1 or data.shape[1] != 1:
        data = data.reshape(-1, 1)
    # max, min, mean, variance are all store in scaler, so we need it
    # to perform inverse transform
    return scaler.inverse_transform(data).reshape(-1)


if __name__ == "__main__":

    epochs = 2

    early_stopping_patience = 5

    data_logger = Datalogger()
    window_size = data_logger.config[0]
    horizon = data_logger.config[1]

    x_train, y_train = data_logger.train_split_sequences()
    # x_test, y_test = data_logger.test_split_sequences()
    x_val, y_val = data_logger.val_split_sequences()

    len_train = len(x_train)
    # len_test = len(x_test)
    len_val = len(x_val)
	
# A1 normalization
    # x_train[:,::,4],x_val[:,::,4] = x_train[:,::,4]/100, x_val[:,::,4]/100
    # x_train[:,::,1],x_val[:,::,1] = x_train[:,::,1]/600, x_val[:,::,1]/600
    # x_train[:,::,0],x_val[:,::,0] = x_train[:,::,0]/100, x_val[:,::,0]/100
# A2 normalization
    x_train[:,::,4],x_val[:,::,4] = x_train[:,::,4]/100, x_val[:,::,4]/100
    x_train[:,::,1],x_val[:,::,1] = x_train[:,::,1]/600, x_val[:,::,1]/600
    x_train[:,::,0],x_val[:,::,0] = x_train[:,::,0]/100, x_val[:,::,0]/100
    x_train[:,::,5],x_val[:,::,5] = x_train[:,::,5]/100, x_val[:,::,5]/100
    x_train[:,::,6],x_val[:,::,6] = x_train[:,::,6]/100, x_val[:,::,6]/100
# A3 normalization
    # x_train[:,::,4],x_val[:,::,4] = x_train[:,::,4]/100, x_val[:,::,4]/100
    # x_train[:,::,1],x_val[:,::,1] = x_train[:,::,1]/600, x_val[:,::,1]/600
    # x_train[:,::,0],x_val[:,::,0] = x_train[:,::,0]/100, x_val[:,::,0]/100
    # x_train[:,::,5],x_val[:,::,5] = x_train[:,::,5]/100, x_val[:,::,5]/100
    # x_train[:,::,6],x_val[:,::,6] = (x_train[:,::,6]+1000)/(7000+1000), (x_val[:,::,6]+1000)/(7000+1000)
    # x_train[:,::,7],x_val[:,::,7] = (x_train[:,::,7]+700)/(7300+700), (x_val[:,::,7]+700)/(7300+700)
    # x_train[:,::,8],x_val[:,::,8] = x_train[:,::,8]/100, x_val[:,::,8]/100





    y_train,y_val = y_train/100, y_val/100
    # backbones choice: "MLP", "MIMLP", "MILSTM"
    backbone = "MLP"

    assert backbone in ["MLP", "MIMLP", "MILSTM"]

    if backbone == "MLP":
        x_train, y_train = model_input(x_train, y_train)
        # x_test, y_test = model_input(x_test, y_test)
        x_val, y_val = model_input(x_val, y_val)
        model = MLP_Attr01(window_size, nodes=10*window_size)
        out_dir = "test/"
    elif backbone == "MIMLP":
        x_train, y_train = model_input(x_train, y_train)
        # x_test, y_test = model_input(x_test, y_test)
        x_val, y_val = model_input(x_val, y_val)
        model = MIMLP_Attr01(window_size, nodes=10*window_size)
        out_dir = "MC2_A_20NODES/"
    elif backbone == "MILSTM":
        n_features = x_train.shape[2]
        model = MILSTM(window_size, n_features)
        out_dir = "LSTM_A_A2/"

    print(model.summary())
    print(
        "Trained on {} data samples, validated on {} samples.".format(len_train, len_val)
    )
    # breakpoint()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[callback]
    )

    print(
        "Trained on {} data samples, validated on {} samples.".format(len_train, len_val)
    )
    print(model.summary())


    model.save(out_dir + "/A_window={}_horizon={}.h5".format(window_size, horizon))

