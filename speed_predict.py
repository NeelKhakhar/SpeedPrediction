import tensorflow as tf
import numpy
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy import array
from numpy import hstack
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Dense
# from keras.layers.merge import concatenate
# from keras import backend as K

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
    
    epochs = 500
    early_stopping_patience = 20 

    data_logger = Datalogger()
    window_size = data_logger.config[0]
    horizon = data_logger.config[1]

    x_train, y_train = data_logger.train_split_sequences()
    x_test, y_test = data_logger.test_split_sequences()
    x_val, y_val = data_logger.val_split_sequences()

    len_train = len(x_train)
    len_test = len(x_test)
    len_val = len(x_val)

    # backbones choice: "MLP", "MIMLP", "MILSTM"
    backbone = "MIMLP"

    assert backbone in ["MLP", "MIMLP", "MILSTM"]
    
    if backbone == "MLP":
        x_train, y_train = model_input(x_train, y_train)
        x_test, y_test = model_input(x_test, y_test)
        x_val, y_val = model_input(x_val, y_val)
        model = MLP_Attr01(window_size, nodes=10*window_size)
        out_dir = "result/MLP_Attr01"
    elif backbone == "MIMLP":
        x_train, y_train = model_input(x_train, y_train)
        x_test, y_test = model_input(x_test, y_test)
        x_val, y_val = model_input(x_val, y_val)
        model = MIMLP_Attr01(window_size, nodes=10*window_size)
        out_dir = "result/MTMLP_Attr01"
    elif backbone == "MILSTM":
        n_features = x_train.shape[2]
        model = MILSTM(window_size, n_features)
        out_dir = "result/MILSTM_Attr01"

    print(model.summary())
    # print(x_train)
    # breakpoint()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[callback]
    )
    # print(history.history)
    
    print(
        "Trained on {} data samples, validated on {} samples, .\
         Evaluate on test data {} samples".format(len_train, len_val, len_test)
    )
    y_actual = numpy.array(y_test)
    # y_actual = denormalize(y_actual)
    y_predicted = numpy.array(model.predict(x_test, verbose=0))
    # y_predicted = denormalize(y_predicted)
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    print(rmse)
    print(model.summary())

    result_dir = out_dir + "/M_3/window={}/horizon={}_rmse={}/".format(window_size, horizon, rmse)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    numpy.savetxt(result_dir+"actual.csv", y_actual, delimiter=",")
    numpy.savetxt(result_dir+"prediction.csv", y_predicted, delimiter=",")

    with open(result_dir + 'model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    np.save(result_dir + 'training_history.npy',history.history)

    # # loading method
    # # history = np.load(result_dir + 'training_history.npy',allow_pickle='TRUE').item()
        
    # plot the prediction vs ground truth on test data
    plt.plot(y_actual, label='actual speed')
    plt.plot(y_predicted, label='predicted speed')
    plt.title("window={}_horizon={}_rmse={}".format(window_size, horizon, rmse))
    plt.savefig(result_dir+'plot.png')
    plt.show()

    # plot the summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(result_dir+'loss.png')
    plt.show()

