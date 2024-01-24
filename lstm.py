"""
zixuan zxzhao@umich.edu
This is the multiple input MLP model 
"""

from numpy import array
from numpy import hstack
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Reshape, Flatten, Conv1D, Input, Add, Lambda, Concatenate, Bidirectional, BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform
# from .alstm import Attention_Customize

def MILSTM(n_steps, n_features, n_outputs):
    """
    predict vehicle speed from attributes set 1 with multi-layer perceptron model
    :param nodes: int, data use to train the model
    :return: model, class 'tensorflow.python.keras.engine.functional.Functional'
    """

    regularizer = l2(0.01)
    
    model = Sequential()
    
    # model.add(LSTM(5, activation='relu', input_shape=(n_steps, n_features), return_sequences= True))
    model.add(LSTM(50, activation='relu', return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.0001, maxval=0.0001), input_shape=(n_steps, n_features)))
    # model.add(LSTM(1, return_sequences= False, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001)))
    # model.add(TimeDistributed(Dense(1, activity_regularizer=regularizer)))
    # model.add(Reshape(target_shape=(n_outputs,1)))
    # model.add(Conv1D(filters= 4, kernel_size=8, kernel_regularizer=regularizer, padding= 'same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizer))
    # model.add(LSTM(1, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001)))
    model.add(Dense(1))

    opt = Adam(clipnorm = 2, learning_rate= 0.001)
    model.compile(optimizer=opt, loss='mse')

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


    return model