
from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input, LSTM, Reshape, Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import concatenate
from keras import backend as K
from keras.regularizers import l2
from keras.initializers import RandomUniform

from keras.optimizers import Adam

def MIMLPLSTM(n_steps, n_features):
    """
    predict vehicle speed from attributes set 1 with multi-layer perceptron model
    :param nodes: int, data use to train the model
    :return: model, class 'tensorflow.python.keras.engine.functional.Functional'
    """
    nodes = n_steps*10
    regularizer = l2(0.01)

    # first input attribute and model, Accelerator pedal operation
    visible1 = Input(shape=(n_steps, 1,))
    dense1 = LSTM(1, return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001))(visible1)

    # second input attribute and model, Steering angle
    visible2 = Input(shape=(n_steps, 1,))
    dense2 = LSTM(1, return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001))(visible2)

    # third input attribute and model, Longitudinal acceleration
    visible3 = Input(shape=(n_steps, 1,))
    dense3 = LSTM(1, return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001))(visible3)

    # fourth input attribute and model, Transversal acceleration
    visible4 = Input(shape=(n_steps, 1,))
    dense4 = LSTM(1, return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001))(visible4)

    # fifth input attribute and model, vehicle speed
    visible5 = Input(shape=(n_steps, 1,))
    dense5 = LSTM(1, return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001))(visible5)

    visible6 = Input(shape=(n_steps, 1,))
    dense6 = LSTM(1, return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001))(visible6)

    visible7 = Input(shape=(n_steps, 1,))
    dense7 = LSTM(1, return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001))(visible7)

    # merge input models, concatenate and flatten the input models to one dimentional
    merge = concatenate([dense1, dense2, dense3, dense4, dense5, dense6, dense7])

    # dropout = Dropout(0.2)(merge)

    # R = Reshape(target_shape=(nodes, n_features))(dropout)
    L = LSTM(50, activation='relu', return_sequences= True, activity_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001))(merge)
    F = Flatten()(L)
    D = Dense(128, activation='relu')(F)
    # output regresion layer
    output = Dense(1)(D)
    # output = LSTM(1, return_sequences=False)(D)
    # output = Dense(1)(merge)

    # construct and compile the model
    model = Model(inputs=[visible1, visible2, visible3, visible4, visible5, visible6, visible7], outputs=output)

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    opt = Adam(learning_rate= 0.0001, clipnorm = 50)
    # opt = Adam(learning_rate=0.001)
    model.compile(
        optimizer= opt, 
        loss = 'mse'
        )

    return model
