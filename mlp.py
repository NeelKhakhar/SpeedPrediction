"""
zixuan zxzhao@umich.edu
This is the multiple input MLP model 
"""

from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import concatenate
from keras import backend as K

from keras.optimizers import Adam

def MIMLP_Attr01(n_steps, nodes = 1000):
    """
    predict vehicle speed from attributes set 1 with multi-layer perceptron model
    :param nodes: int, data use to train the model
    :return: model, class 'tensorflow.python.keras.engine.functional.Functional'
    """
    # first input attribute and model, Accelerator pedal operation
    visible1 = Input(shape=(n_steps,))
    dense1 = Dense(nodes, activation='relu')(visible1)

    # second input attribute and model, Steering angle
    visible2 = Input(shape=(n_steps,))
    dense2 = Dense(nodes, activation='relu')(visible2)

    # third input attribute and model, Longitudinal acceleration
    visible3 = Input(shape=(n_steps,))
    dense3 = Dense(nodes, activation='relu')(visible3)

    # fourth input attribute and model, Transversal acceleration
    visible4 = Input(shape=(n_steps,))
    dense4 = Dense(nodes, activation='relu')(visible4)

    # fifth input attribute and model, vehicle speed
    visible5 = Input(shape=(n_steps,))
    dense5 = Dense(nodes, activation='relu')(visible5)

    # merge input models, concatenate and flatten the input models to one dimentional
    merge = concatenate([dense1, dense2, dense3, dense4, dense5])

    dropout = Dropout(0.2)(merge)
    
    # output regresion layer
    output = Dense(1)(dropout)

    # output = Dense(1)(merge)

    # construct and compile the model
    model = Model(inputs=[visible1, visible2, visible3, visible4, visible5], outputs=output)

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
    opt = Adam(learning_rate=0.001)
    model.compile(
        optimizer='adam', 
        loss = 'mse'
        )

    return model


def MLP_Attr01(n_steps, nodes = 1000):
    """
    predict vehicle speed from attributes set 1 with multi-layer perceptron model
    :param nodes: int, data use to train the model
    :return: model, class 'tensorflow.python.keras.engine.functional.Functional'
    """
    # first input attribute and model, Accelerator pedal operation
    visible1 = Input(shape=(n_steps,))
    # dense1 = Dense(nodes, activation='relu')(visible1)

    # second input attribute and model, Steering angle
    visible2 = Input(shape=(n_steps,))
    # dense2 = Dense(nodes, activation='relu')(visible2)

    # third input attribute and model, Longitudinal acceleration
    visible3 = Input(shape=(n_steps,))
    # dense3 = Dense(nodes, activation='relu')(visible3)

    # fourth input attribute and model, Transversal acceleration
    visible4 = Input(shape=(n_steps,))
    # dense4 = Dense(nodes, activation='relu')(visible4)

    # fifth input attribute and model, vehicle speed
    visible5 = Input(shape=(n_steps,))
    # dense5 = Dense(nodes, activation='relu')(visible5)

    # merge input models, concatenate and flatten the input models to one dimentional
    merge = concatenate([visible1, visible2, visible3, visible4, visible5])

    dense1 = Dense(nodes, activation="relu")(merge)
    
    # output regresion layer
    output = Dense(1)(dense1)

    # construct and compile the model
    model = Model(inputs=[visible1, visible2, visible3, visible4, visible5], outputs=output)

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
        
    opt = Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt, 
        loss = 'mse'
        )

    return model