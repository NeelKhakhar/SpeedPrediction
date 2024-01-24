import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.initializers import RandomUniform


class Attention_Customize(tf.keras.layers.Layer):
    def __init__(self,return_sequences=True,**kwargs):
        super(Attention_Customize, self).__init__(**kwargs)
        self.return_sequences=True

    def get_config(self):
        config = {
            "return_sequences":self.return_sequences
        }
        base_config =super(Attention_Customize,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention_Customize, self).build(input_shape)

    def call(self, x, **kwargs):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences == True:
            return output
        else:
            return K.sum(output,axis=1)

'''
class Attention_Customize(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True, dropout_rate=0.1, **kwargs):
        super(Attention_Customize, self).__init__(**kwargs)
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {"return_sequences": self.return_sequences, "dropout_rate": self.dropout_rate}
        base_config = super(Attention_Customize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform"
        )
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention_Customize, self).build(input_shape)

    def call(self, x, **kwargs):
        e = K.relu(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        if self.dropout_rate > 0:
            a = K.dropout(a, level=self.dropout_rate)
        output = K.batch_dot(a, x, axes=1)
        if self.return_sequences:
            return output
        else:
            return K.sum(output, axis=1)
'''


def ALSTM(n_steps, n_features):
    """
    predict vehicle speed from attributes set 1 with multi-layer perceptron model
    :param nodes: int, data use to train the model
    :return: model, class 'tensorflow.python.keras.engine.functional.Functional'
    """
    '''
    regularizer = l1_l2(0.01)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(n_steps, n_features)))
    model.add(LSTM(5, activation='relu', return_sequences=True, recurrent_dropout=0.2,
             recurrent_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001)))
    model.add(Attention_Customize(return_sequences=True))
    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    '''
    #m1,m2 cmt out the two single dr
    regularizer = l1_l2(0.01)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(n_steps, n_features)))
    #model.add(Dropout(0.2))
    model.add(LSTM(32, activation='tanh', return_sequences=True, dropout=0.2,
             recurrent_regularizer=regularizer, kernel_initializer=RandomUniform(minval=-.001, maxval=0.001)))
    model.add(Attention_Customize(return_sequences=True))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    return model



if __name__ == "__main__":
    model = ALSTM(n_steps=300,n_features=7)
    print(model.summary())

