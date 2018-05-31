from keras.layers import Input, merge
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import load_model
from utils.preprocess import getnorm
import tensorflow as tf
import re
import matplotlib.pyplot as plt


class DenselyConnectedNetwork(object):

    def __init__(self, observ, hidden, output, num_layers, drop_out, boost):
        self.drop_out = drop_out
        self.boost = boost

        inputs = Input(shape=(observ,))
        with tf.variable_scope("transition"):
            x = Dense(hidden, activation='relu')(inputs)
            x = Dropout(drop_out)(x)
            interm_inputs = merge([x, inputs], mode='concat')
            if num_layers > 1:
                for i in range(num_layers - 1):
                    x = Dense(hidden, activation='relu')(interm_inputs)
                    x = Dropout(drop_out)(x)
                    interm_inputs = merge([x, interm_inputs], mode='concat')
            predictions = Dense(output, activation='linear')(interm_inputs)
            self.DeepNet = Model(input=inputs, output=predictions)
        self.DeepNet.compile(optimizer='rmsprop', loss=self.boosted_mean_squared_error)

    def boosted_mean_squared_error(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) * self.boost, axis=-1)

    def train(self, data, label, epoch, normalize=False):
        mean = []
        std = []
        if normalize:
            normalized_data, mean, std = getnorm(data)
        else:
            normalized_data = data
        self.history = self.DeepNet.fit(normalized_data, label, validation_split=0.1, batch_size=128, nb_epoch=epoch)
        return mean, std

    def test(self, datapoint, normalize=False, mean=[], std=[]):
        if normalize:
            normalized_datapoint, _, _ = getnorm(datapoint, mean, std)
        else:
            normalized_datapoint = datapoint
        return self.DeepNet.predict(normalized_datapoint, batch_size=128, verbose=0)

    def loadmodel(self, modelpath):
        self.DeepNet = load_model(modelpath)

    def save(self, modelpath):
        sess = K.get_session()
        variables = tf.trainable_variables()
        var_dict = {}
        #import ipdb; ipdb.set_trace()
        for v in variables:
            if "transition" in v.name:
                var_dict["name"] = v
        for k in var_dict.keys():
            print(k)
        saver = tf.train.Saver(var_dict)
        saver.save(sess, modelpath)

        # Keras save function
        # self.DeepNet.save(modelpath)

    def getmodel(self):
        return self.DeepNet