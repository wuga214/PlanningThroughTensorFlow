from keras.layers import Input, concatenate
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import load_model
from utils.preprocess import getnorm
import tensorflow as tf
import re
from utils.io import save_pickle
import matplotlib.pyplot as plt


class DenselyConnectedNetwork(object):

    def __init__(self, observ, hidden, output, num_layers, drop_out, boost):
        self.drop_out = drop_out
        self.boost = boost

        inputs = Input(shape=(observ,))
        with tf.variable_scope("transition"):
            x = Dense(hidden, activation='relu')(inputs)
            x = Dropout(drop_out)(x)
            interm_inputs = concatenate([x, inputs])
            if num_layers > 1:
                for i in range(num_layers - 1):
                    x = Dense(hidden, activation='relu')(interm_inputs)
                    x = Dropout(drop_out)(x)
                    interm_inputs = concatenate([x, interm_inputs])
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

    def save(self, weights_path, weights_name):
        sess = K.get_session()
        variables = tf.trainable_variables()
        var_dict = dict()
        for v in variables:
            if "transition" in v.name:
                name = re.sub('transition/', '', v.name)
                name = re.sub(':0', '', name)
                layer_name, var_name = name.split('/')
                layer_name = re.sub('dense_', '', layer_name)
                if not var_dict.get(layer_name):
                    var_dict[layer_name] = dict()
                var_dict[layer_name][var_name] = v
        for k in var_dict.keys():
            print(k)
            for j in var_dict[k].keys():
                print('---{0}'.format(j))

        weights = sess.run(var_dict)
        save_pickle(weights, weights_path, weights_name)

        # Keras save function
        # self.DeepNet.save(modelpath)

    def getmodel(self):
        return self.DeepNet