from keras.layers import Input, merge
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import load_model
from utils.preprocess import getnorm
import matplotlib.pyplot as plt


class DenselyConnectedNetwork(object):

    def __init__(self, observ, hidden, output, num_layers, drop_out, boost):
        self.drop_out = drop_out
        self.boost = boost

        inputs = Input(shape=(observ,))
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
        self.DeepNet.save(modelpath)

    def getmodel(self):
        return self.DeepNet

    def showhistory(self):
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('train_curve.png')