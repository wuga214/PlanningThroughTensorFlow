import os
import pandas as pd
import numpy as np
from keras.layers import Dense
import pickle


def path_finder(path):
    script_dir = os.path.dirname('__file__')
    fullpath = os.path.join(script_dir, path)
    return fullpath


def read_data(path):
    fullpath=path_finder(path)
    return pd.read_csv(fullpath, sep=',', header=0)


def dump_norm_info(pd_data, mean_DNN, std_DNN, domain, path):
    fullpath = '{0}Network_Normalization_{1}.txt'.format(path, domain)
    filehandler = open(fullpath,'w')
    Headers=list(pd_data.columns.values)
    for i in range(len(Headers)):
        filehandler.write('N0'+str(i)+','+Headers[i].replace(': ', ',')+','+str(mean_DNN[i])+','+str(std_DNN[i])+'\n')
    filehandler.close()
    print 'Normalization File Complete!'


def dump_net_iohead(pd_data, pd_label, num_input, num_output, hidden_prefix, net_depth, domain, type, path):
    #Type in {Regular,Delta}
    headers = list(pd_data.columns.values)+list(pd_label.columns.values)
    fullpath = '{0}Headers_RDDL_{1}.txt'.format(path, domain)
    filehandler = open(fullpath, 'w')
    for i in range(num_input):
        filehandler.write('N0'+str(i)+','+headers[i].replace(': ', ',')+'\n')
    for i in range(num_output):
        filehandler.write(hidden_prefix+str(net_depth)+str(i)+','+type+','+headers[i+num_input].replace(':', ',')+'\n')
    filehandler.close()
    print 'Headers File Complete!'


class NetTopology(object):
    def __init__(self, layers, mean, std, train_type='regular'):
        self.layers = layers
        self.input_dim = layers[0].get_config().get('batch_input_shape')[1]
        self.num_upper_layers = len(layers)
        self.mean = mean
        self.std = std
        self.train_type = train_type
        self.nodenames = []
        layernodename = []
        for i in range(0, self.input_dim):
            layernodename.append('N0' + str(i))
        layernodename.append('B0')
        self.nodenames.append(layernodename)

    def layerwise_transform(self, layer, layer_id, hiddenstart='N', writefile=False, filehandler=None, lastlayer=False):
        input_dim, output_dim = layer.get_weights()[0].shape
        if (lastlayer == True):
            activation = self.train_type
        else:
            activation = layer.get_config().get('activation')
        layernodename = []
        weights_bias = layer.get_weights()
        weights = weights_bias[0]
        bias = weights_bias[1]
        for i in range(0, output_dim):
            layernodename.append(hiddenstart + str(layer_id) + str(i))
        for i in range(0, output_dim):
            row = [layernodename[i], activation]
            if (layer_id == 1):
                adjustedweights = []
                for j in range(0, input_dim):
                    row.append(self.nodenames[-1][j])
                    if self.std[j] != 0:
                        adjustedweights.append(weights[j][i] / self.std[j])
                        row.append(weights[j][i] / self.std[j])
                    else:
                        adjustedweights.append(0)
                        row.append(0)
                row.append(self.nodenames[-1][-1])
                row.append(bias[i] - np.dot(np.array(adjustedweights), np.array(self.mean)))
            else:
                if input_dim == (len(self.nodenames[-1]) - 1):
                    for j in range(0, input_dim):
                        row.append(self.nodenames[-1][j])
                        row.append(weights[j][i])
                    row.append(self.nodenames[-1][-1])
                    row.append(bias[i])
                else:
                    index_shift = 0
                    for k in range(len(self.nodenames) - 1, 0, -1):
                        for j in range(0, len(self.nodenames[k]) - 1):
                            row.append(self.nodenames[k][j])
                            row.append(weights[j + index_shift][i])
                        index_shift = index_shift + len(self.nodenames[k]) - 1
                    adjustedweights = []
                    for j in range(0, len(self.nodenames[0]) - 1):
                        row.append(self.nodenames[0][j])
                        if self.std[j] != 0:
                            adjustedweights.append(weights[j + index_shift][i] / self.std[j])
                            row.append(weights[j + index_shift][i] / self.std[j])
                        else:
                            adjustedweights.append(0)
                            row.append(0)
                    row.append(self.nodenames[-1][-1])
                    row.append(bias[i] - np.dot(np.array(adjustedweights), np.array(self.mean)))

            if writefile:
                filehandler.write(','.join(map(str, row)) + '\n')
            else:
                print ','.join(map(str, row))
        layernodename.append('B' + str(layer_id))
        self.nodenames.append(layernodename)

    def net_transform(self, hiddenstart='N', writefile=False, path=None, domain=None, overwrite=False):
        filehandler = None
        if writefile:
            fullpath = "{0}Network_RDDL_{1}.txt".format(path, domain)
            if overwrite is True:
                filehandler = open(fullpath, 'w')
            else:
                filehandler = open(fullpath, 'a')
        counter = 0
        for i in range(0, self.num_upper_layers):
            if type(self.layers[i]) is Dense:
                if (i == self.num_upper_layers - 1):
                    self.layerwise_transform(self.layers[i], counter + 1, hiddenstart, writefile, filehandler, True)
                else:
                    self.layerwise_transform(self.layers[i], counter + 1, hiddenstart, writefile, filehandler, False)
                counter = counter + 1
        print 'Network Dumping Done!'


def save_pickle(dictionary, path, name):
    with open('{0}/{1}.pickle'.format(path, name), 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path, name):
    with open('{0}/{1}.pickle'.format(path, name), 'rb') as handle:
        return pickle.load(handle)