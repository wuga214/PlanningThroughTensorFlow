import tensorflow as tf
from keras.layers import Dropout, Dense, merge
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops, array_ops, init_ops, nn_ops
from utils.preprocess import getnorm


class TrainedCell(rnn_cell_impl.LayerRNNCell):
    # Reward function is assumed to be hard defined

    def __init__(self,
                 num_state_units,
                 num_reward_units,
                 num_hidden_units,
                 num_hidden_layers,
                 dropout,
                 domain,
                 transition_weights,
                 transition_normalize,
                 reuse=None, name=None):
        super(TrainedCell, self).__init__(_reuse=reuse, name=name)
        self._num_state_units = num_state_units
        self._num_reward_units = num_reward_units
        self._num_hidden_units = num_hidden_units
        self._num_hidden_layers = num_hidden_layers
        self._dropout = dropout
        self._domain = domain
        self._transition_weights = transition_weights
        self._transition_normalize = transition_normalize
        self._weights = dict()

    @property
    def state_size(self):
        return self._num_state_units

    @property
    def output_size(self):
        return self._num_reward_units

    def build(self, inputs_shape):

        for layer_name in self._transition_weights.keys():
            self._weights[layer_name] = dict()
            for weight_name in self._transition_weights.get(layer_name).keys():
                weight = self._transition_weights[layer_name][weight_name]
                self._weights[layer_name][weight_name] = self.add_variable('{0}_{1}'.format(layer_name,
                                                                                            weight_name),
                                                                           shape=weight.shape,
                                                                           trainable=False)

    def load_weights(self, sess):
        for layer_name in self._transition_weights.keys():
            for weight_name in self._transition_weights.get(layer_name).keys():
                weight = self._transition_weights[layer_name][weight_name]
                assign_op = self._weights[layer_name][weight_name].assign(weight)
                sess.run(assign_op)

    def call(self, inputs, state):
        print(inputs.get_shape())
        print(state.get_shape())


        with tf.variable_scope("Transition"):
            init = array_ops.concat([inputs, state], 1)

            normalized_init = (init - self._transition_normalize[0]) / self._transition_normalize[1]

            x = tf.nn.relu(nn_ops.bias_add(math_ops.matmul(normalized_init,
                                                           self._weights['1']['kernel']),
                                           self._weights['1']['bias']))
            interm_inputs = array_ops.concat([x, normalized_init], 1)
            if self._num_hidden_layers > 1:
                for i in range(self._num_hidden_layers - 1):
                    x = tf.nn.relu(nn_ops.bias_add(math_ops.matmul(interm_inputs,
                                                                   self._weights[str(i+2)]['kernel']),
                                                   self._weights[str(i+2)]['bias']))
                    interm_inputs = array_ops.concat([x, interm_inputs], 1)

            next_state = nn_ops.bias_add(math_ops.matmul(interm_inputs,
                                                         self._weights[str(self._num_hidden_layers+1)]['kernel']),
                                         self._weights[str(self._num_hidden_layers+1)]['bias'])

        with tf.variable_scope("Reward"):
            reward = self._domain.Reward(next_state, inputs)

        return tf.concat(axis=1, values=[reward, next_state]), next_state
