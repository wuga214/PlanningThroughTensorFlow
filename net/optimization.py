import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from net.cell import TrainedCell


class ActionOptimizer(object):
    def __init__(self,
                 num_step,
                 num_act,
                 batch_size,
                 domain_settings,
                 num_state_units,
                 num_reward_units,
                 num_hidden_units,
                 num_hidden_layers,
                 dropout,
                 pretrained,
                 normalize,
                 action_mean,
                 learning_rate=0.005):
        self.action = tf.Variable(tf.truncated_normal(shape=[batch_size, num_step, num_act],
                                                      mean=action_mean, stddev=0.05), name="action")
        print(self.action)
        self.batch_size = batch_size
        self.num_step = num_step
        self.learning_rate = learning_rate
        self.sess = tf.Session()
        cell = TrainedCell(num_state_units,
                           num_reward_units,
                           num_hidden_units,
                           num_hidden_layers,
                           dropout,
                           domain_settings,
                           pretrained,
                           normalize)
        self._p_create_rnn_graph(cell)
        self._p_create_loss()
        self.sess.run(tf.global_variables_initializer())
        cell.load_weights(self.sess)
        self.sess.graph.finalize()

    def _p_create_rnn_graph(self, cell):
        self.initial_state = tf.Variable(cell.zero_state(self.batch_size, dtype=tf.float32))
        print('action batch size:{0}'.format(array_ops.shape(self.action)[0]))
        print('Initial_state shape:{0}'.format(self.initial_state))
        rnn_outputs, state = tf.nn.dynamic_rnn(cell, self.action, dtype=tf.float32, initial_state=self.initial_state)
        # need output intermediate states as well
        self.rnn_outputs = rnn_outputs
        concated = tf.concat(axis=0, values=rnn_outputs)
        print('concated shape:{0}'.format(concated.get_shape()))
        something_unpacked = tf.unstack(concated, axis=2)
        self.outputs = tf.reshape(something_unpacked[0], [-1, self.num_step, 1])
        print('self.outputs:{0}'.format(self.outputs.get_shape()))
        self.intern_states = tf.stack([something_unpacked[x+1] for x in range(len(something_unpacked)-1)], axis=2)
        self.last_state = state
        self.pred = tf.reduce_sum(self.outputs, 1)
        self.average_pred = tf.reduce_mean(self.pred)
        print("self.pred:{0}".format(self.pred))

        self.lower_action_bound = tf.placeholder(tf.float32)
        self.upper_action_bount = tf.placeholder(tf.float32)

        self.reset_action = tf.assign(self.action, tf.clip_by_value(self.action,
                                                                    self.lower_action_bound,
                                                                    self.upper_action_bount))

        self.given_inital_state = tf.placeholder(tf.float32, shape=self.initial_state.shape)
        self.reset_initial_state = self.initial_state.assign(self.given_inital_state)

    def _p_create_loss(self):

        objective = tf.reduce_mean(tf.square(self.pred))
        self.loss = objective
        print(self.loss.get_shape())
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.action])

    def Optimize(self, clip_bounds, epoch=100, show_progress=False):

        new_loss = self.sess.run([self.loss])
        print('Loss in epoch {0}: {1}'.format("Initial", new_loss))
        if show_progress:
            progress = []
        for epoch in range(epoch):
            training = self.sess.run([self.optimizer])
            if clip_bounds == None:
                action_upperbound = self.sess.run(self.intern_states)
                self.sess.run(self.reset_action, feed_dict={self.lower_action_bound: 0,
                                                            self.upper_action_bount: action_upperbound})
            else:
                self.sess.run(self.reset_action, feed_dict={self.lower_action_bound: clip_bounds[0],
                                                            self.upper_action_bount: clip_bounds[1]})
            if True:
                new_loss = self.sess.run([self.average_pred])
                print('Loss in epoch {0}: {1}'.format(epoch, new_loss))
            if show_progress and epoch % 10 == 0:
                progress.append(self.sess.run(self.intern_states))
        minimum_costs_id = np.argmax(self.sess.run(self.pred), 0)
        print(minimum_costs_id)
        best_action = np.round(self.sess.run(self.action)[minimum_costs_id[0]], 4)
        # print('Optimal Action Squence:{0}'.format(best_action))
        # pred_list = self.sess.run(self.pred)
        # pred_list = np.sort(pred_list.flatten())[::-1]
        # pred_list = pred_list[:5]
        # pred_mean = np.mean(pred_list)
        # pred_std = np.std(pred_list)
        # print('Best Cost: {0}'.format(pred_list[0]))
        # print('Sorted Costs:{0}'.format(pred_list))
        # print('MEAN: {0}, STD:{1}'.format(pred_mean, pred_std))
        # print('The last state:{0}'.format(self.sess.run(self.last_state)[minimum_costs_id[0]]))
        # print('Rewards each time step:{0}'.format(self.sess.run(self.outputs)[minimum_costs_id[0]]))
        # print('Intermediate states:{0}'.format(self.sess.run(self.intern_states)[minimum_costs_id[0]]))
        # if show_progress:
        #     progress = np.array(progress)[:, minimum_costs_id[0]]
        #     print('progress shape:{0}'.format(progress.shape))
        #     np.savetxt("progress.csv", progress.reshape((progress.shape[0], -1)), delimiter=",", fmt='%2.5f')
        return best_action

    def set_initial_state(self, initial_state):
        self.sess.run(self.reset_initial_state, feed_dict={self.given_inital_state: initial_state})