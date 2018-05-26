import tensorflow as tf
import numpy as np


class RESERVOIR(object):
    def __init__(self,
                 batch_size,
                 default_settings):
        self.batch_size = batch_size
        self.reservoirs = default_settings['reservoirs']
        self.reservoir_num = len(default_settings['reservoirs'])
        self.biggestmaxcap = tf.constant(default_settings["biggestmaxcap"], dtype=tf.float32)
        self.zero = tf.constant(0, shape=[self.batch_size, self.reservoir_num], dtype=tf.float32)
        self._high_bounds(default_settings["high_bound"])
        self._low_bounds(default_settings["low_bound"])
        self._rains(default_settings["rain"])
        self._max_cap(default_settings["max_cap"])
        self._downstream(default_settings["downstream"])
        self._downtosea(default_settings["downtosea"])

    def _max_cap(self, max_cap_list):
        self.max_cap = tf.constant(max_cap_list, dtype=tf.float32)

    def _high_bounds(self, high_bound_list):
        self.high_bound = tf.constant(high_bound_list, dtype=tf.float32)

    def _low_bounds(self, low_bound_list):
        self.low_bound = tf.constant(low_bound_list, dtype=tf.float32)

    def _rains(self, rain_list):
        self.rain = tf.constant(rain_list, dtype=tf.float32)

    def _downstream(self, downstream):
        np_downstream = np.zeros((self.reservoir_num, self.reservoir_num))
        for i in downstream:
            m = self.reservoirs.index(i[0])
            n = self.reservoirs.index(i[1])
            np_downstream[m, n] = 1
        self.downstream = tf.constant(np_downstream, dtype=tf.float32)

    def MAXCAP(self):
        return self.max_cap

    def HIGH_BOUND(self):
        return self.high_bound

    def LOW_BOUND(self):
        return self.low_bound

    def RAIN(self):
        return self.rain

    def DOWNSTREAM(self):
        return self.downstream

    def BIGGESTMAXCAP(self):
        return self.biggestmaxcap

    def Reward(self, states):
        new_rewards = tf.where(
            tf.logical_and(tf.greater_equal(states, self.LOW_BOUND()), tf.less_equal(states, self.HIGH_BOUND())),
            self.zero,
            tf.where(tf.less(states, self.LOW_BOUND()),
                      -5 * (self.LOW_BOUND() - states),
                      -100 * (states - self.HIGH_BOUND()))
            )
        new_rewards += tf.abs(((self.HIGH_BOUND() + self.LOW_BOUND()) / 2.0) - states) * (-0.1)
        return tf.reduce_sum(new_rewards, 1, keep_dims=True)


class NAVI(object):
    def __init__(self,
                 batch_size,
                 default_settings):
        self.__dict__.update(default_settings)
        self.batch_size = batch_size

    def MINMAZEBOUND(self):
        return self.min_maze_bound

    def MAXMAZEBOUND(self):
        return self.max_maze_bound

    def MINACTIONBOUND(self):
        return self.min_act_bound

    def MAXACTIONBOUND(self):
        return self.max_act_bound

    def GOAL(self):
        return self.goal

    def CENTER(self):
        return self.centre

    def Reward(self, states):
        new_reward = -tf.reduce_sum(tf.abs(states - self.GOAL()), 1, keep_dims=True)
        return new_reward


# Matrix computation version update
class HVAC(object):
    def __init__(self,
                 adj_outside,  # Adjacent to outside
                 adj_hall,  # Adjacent to hall
                 adj,  # Adjacent between rooms
                 rooms,  # Room names
                 batch_size,
                 default_settings):
        self.__dict__.update(default_settings)
        self.rooms = rooms
        self.batch_size = batch_size
        self.room_size = len(rooms)
        self.zero = tf.constant(0, shape=[self.batch_size, self.room_size], dtype=tf.float32)
        self._init_ADJ_Matrix(adj)
        self._init_ADJOUT_MATRIX(adj_outside)
        self._init_ADJHALL_MATRIX(adj_hall)

    def _init_ADJ_Matrix(self, adj):
        np_adj = np.zeros((self.room_size, self.room_size))
        for i in adj:
            m = self.rooms.index(i[0])
            n = self.rooms.index(i[1])
            np_adj[m, n] = 1
            np_adj[n, m] = 1
        self.adj = tf.constant(np_adj, dtype=tf.float32)
        print('self.adj shape:{0}'.format(self.adj.get_shape()))

    def _init_ADJOUT_MATRIX(self, adj_outside):
        np_adj_outside = np.zeros((self.room_size,))
        for i in adj_outside:
            m = self.rooms.index(i)
            np_adj_outside[m] = 1
        self.adj_outside = tf.constant(np_adj_outside, dtype=tf.float32)

    def _init_ADJHALL_MATRIX(self, adj_hall):
        np_adj_hall = np.zeros((self.room_size,))
        for i in adj_hall:
            m = self.rooms.index(i)
            np_adj_hall[m] = 1
        self.adj_hall = tf.constant(np_adj_hall, dtype=tf.float32)

    def ADJ(self):
        return self.adj

    def ADJ_OUTSIDE(self):
        return self.adj_outside

    def ADJ_HALL(self):
        return self.adj_hall

    def R_OUTSIDE(self):
        return self.outside_resist

    def R_HALL(self):
        return self.hall_resist

    def R_WALL(self):
        return self.wall_resist

    def CAP(self):
        return self.cap

    def CAP_AIR(self):
        return self.cap_air

    def COST_AIR(self):
        return self.cost_air

    def TIME_DELTA(self):
        return self.time_delta

    def TEMP_AIR(self):
        return self.temp_air

    def TEMP_UP(self):
        return self.temp_up

    def TEMP_LOW(self):
        return self.temp_low

    def TEMP_OUTSIDE(self):
        return self.temp_outside

    def TEMP_HALL(self):
        return self.temp_hall

    def PENALTY(self):
        return self.penalty

    def AIR_MAX(self):
        return self.air_max

    def ZERO(self):
        return self.zero

    def Reward(self, states, actions):
        batch_size, state_size = states.get_shape()
        # break_penalty = tf.select(tf.logical_or(tf.less(states,self.TEMP_LOW()),\
        #                                        tf.greater(states,self.TEMP_UP())),self.PENALTY()+self.ZERO(),self.ZERO())
        dist_penalty = tf.abs(((self.TEMP_UP() + self.TEMP_LOW()) / tf.constant(2.0, dtype=tf.float32)) - states)
        ener_penalty = actions * self.COST_AIR()
        new_rewards = -tf.reduce_sum(tf.constant(10.0, tf.float32) * dist_penalty + ener_penalty, 1, keep_dims=True)
        return new_rewards