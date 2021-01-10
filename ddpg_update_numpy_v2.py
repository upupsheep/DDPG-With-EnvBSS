from __future__ import absolute_import, division, print_function
import os.path
import sys
import math
import time

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym_BSS  # noqa: F401

name = sys.argv[1] if len(sys.argv) > 1 else 'BSSEnvTest-v0'
env = gym.make(name)  # gym.Env
env.seed(42)
tf.compat.v1.disable_eager_execution()

# print(env.observation_space, env.action_space)
print(name)
print(env.metadata)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#####################  hyper parameters  ####################
LR_A = 0.0001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.001
MEMORY_CAPACITY = 10000  # 10000
c = 0.1  # 0.1
BATCH_SIZE = 64  # 128
episode_num = 5000  # 10000
LAMBDA = 10000
EPSILON = 0.1
#####################  BSS data functions  ####################
# penalty term
mu = 0.0

###############################  DDPG  ####################################
""" Writing activation function """


def clipping(action_mtx):
    # print("\n--- In clipping activation function ---")
    # print("a_bound: ", a_bound)
    print("clipping action mtx: ", action_mtx)
    # print("x: ", type(action_mtx))
    # if type(action_mtx) is tuple:
    #     # [[xx, xx, xx, xx]], and scaled_a here
    #     action = action_mtx[0] * a_bound
    # else:
    #     action = action_mtx * a_bound
    # print("scaled x: ", action)
    clipping_result = np.zeros(action_mtx.shape)
    batch_num = action_mtx.shape[0]
    for batch_idx in range(batch_num):
        action = action_mtx[batch_idx] * a_bound
        # print("action mtx: ", action_mtx)
        # print("clipping action: ", action)
        # adjust to y
        maxa = action[int(np.argmax(action))]
        mina = action[int(np.argmin(action))]
        lower = np.zeros(a_dim)
        y = np.zeros(a_dim)

        # Avoid [nan nan nan nan]
        # '''

        is_nan = []
        for a in action:
            is_nan.append(math.isnan(a))
        # print("is nan: ", is_nan)
        if np.all(is_nan):
            exit(0)
            return np.array(a_bound)
        # '''

        # print(env.nbikes, "bike_num")
        # print(a_bound, "abound")
        for i in range(a_dim):
            # if x[k] is in the bound, then no need to clip
            if action[i] <= a_bound[i] and action[i] >= lower[i]:
                y[i] = action[i]
            else:
                y[i] = lower[i]+(a_bound[i]-lower[i]) * \
                    (action[i]-mina)/(maxa-mina+1e-6)
                # if math.isnan(y[i])
                # if maxa == mina:
                #     exit(0)
        # print("clipping y: ", y)
        # print("------------------\n")

        mu = float(LAMBDA) * float(np.abs(1 - np.sum(y)) +
                                   np.abs(env.nbikes - np.sum(y)))
        clipping_result[batch_idx] = y

    return clipping_result


# np_clipping = np.vectorize(clipping)  # vectorize the python function # <-- no need


""" Gradient of Activation """


def d_clipping(action_mtx):
    # print("\n--- In (d) clipping activation function ---")
    # print("(d) action mtx: ", action_mtx * a_bound)
    batch_num = action_mtx.shape[0]
    clipping_gradient_result = np.zeros((a_dim, a_dim))

    for batch_idx in range(batch_num):
        # [[xx, xx, xx, xx]], and scaled_a here
        x = action_mtx[batch_idx] * a_bound
        # print("scaled x: ", action)
        lower = np.zeros(a_dim)

        # Avoid [nan nan nan nan]
        # '''
        is_nan = []
        for a in action:
            is_nan.append(math.isnan(a))
        # print("is nan: ", is_nan)
        if np.all(is_nan):
            exit(0)
            # assert np.all(is_nan)
            x = np.array(a_bound)
        # '''

        # compute gradient
        max_i = np.argmax(x)
        min_i = np.argmin(x)
        # print("max_i: ", max_i)
        # print("min_i: ", min_i)

        grad = np.zeros((a_dim, a_dim))

        for i in range(a_dim):
            if (i == max_i or i == min_i):
                continue
            # y[k] = upper[k] + (upper[k]-lower[k]) * {(x[i]-min(x))/(max(x)-min(x))}
            grad[i][i] = (a_bound[i]-lower[i]) / (x[max_i] - x[min_i] + 1e-6)

        clipping_gradient_result += grad
    print('clipping_gradient: ', clipping_gradient_result / batch_num)
    # print("------------------\n")
    return clipping_gradient_result / batch_num

# np_d_clipping = np.vectorize(d_clipping) # don't need this one!


def np_d_clipping_32(x):
    return d_clipping(x).astype(np.float32)


def tf_d_clipping(x, name=None):
    with tf.compat.v1.name_scope(name, "d_clipping", [x]) as name:
        y = py_func(np_d_clipping_32,  # forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    stateful=False)

        # when using with the code, it is used to specify the rank of the input.
        y[0].set_shape(x.get_shape())
        return y[0]


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.compat.v1.py_func(func, inp, Tout, stateful=stateful, name=name)


""" Gradient Function """


def clipping_grad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_clipping(x)  # defining the gradient
    # grad * n_gr : [3, 3] vs [64, 3]
    n_gr_avg = tf.reduce_mean(n_gr, 0)
    return grad * n_gr_avg


""" Combining it all together """


def np_clipping_32(x):
    return clipping(x).astype(np.float32)


def tf_clipping(x, name=None):
    with tf.compat.v1.name_scope(name, "clipping", [x]) as name:
        y = py_func(np_clipping_32,  # forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    grad=clipping_grad)  # the function that overrides gradient

        # when using with the code, it is used to specify the rank of the input.
        y[0].set_shape(x.get_shape())
        return y[0]


""" Writing activation function """


def optLayer(y_mtx):
    # print("\n--- In optLayer activation function ---")
    # print("optlayer y_mtx: ", y_mtx.shape)
    # adjust to y
    # exit(0)
    # maxa = action[int(np.argmax(action))]
    # mina = action[int(np.argmin(action))]
    lower = np.zeros(a_dim, dtype=np.float32)
    # y = np.zeros(a_dim)
    # # print(env.nbikes,"bike_num")

    batch_num = y_mtx.shape[0]
    opt_result = np.zeros(y_mtx.shape)
    # # print(a_bound,"abound")
    # for i in range(a_dim):
    #     y[i] = lower[i]+(a_bound[i]-lower[i])*(action[i]-mina)/(maxa-mina)
    # print(y,"y")
    for batch_idx in range(batch_num):
        y = y_mtx[batch_idx]
        # adjust to z
        z = np.zeros(a_dim, dtype=np.float32)

        #start algorithm#
        phase = 0  # lower=0 , upeer=1 , done=2
        C_unclamp = env.nbikes             # how many left bike to distribute
        set_unclamp = set(range(a_dim))    # unclamp set
        unclamp_num = a_dim                # unclamp number=n'
        grad_z = np.zeros((a_dim, a_dim))   # grad_z is 4*4 arrray
        while phase != 2:
            sum_y = 0
            set_clamp_round = set()  # indices clamped in this iteration of the while loop
            # algorithm line 7
            for i in range(a_dim):
                if i in set_unclamp:
                    sum_y = sum_y+y[i]
            for i in range(a_dim):
                if i in set_unclamp:
                    z[i] = y[i]+(C_unclamp-sum_y)/unclamp_num
            # print(z,"z")
            # print(sum_y,"sum_y")
            # algorithm line8
            for i in range(a_dim):
                if i in set_unclamp:
                    for j in range(a_dim):
                        if j in set_unclamp:
                            if (i != j):
                                grad_z[i][j] = -1/unclamp_num
                            else:
                                grad_z[i][j] = 1 - (1/unclamp_num)
        # print(grad_z)
            # algorithm line 9
            for j in range(a_dim):
                if j not in set_unclamp:
                    for i in range(a_dim):
                        grad_z[i][j] = 0
        # print(grad_z,"grad before clamp in this iteration")

            # algorithm lin 10~20
            for i in range(a_dim):
                if i in set_unclamp:
                    if z[i] < lower[i] and phase == 0:
                        z[i] = lower[i]
                        for j in range(a_dim):
                            grad_z[i][j] = 0
                        set_clamp_round.add(i)
                    elif (z[i] > a_bound[i]) and phase == 1:
                        z[i] = a_bound[i]
                        for j in range(a_dim):
                            grad_z[i][j] = 0
                        set_clamp_round.add(i)
        # print(z,"z_after clamp")
        # print(grad_z,"grad after clamp")
            # algorithm 21~25
            unclamp_num = unclamp_num-len(set_clamp_round)
        #   print(unclamp_num,"unclamp")
            for i in range(a_dim):
                if i in set_clamp_round:
                    C_unclamp = C_unclamp-z[i]
        # print(C_unclamp,"C")
            set_unclamp = set_unclamp.difference(set_clamp_round)
        #  print(set_unclamp,"unclamp set")
            if len(set_clamp_round) == 0:
                phase = phase+1

        # debug after optlayer
        # print('optlayer z: ', z)
        final_sum = 0
        for i in range(a_dim):
            final_sum = final_sum+z[i]
            # make sure not violate the local constraint
            assert lower[i] <= z[i] <= a_bound[i]
        final_sum = round(final_sum, 2)
    # print(final_sum)
        assert final_sum == env.nbikes     # make sure sum is equal to bike number
        if np.sum(y) == env.nbikes:
            assert z == y

            # print("z: ", sum(z))
            # opt_result[batch_idx] = z
        # print("opt_result: ", opt_result)
        opt_result[batch_idx] = z
        # print("------------------\n")
    return opt_result


# np_clipping = np.vectorize(clipping)  # vectorize the python function # <-- no need


""" Gradient of Activation """


def d_optLayer(y_mtx):
    # print("\n--- In (d) optLayer activation function ---")
    # adjust to y
    # exit(0)
    # maxa = action[int(np.argmax(action))]
    # mina = action[int(np.argmin(action))]
    lower = np.zeros(a_dim)
    # y = np.zeros(a_dim)
    # # print(env.nbikes,"bike_num")

    # # print(a_bound,"abound")
    # for i in range(a_dim):
    #     y[i] = lower[i]+(a_bound[i]-lower[i])*(action[i]-mina)/(maxa-mina)
    # # print(y,"y")
    # adjust to z

    opt_gradient_result = np.zeros((a_dim, a_dim))
    batch_num = y_mtx.shape[0]
    for batch_idx in range(batch_num):
        y = y_mtx[batch_idx]
        z = np.zeros(a_dim)

        #start algorithm#
        phase = 0  # lower=0 , upeer=1 , done=2
        C_unclamp = env.nbikes             # how many left bike to distribute
        set_unclamp = set(range(a_dim))    # unclamp set
        unclamp_num = a_dim                # unclamp number=n'
        grad_z = np.zeros((a_dim, a_dim))   # grad_z is 4*4 arrray
        while phase != 2:
            sum_y = 0
            set_clamp_round = set()  # indices clamped in this iteration of the while loop
            # algorithm line 7
            for i in range(a_dim):
                if i in set_unclamp:
                    sum_y = sum_y+y[i]
            for i in range(a_dim):
                if i in set_unclamp:
                    z[i] = y[i]+(C_unclamp-sum_y)/unclamp_num
            # print(z,"z")
            # print(sum_y,"sum_y")
            # algorithm line8
            for i in range(a_dim):
                if i in set_unclamp:
                    for j in range(a_dim):
                        if j in set_unclamp:
                            if (i != j):
                                grad_z[i][j] = -1/unclamp_num
                            else:
                                grad_z[i][j] = 1 - (1/unclamp_num)
        # print(grad_z)
            # algorithm line 9
            for j in range(a_dim):
                if j not in set_unclamp:
                    for i in range(a_dim):
                        grad_z[i][j] = 0
        # print(grad_z,"grad before clamp in this iteration")

            # algorithm lin 10~20
            for i in range(a_dim):
                if i in set_unclamp:
                    if z[i] < lower[i] and phase == 0:
                        z[i] = lower[i]
                        for j in range(a_dim):
                            grad_z[i][j] = 0
                        set_clamp_round.add(i)
                    elif (z[i] > a_bound[i]) and phase == 1:
                        z[i] = a_bound[i]
                        for j in range(a_dim):
                            grad_z[i][j] = 0
                        set_clamp_round.add(i)
        # print(z,"z_after clamp")
        # print(grad_z,"grad after clamp")
            # algorithm 21~25
            unclamp_num = unclamp_num-len(set_clamp_round)
        #   print(unclamp_num,"unclamp")
            for i in range(a_dim):
                if i in set_clamp_round:
                    C_unclamp = C_unclamp-z[i]
        # print(C_unclamp,"C")
            set_unclamp = set_unclamp.difference(set_clamp_round)
        #  print(set_unclamp,"unclamp set")
            if len(set_clamp_round) == 0:
                phase = phase+1

        # debug after optlayer
    #     final_sum = 0
    #     for i in range(a_dim):
    #         final_sum = final_sum+z[i]
    #         # make sure not violate the local constraint
    #         assert lower[i] <= z[i] <= a_bound[i]
    #     final_sum = round(final_sum, 2)
    #    # print(final_sum)
    #     assert final_sum == env.nbikes     # make sure sum is equal to bike number
    #     if np.sum(y) == env.nbikes:
    #         assert z == y
        opt_gradient_result += grad_z
    print("opt_gradient: ", opt_gradient_result / batch_num)
    # print("------------------\n")
    return opt_gradient_result / batch_num

# np_d_clipping = np.vectorize(d_clipping) # don't need this one!


def np_d_optLayer_32(x):
    return d_optLayer(x).astype(np.float32)


def tf_d_optLayer(x, name=None):
    with tf.compat.v1.name_scope(name, "d_optLayer", [x]) as name:
        y = py_func(np_d_optLayer_32,  # forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    stateful=False)

        # when using with the code, it is used to specify the rank of the input.
        y[0].set_shape(x.get_shape())
        return y[0]


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad_1' + str(np.random.randint(0, 1E+2))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.compat.v1.py_func(func, inp, Tout, stateful=stateful, name=name)


""" Gradient Function """


def optLayer_grad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_optLayer(x)  # defining the gradient
    # print('grad: ', grad)
    # print('n_gr: ', n_gr)
    # exit(0)
    # print('mul: ', grad * n_gr)
    # [3,3] vs [64,3]
    # print('[0]: ', tf.gather(n_gr, 0))
    # exit(0)
    n_gr_avg = tf.reduce_mean(n_gr, 0)
    return grad * n_gr_avg


""" Combining it all together """


def np_optLayer_32(x):
    return optLayer(x).astype(np.float32)


def tf_optLayer(x, name=None):
    with tf.compat.v1.name_scope(name, "optLayer", [x]) as name:
        y = py_func(np_optLayer_32,  # forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    grad=optLayer_grad)  # the function that overrides gradient

        # when using with the code, it is used to specify the rank of the input.
        y[0].set_shape(x.get_shape())
        return y[0]


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.compat.v1.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        # mu = tf.abs(1 - tf.reduce_sum(self.a))

        td_error = tf.compat.v1.losses.mean_squared_error(
            labels=q_target, predictions=q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(
            LR_C).minimize(td_error, var_list=self.ce_params)

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(LR_A)
        a_loss = - tf.reduce_mean(input_tensor=q)    # maximize the q
        self.grads_and_vars = list(optimizer.compute_gradients(
            a_loss, self.ae_params))
        # print("grad[2]: ", self.grads_and_vars[2][0])
        # self.grads_and_vars = [((grad @ opt_grad), var)
        #                        for grad, var in self.grads_and_vars_noOpt]
        # for grad, var in self.grads_and_vars:
        # print("grad_and_var: ", self.grads_and_vars)
        # print("grad[2][0]: ", self.grads_and_vars[2][0])
        # print("opt_grad: ", opt_grad)
        # exit(0)
        # print("g and v [4]: ", self.grads_and_vars[4])
        # print("g and v [5]: ", self.grads_and_vars[5])
        # exit(0)

        # self.grads_and_vars[4] = (
        #     self.grads_and_vars[4][0] @ opt_grad, self.grads_and_vars[4][1])

        '''
        opt_weight = np.ones((self.a_dim, self.a_dim))
        opt_bias_grad = np.zeros((self.a_dim, self.a_dim))
        opt_bias_weight = np.zeros((self.a_dim, self.a_dim))
        tf_opt_grad = tf.convert_to_tensor(opt_grad, dtype=tf.float32)
        tf_opt_weight = tf.convert_to_tensor(opt_weight, dtype=tf.float32)
        tf_opt_bias_grad = tf.convert_to_tensor(
            opt_bias_grad, dtype=tf.float32)
        tf_opt_bias_weight = tf.convert_to_tensor(
            opt_bias_weight, dtype=tf.float32)
        self.grads_and_vars.append((tf_opt_grad, tf_opt_weight))
        self.grads_and_vars.append((tf_opt_bias_grad, tf_opt_bias_weight))
        '''

        # print("grad_and_vars: ", self.grads_and_vars)
       # self.optgrad=tf.zeros([a_dim, a_dim])
       # self.gv_opt_fn=[(gv[0]*self.optgrad,gv[1])for gv in self.grads_and_vars]
        self.opttrain = optimizer.apply_gradients(self.grads_and_vars)

        # print("a_loss: ", a_loss)
        # print("self.ae_params: ", self.ae_params)
        self.atrain = tf.compat.v1.train.AdamOptimizer(
            LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):
        # print("self.a: ", self.a)
        # return self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        # print("AAAA: ", self.sess.run(self.a, {self.S: s[np.newaxis, :]}))
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # print("learn!!!")
        # soft target replacement
        # variable_names = [v.name for v in tf.compat.v1.trainable_variables()]
        # values = self.sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("variable: ", k)
        #     # print("shape: ", v.shape)
        #     # print(v)
        # exit(0)
        self.sess.run(self.soft_replace)

        # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # Get sampling range
        record_range = min(self.pointer, MEMORY_CAPACITY)
        # Randomly sample indices
        indices = np.random.choice(record_range, size=BATCH_SIZE)

        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        a, g = self.sess.run([self.atrain, self.grads_and_vars], {self.S: bs})
        # a, g = self.sess.run(self.atrain, {self.S: bs})
        # '''
        # one more layer 9, 2 >> 11, 2  but add(a_loss, self.ae_params) become 4, 2
        print("=== g ===")
        print(np.array(g).shape)
        print("=========")
        print(g[0][0].shape)
        print(g[0][1].shape)
        print("Q")
        print(g[1][0].shape)
        print(g[1][1].shape)
        print("QQ")
        print(g[2][0].shape)
        print(g[2][1].shape)
        print("QQQ")
        print(g[3][0].shape)
        print(g[3][1].shape)
        print("QQQQ")
        print(g[4][0].shape)
        print(g[4][1].shape)
        print("QQQQQQ")
        print(g[5][0].shape)
        print(g[5][1].shape)
        print("QQQQ")
        print(g[4][0])
        print(g[4][1])
        # print(g.gg)  # to terminal
        # '''
        # print("bs: ", bs)
        # self.sess.run(self.atrain, {self.S: bs})

        self.sess.run(self.ctrain, {self.S: bs,
                                    self.a: ba, self.R: br, self.S_: bs_})
        # print(g.gg)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net_1 = tf.compat.v1.layers.dense(
                s, 32, activation=tf.nn.relu, name='l1', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),  trainable=trainable)
            net_2 = tf.compat.v1.layers.dense(
                net_1, 16, activation=tf.nn.relu, name='l2', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1), trainable=trainable)
            a = tf.compat.v1.layers.dense(
                net_2, self.a_dim, activation=tf.nn.tanh, name='a', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1), trainable=trainable)
            '''
            scaled_a = tf.multiply(a, self.a_bound, name='scaled_a')
            print('scaled_a: ', scaled_a)
            '''
            # customized activation function (clipping)
            a_clip = tf_clipping(a)
            # a_clip = tf.compat.v1.layers.dense(a, self.a_dim, activation=tf_clipping, name='a_clip',
            #                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1), trainable=trainable)
            a_opt = tf_optLayer(a_clip)
            # a_opt = tf.compat.v1.layers.dense(a_clip, self.a_dim, activation=tf_optLayer, name='a_opt',
            #                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1), trainable=trainable)
            return a_opt

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            # Q(s, a)
            n_l1 = 32
            w1_s = tf.compat.v1.get_variable(
                'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable(
                'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable(
                'b1', [1, n_l1], trainable=trainable)

            # penalty term
            mu_vector = tf.fill([1, n_l1], mu)
            penalty_term = tf.compat.v1.get_variable(
                name='penalty_term', initializer=mu_vector, trainable=trainable)
            # print("penalty_term: ", penalty_term.trainable)
            # exit(0)

            net_1_act = tf.nn.relu(tf.matmul(s, w1_s) +
                                   tf.reshape(
                tf.matmul([a], w1_a), [-1, 32]) + b1 - penalty_term)  # (1, None, 30) -> (None, 30)
            # + tf.multiply(float(LAMBDA), xxyy)
            # Q(s,a)
            net_1 = tf.compat.v1.layers.dense(
                net_1_act, 16, name='l1_c', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1), trainable=trainable)
            # net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            net_2 = tf.compat.v1.layers.dense(
                net_1, 1, activation=tf.nn.relu, name='l2_c', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1), trainable=trainable)
            return net_2

################ Opt layer#####################


def OptLayer_function(action):
    # adjust to y
    # exit(0)
    print("action: ", action)
    maxa = action[int(np.argmax(action))]
    mina = action[int(np.argmin(action))]
    lower = np.zeros(a_dim)
    y = np.zeros(a_dim)
    # print(env.nbikes,"bike_num")

    # print(a_bound,"abound")
    for i in range(a_dim):
        if action[i] <= a_bound[i] and action[i] >= lower[i]:  # no need to clip
            y[i] = action[i]
        else:
            y[i] = lower[i]+(a_bound[i]-lower[i])*(action[i]-mina)/(maxa-mina)
    print(y, "y")
    # adjust to z
    z = np.zeros(a_dim)

    #start algorithm#
    phase = 0  # lower=0 , upeer=1 , done=2
    C_unclamp = env.nbikes             # how many left bike to distribute
    set_unclamp = set(range(a_dim))    # unclamp set
    unclamp_num = a_dim                # unclamp number=n'
    grad_z = np.zeros((a_dim, a_dim))   # grad_z is 4*4 arrray
    while phase != 2:
        sum_y = 0
        set_clamp_round = set()  # indices clamped in this iteration of the while loop
        # algorithm line 7
        for i in range(a_dim):
            if i in set_unclamp:
                sum_y = sum_y+y[i]
        for i in range(a_dim):
            if i in set_unclamp:
                z[i] = y[i]+(C_unclamp-sum_y)/unclamp_num
        # print(z,"z")
        # print(sum_y,"sum_y")
        # algorithm line8
        for i in range(a_dim):
            if i in set_unclamp:
                for j in range(a_dim):
                    if j in set_unclamp:
                        if (i != j):
                            grad_z[i][j] = -1/unclamp_num
                        else:
                            grad_z[i][j] = 1 - (1/unclamp_num)
       # print(grad_z)
        # algorithm line 9
        for j in range(a_dim):
            if j not in set_unclamp:
                for i in range(a_dim):
                    grad_z[i][j] = 0
      # print(grad_z,"grad before clamp in this iteration")

        # algorithm lin 10~20
        for i in range(a_dim):
            if i in set_unclamp:
                if z[i] < lower[i] and phase == 0:
                    z[i] = lower[i]
                    for j in range(a_dim):
                        grad_z[i][j] = 0
                    set_clamp_round.add(i)
                elif (z[i] > a_bound[i]) and phase == 1:
                    z[i] = a_bound[i]
                    for j in range(a_dim):
                        grad_z[i][j] = 0
                    set_clamp_round.add(i)
       # print(z,"z_after clamp")
       # print(grad_z,"grad after clamp")
        # algorithm 21~25
        unclamp_num = unclamp_num-len(set_clamp_round)
     #   print(unclamp_num,"unclamp")
        for i in range(a_dim):
            if i in set_clamp_round:
                C_unclamp = C_unclamp-z[i]
       # print(C_unclamp,"C")
        set_unclamp = set_unclamp.difference(set_clamp_round)
      #  print(set_unclamp,"unclamp set")
        if len(set_clamp_round) == 0:
            phase = phase+1

    # debug after optlayer
    print("z: ", z)
    final_sum = 0
    for i in range(a_dim):
        final_sum = final_sum+z[i]
        # make sure not violate the local constraint
        assert lower[i] <= z[i] <= a_bound[i]
    final_sum = round(final_sum, 2)
    print(final_sum)
    assert final_sum == env.nbikes     # make sure sum is equal to bike number
    # if np.sum(y) == env.nbikes:
    #     print(y, env.nbikes)
    #     assert z == y
    return z


###############################  training  ####################################
Rs = []
ewma_reward = 0  # EWMA reward for tracking the learning progress
ewma_reward_s = []

# 2*ZONE+1 ZONE's Demand,zone's number of resource on zone K (dS_) +time
s_dim = env.observation_space.shape[0]
# equal to get_observe function in env
a_dim = env.action_space.shape[0]  # 4

# print(a_dim,"YEEEEEEE")
# print(env.action_space.low,"low")
a_bound = env.action_space.high  # bound , in txt file
opt_grad = np.zeros((a_dim, a_dim), dtype=np.float32)

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration

for ep in range(episode_num):  # 100000
    R = 0
    ld_pickup = 0
    ld_dropoff = 0
    revenue = 0
    scenario = None
    done = False
    s = env.reset()  # [0,0,0,0,8,7,8,8,0]
    # print(s)
    # policy = read_supriyo_policy_results(env)
    while not done:
        # action = None
        action = ddpg.choose_action(s)
        # print("before: ", action)
        # if action[0] == 0. and action[1] == 0. and action[2] == 0.:
        #     exit(0)
        # if np.all(action == 30.):
        #     exit(0)
        # print("sum of action: ", sum(action))
        # Add exploration noise
        # action = clipping(np.random.normal(action, var))

        # noise = np.random.randint(0, a_bound/2)
        # action = OptLayer_function(np.random.normal(action, var))
        # action = OptLayer_function(action + noise)
        # print("after: ", action)

        # print("In DDPG main, x =", action)
        # action, opt_grad = OptLayer_function(action, a_dim, a_bound, env)
        # print("opt_grad: ", opt_grad)
        # exit(0)
        # print(action,"After_modify")
        # print(obs)
        # action = get_supriyo_policy_action(env, obs, policy)

        # action = None
        s_, r, done, info = env.step(action)
        # print(done)
        # print("{}, {}".format(ddpg.pointer, done))
        ddpg.store_transition(s, action, r, s_)
        # time.sleep(0.5)
        if ddpg.pointer > c*MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            # print("LEARN!!!")
            ddpg.learn()
        s = s_
        R += r
        ld_pickup += info["lost_demand_pickup"]
        ld_dropoff += info["lost_demand_dropoff"]
        revenue += info["revenue"]
        scenario = info["scenario"]

    # update EWMA reward and log the results
    ewma_reward = 0.05 * R + (1 - 0.05) * ewma_reward

    print({
        'episode': ep,
        'ewma reward': ewma_reward,
        # 'ep reward': R,
        'Explore': var,
        'lost_demand_pickup': ld_pickup,
        "lost_demand_dropoff": ld_dropoff,
        "revenue": revenue,
        "scenario": scenario
    })
    Rs.append(R)
    ewma_reward_s.append(ewma_reward)

Rs = np.array(Rs)
ewma_reward_s = np.array(ewma_reward_s)

print('')
print('---------------------------')
print('Average reward per episode:', np.average(Rs))

"""
Save rewards to file
"""
np.save('ewma_reward', ewma_reward_s)
np.save('ep_reward', Rs)

xAxis = np.arange(episode_num)
yAxis = ewma_reward_s

plt.plot(xAxis, yAxis)
plt.title('Memory: {}, Batch size: {}, Episode: {}'.format(
    MEMORY_CAPACITY, BATCH_SIZE, episode_num))
plt.xlabel('Episode')
plt.ylabel('EWMA Reward')
plt.show()
