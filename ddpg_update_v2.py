import os.path
import sys

import gym
import numpy as np
import tensorflow as tf
import gym_BSS  # noqa: F401

name = sys.argv[1] if len(sys.argv) > 1 else 'BSSEnvTest-v0'
env = gym.make(name)  # gym.Env
env.seed(42)
# print(env.observation_space, env.action_space)
print(name)
print(env.metadata)

#####################  hyper parameters  ####################
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
#####################  BSS data functions  ####################


def get_supriyo_policy_action(env, obs, policy):
    ypcmu, yncmu = policy
    env = env.unwrapped
    current_alloc = obs[env.nzones:2 * env.nzones]
    # print(current_alloc)
    # print(sum(current_alloc))

    current_time = int(obs[-1])
    # print(current_alloc)
    yp_t = np.array(ypcmu[current_time])
    yn_t = np.array(yncmu[current_time])
    return current_alloc + yn_t - yp_t


def read_supriyo_policy_results(env):
    env = env.unwrapped
    scenario = env._scenario
    ypcmu = [[0.0 for k in range(env.nzones)] for j in range(env.ntimesteps)]
    yncmu = [[0.0 for k in range(env.nzones)] for j in range(env.ntimesteps)]
    f1 = open(os.path.join(env.data_dir, "Our_policy",
                           "policy_result{0}.csv".format(scenario)))
    line = f1.readline()
    # line = f1.readline()
    print(scenario)
    while(line != ""):
        line = line.strip(" \n")
        line = line.split(",")
        if(int(line[0]) < 100):
            ypcmu[int(line[0]) + 1][int(line[1])] = float(line[2])  # move out
            yncmu[int(line[0]) + 1][int(line[1])] = float(line[3])  # move in
        line = f1.readline()
    f1.close()
    return (ypcmu, yncmu)

###############################  DDPG  ####################################


def MyCapper(gv, grad):
    print("============")
    print("gv: ", gv)
    print("gv[0] ", gv[0])
    print("gv[1] ", gv[1])
    # print("grad: ", grad)
    print("============")
    return 1


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, env, s_init):
        tf.compat.v1.disable_eager_execution()

        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.compat.v1.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.env = env
        self.S = tf.compat.v1.placeholder(tf.float64, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float64, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float64, [None, 1], 'r')

        with tf.compat.v1.variable_scope('Actor'):
            self.a, a_grad = self._build_a(
                self.S, scope='eval', trainable=True)
            a_, _ = self._build_a(self.S_, scope='target', trainable=False)
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
        td_error = tf.compat.v1.losses.mean_squared_error(
            labels=q_target, predictions=q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(
            td_error, var_list=self.ce_params)

        print(q)
        print(td_error)
        print("a_grad: ", a_grad)

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(LR_A)
        # self.atrain = optimizer.apply_gradients(a_grad)

        a_loss = - tf.reduce_mean(input_tensor=q)    # maximize the q
        # self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        grads_and_vars = optimizer.compute_gradients(a_loss, self.ae_params)
        print("grades and vars: ", grads_and_vars)
        capped_grads_and_vars = [(MyCapper(gv, a_grad))
                                 for gv in grads_and_vars]
        self.atrain = optimizer.apply_gradients(capped_grads_and_vars)

        self.sess.run(tf.compat.v1.global_variables_initializer(), feed_dict={
                      self.S: s_init[np.newaxis, :], self.S_: s_init[np.newaxis, :]})
        #

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs,
                                    self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(
                s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            print(net, "net")
            a = tf.compat.v1.layers.dense(
                net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            print(a, "a")
            a = tf.multiply(a, self.a_bound, name='scaled_a')
            print(a, "a")
            a, grad = OptLayer_function(a, self.a_dim, self.a_bound, self.env)
            # grad = 0
            print(a, "a_after_opt")
            print(grad)
            return a, grad

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.compat.v1.get_variable(
                'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable(
                'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable(
                'b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # Q(s,a)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable)

################ Opt layer#####################


def OptLayer_function(action, a_dim, a_bound, env):
        # adjust to y
    print(action, "action")

    maxa = tf.reduce_max(input_tensor=action)
    mina = tf.reduce_min(input_tensor=action)
    lower = tf.zeros(a_dim, dtype=tf.float64)
    tfa_bound = tf.convert_to_tensor(value=a_bound, dtype=tf.float64)
    y = tf.zeros(a_dim, dtype=tf.float64)
    y = lower+(tfa_bound-lower)*(action-mina)/(maxa-mina)

    print(y, "y")
    # maxa=action[tf.math.argmax(action)]
    # mina=action[np.argmin(action)]
    # lower=np.zeros(a_dim)
    # y=np.zeros(a_dim)

    # adjust to z
    z = tf.zeros(a_dim, dtype=tf.float64)
    # start algorithm#
    phase = 0  # lower=0 , upeer=1 , done=2
    # how many left bike to distribute
    C_unclamp = tf.Variable(float(30), dtype=tf.float64)
    set_unclamp = set(range(a_dim))    # unclamp set
    # unclamp number=n'
    unclamp_num = tf.Variable(float(a_dim), dtype=tf.float64)
    # grad_z is 4*4 arrray
    grad_z = tf.zeros([a_dim, a_dim], dtype=tf.float64)
    first = True
    while phase != 2:
        sum_y = tf.Variable(0.)
        cond = np.zeros(a_dim)
        set_clamp_round = set()  # indices clamped in this iteration of the while loop
        # algorithm line 7
        """
        for i in range(a_dim):
            if i in set_unclamp:
                # need better way, can change to the tf.where method
                sum_y = sum_y+tf.gather(y, i)
               """

        for i in range(a_dim):
            if i in set_unclamp:
                cond[i] = True
            else:
                cond[i] = False  # not calculate.
        # case_sum_true = y
        case_sum_true = tf.reshape(y, [a_dim])
        print("case_sum_true: ", case_sum_true)
        case_sum_false = tf.zeros(a_dim, dtype=tf.float64)
        sum_y = tf.compat.v1.where(cond, case_sum_true, case_sum_false)
        sum_y = tf.reduce_sum(input_tensor=sum_y)
        print(sum_y)
        print(cond, "cond test")
        case_true = y+(C_unclamp-sum_y)/unclamp_num
        case_true = tf.reshape(case_true, [a_dim])
        case_false = z
        z = tf.compat.v1.where(cond, case_true, case_false)
        condxy = np.zeros([a_dim, a_dim])
        # make sure the tensor shape the same to do tf.where
        grad_operator = tf.zeros([a_dim, a_dim], dtype=tf.float64)
        # algorithm line 8  3 phase to change
        for i in range(a_dim):
            for j in range(a_dim):
                if i not in set_unclamp:
                    condxy[i][j] = False
                elif j not in set_unclamp:
                    condxy[i][j] = False
                else:
                    condxy[i][j] = True
        case_grad_false = grad_z
        case_grad_true = grad_operator+1.0-(1.0/unclamp_num)
        grad_z = tf.compat.v1.where(condxy, case_grad_true, case_grad_false)

        for i in range(a_dim):
            if cond[i] == True:
                for j in range(a_dim):
                    if cond[j] == True and i == j:
                        condxy[i][j] = False
                    else:
                        condxy[i][j] = True
        case_grad_true = grad_operator-(1.0/unclamp_num)
        case_grad_false = grad_z
        grad_z = tf.compat.v1.where(condxy, case_grad_true, case_grad_false)

        # algorithm line 9
        for j in range(a_dim):
            if cond[j] == False:
                for i in range(a_dim):
                    condxy[i][j] = True
            else:
                for i in range(a_dim):
                    condxy[i][j] = False
        print(condxy, "BUFFFFF")
        case_grad_0_true = grad_operator
        case_grad_0_false = grad_z
        grad_z = tf.compat.v1.where(
            condxy, case_grad_0_true, case_grad_0_false)
        # algorithm lin 10~20
        if phase == 0:
            mask = tf.greater(lower, z)
            print("mask: ", mask)
            proto_tensor = tf.make_tensor_proto(mask)
            ndarry = tf.make_ndarray(proto_tensor)
            for i in range(a_dim):
                if i not in set_unclamp:
                    ndarry[i] = False
            z = tf.compat.v1.where(mask, lower, z)  # true,means i>z
            for i in range(a_dim):
                if ndarry[i] == True:
                    set_clamp_round.add(i)
                    for j in range(a_dim):
                        condxy[i][j] = True
                else:
                    for j in range(a_dim):
                        condxy[i][j] = False
            grad_z = tf.compat.v1.where(condxy, grad_operator, grad_z)
            temp_z = grad_z
        elif phase == 1:
            mask2 = tf.greater(z, tfa_bound)
            print(mask2, "maske_type")
            proto_tensor = tf.make_tensor_proto(mask2)
            ndarry = tf.make_ndarray(proto_tensor)

            for i in range(a_dim):
                if i not in set_unclamp:
                    ndarry[i] = False
            print(ndarry, "change to arrray")
            z = tf.compat.v1.where(mask2, tfa_bound, z)
            for i in range(a_dim):
                if ndarry[i] == True:
                    set_clamp_round.add(i)
                    for j in range(a_dim):
                        condxy[i][j] = True
                else:
                    for j in range(a_dim):
                        condxy[i][j] = False

            grad_z = tf.compat.v1.where(condxy, grad_operator, grad_z)
            temp_z = grad_z
            print(set_clamp_round, "IME here")
        ''''''
        # algorithm 21~25
        unclamp_num = unclamp_num-len(set_clamp_round)
        for i in range(a_dim):
            if i in set_clamp_round:
                C_unclamp = C_unclamp-z[i]
        set_unclamp = set_unclamp.difference(set_clamp_round)
        if len(set_clamp_round) == 0:
            phase = phase+1
       # if(first==True):
      #      sess=tf.Session()
     #       sess.run(tf.global_variables_initializer())
        first = False
    #    print(sess.run([y,tempmask]))
        print(z, "Z in this round")
        print(grad_z, "grad_z this round")

    # debug after optlayer
    final_sum = tf.reduce_sum(input_tensor=z)
    assert final_sum == 30
    mask = tf.greater(lower, z)
    mask2 = tf.greater(z, a_bound)
    proto_tensor = tf.make_tensor_proto(mask)
    ndarry = tf.make_ndarray(proto_tensor)
    proto_tensor = tf.make_tensor_proto(mask2)
    ndarry2 = tf.make_ndarray(proto_tensor)
    assert (ndarry == ndarry2).all() and (ndarry == False).all()

    z_shape = z.shape[0]
    print("z shape: ", z_shape)
    z_reshape = tf.reshape(z, (1, z_shape))
    print("z_reshape: ", z_reshape.shape)

    print(z)
    print(grad_z)
    print(z_reshape)
    return z_reshape, grad_z


###############################  training  ####################################
Rs = []
# 2*ZONE+1
# the first ZONE number is demand(i.e. how many bikes are taken away in this ZONE)
# the last ZONE number is the amount of resource on zone K (dS_) + time
s_dim = env.observation_space.shape[0]
print(s_dim)
# equal to get_observe function in env
a_dim = env.action_space.shape[0]
s = env.reset()
# print(a_dim,"YEEEEEEE")
# print(env.action_space.low,"low")
a_bound = env.action_space.high  # higher bound, which is set in the .txt file

ddpg = DDPG(a_dim, s_dim, a_bound, env, s)

var = 3  # control exploration

for ep in range(100):
    R = 0
    ld_pickup = 0
    ld_dropoff = 0
    revenue = 0
    scenario = None
    done = False
    s = env.reset()  # [0,0,0,0,8,7,8,8,0]
    print(s)
    # policy = read_supriyo_policy_results(env)
    while not done:
        # action = None
        action = ddpg.choose_action(s)
        print(action, "x")
        # OptLayer_function(action,a_dim,a_bound,env)

        # print(obs)
        # action = get_supriyo_policy_action(env, obs, policy)

        # action = None
        s_, r, done, info = env.step(action)
        ddpg.store_transition(s, action, r / 10, s_)
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()
        s = s_
        R += r
        ld_pickup += info["lost_demand_pickup"]
        ld_dropoff += info["lost_demand_dropoff"]
        revenue += info["revenue"]
        scenario = info["scenario"]

    print({
        'episode': ep,
        'reward': R,
        'Explore': var,
        'lost_demand_pickup': ld_pickup,
        "lost_demand_dropoff": ld_dropoff,
        "revenue": revenue,
        "scenario": scenario
    })
    Rs.append(R)

print('')
print('---------------------------')
print('Average reward per episode:', np.average(Rs))
