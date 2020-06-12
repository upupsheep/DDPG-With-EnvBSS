
"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time

import os.path
import sys

import gym_BSS


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
# ENV_NAME = 'Pendulum-v0'
ENV_NAME = 'BSSEnvTest-v0'

#####################  BSS data functions  ####################

def get_supriyo_policy_action(env, obs, policy):
    ypcmu, yncmu = policy
    env = env.unwrapped
    current_alloc = obs[env.nzones:2 * env.nzones]
    current_time = int(obs[-1])
    yp_t = np.array(ypcmu[current_time])
    yn_t = np.array(yncmu[current_time])
    return current_alloc + yn_t - yp_t


def read_supriyo_policy_results(env):
    env = env.unwrapped
    scenario = env._scenario
    ypcmu = [[0.0 for k in range(env.nzones)] for j in range(env.ntimesteps)]
    yncmu = [[0.0 for k in range(env.nzones)] for j in range(env.ntimesteps)]
    f1 = open(
        os.path.join(env.data_dir, "Our_policy",
                     "policy_result{0}.csv".format(scenario)))
    line = f1.readline()
    line = f1.readline()
    while (line != ""):
        line = line.strip(" \n")
        line = line.split(",")
        if (int(line[0]) < 100):
            ypcmu[int(line[0]) + 1][int(line[1])] = float(line[2])
            yncmu[int(line[0]) + 1][int(line[1])] = float(line[3])
        line = f1.readline()
    f1.close()
    return (ypcmu, yncmu)

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

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
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(42)
# print(env.observation_space, env.action_space)
print('ENV_NAME: ', ENV_NAME)
print(env.metadata)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    # policy = read_supriyo_policy_results(env)
    ep_reward = 0
    ### info ###
    ld_pickup = 0
    ld_dropoff = 0
    revenue = 0
    scenario = None
    #############
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = None
        a = ddpg.choose_action(s)
        print('action random: ', a)
        # a = get_supriyo_policy_action(env, s, policy)        
        # a = np.clip(np.random.normal(a, var), 0.0, max(a_bound))    # add randomness to action selection for exploration
        y = np.zeros(a.shape)
        for i in range(a.shape[0]):
            upper_bound = a_bound[i]
            lower_bound = 0.0
            y[i] = lower_bound + (upper_bound - lower_bound) * (a[i] - np.min(a)) / (np.max(a) - np.min(a))
        print('y: ', y)

        z = np.zeros(y.shape)
        for i in range(y.shape[0]):
            z[i] = y[i] + (env.nbikes - np.sum(y)) / (y.shape[0])
        print('z: ', z)
        
        # for i in range(len(a)):
        #     a[i] = np.clip(a[i], 0.0, a_bound[i])
        # print('a_bound: ', a_bound)
        # print('action clipped: ', a)
        s_, r, done, info = env.step(z)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r

        ### info ###
        ld_pickup += info["lost_demand_pickup"]
        ld_dropoff += info["lost_demand_dropoff"]
        revenue += info["revenue"]
        scenario = info["scenario"]
        ##############

        if j == MAX_EP_STEPS-1:
            print({
                'Episode': i, 
                'Reward': int(ep_reward), 
                'Explore': var, 
                'lost_demand_pickup': ld_pickup,
                "lost_demand_dropoff": ld_dropoff,
                "revenue": revenue,
                "scenario": scenario
            })
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)