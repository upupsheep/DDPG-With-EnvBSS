from __future__ import absolute_import, division, print_function
import os.path
import sys
import math

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from cvxpylayers.tensorflow import CvxpyLayer
import cvxpy as cp
import gym_BSS  # noqa: F401

tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())
# tf.config.run_functions_eagerly(True)

tf.keras.backend.set_floatx('float64')

# problem = "Pendulum-v0"
problem = sys.argv[1] if len(sys.argv) > 1 else 'BSSEnvTest-v0'
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high
lower_bound = np.zeros(num_actions)
# lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
# print("Min Value of Action ->  {}".format(lower_bound))

print(env.metadata)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) *
            np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def clipping(action):
    # print("\n--- In clipping activation function ---")
    # print("a_bound: ", a_bound)
    # print("action mtx: ", action)
    # print("x: ", type(action_mtx))
    # if type(action_mtx) is tuple:
    #     # [[xx, xx, xx, xx]], and scaled_a here
    #     action = action_mtx[0] * a_bound
    # else:
    #     action = action_mtx * a_bound
    # print("scaled x: ", action)
    # action = action_mtx[0] * a_bound
    # adjust to y
    maxa = action[int(np.argmax(action))]
    mina = action[int(np.argmin(action))]
    lower = np.zeros(num_actions)
    y = np.zeros(num_actions)

    # Avoid [nan nan nan nan]
    # '''
    is_nan = []
    for a in action:
        is_nan.append(math.isnan(a))
    # print("is nan: ", is_nan)
    if np.all(is_nan):
        return np.array(upper_bound)
    # '''

    # print(env.nbikes, "bike_num")
    # print(a_bound, "abound")
    for i in range(num_actions):
        # if x[k] is in the bound, then no need to clip
        if action[i] <= upper_bound[i] and action[i] >= lower[i]:
            y[i] = action[i]
        else:
            y[i] = lower[i]+(upper_bound[i]-lower[i]) * \
                (action[i]-mina)/(maxa-mina)
    # print("y: ", y)
    # print("------------------\n")

    # mu = float(LAMBDA) * float(np.abs(1 - np.sum(y)) +
    #                            np.abs(env.nbikes - np.sum(y)))

    return y


class OptLayer(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(OptLayer, self).__init__()
        self.W = tf.Variable(1e-3 * tf.random.normal((output_dim, input_dim), dtype=tf.float64))
        self.b = tf.Variable(1e-3 * tf.random.normal((output_dim,),  dtype=tf.float64))
        
        # u = tf.fill(input_dim, 35.) # 35.
        u = tf.convert_to_tensor(upper_bound)

        y = cp.Variable(output_dim)
        Wtilde = cp.Variable((output_dim, input_dim))
        W = cp.Parameter((output_dim, input_dim))
        b = cp.Parameter(output_dim)
        x = cp.Parameter(input_dim)
        obj = cp.Minimize(cp.sum_squares(Wtilde @ x - b - y))
        cons = [cp.sum(y) == 90., 0 <= y, y <= u, Wtilde == W]
        problem = cp.Problem(obj, cons)
        self.cvxpy_layer = CvxpyLayer(problem, [W, b, x], [y])

    def call(self, x):
        # print('x: ', x)
        def set_weight(x):
            return self.W
        
        def set_bias(x):
            return self.b

        if tf.rank(x) == 2:
            # when x is batched, repeat W and b
            # print('x.shape: ', tf.shape(x))
            batch_size = x.shape[0]
            # batch_size = tf.shape(x)[0]
            # print('batch_size: ', batch_size)
            return self.cvxpy_layer(tf.stack([self.W for _ in tf.range(batch_size)]), tf.stack([self.b for _ in tf.range(batch_size)]), x)[0]
            # tf.compat.v1.enable_eager_execution()
            # print(tf.executing_eagerly())
            # return self.cvxpy_layer(tf.map_fn(fn=lambda inp: set_weight(inp), elems=x), tf.map_fn(fn=lambda inp: set_bias(inp), elems=x), x)[0]
        else:
            return self.cvxpy_layer(self.W, self.b, x)[0]


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    # @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        # print('UPDATE!!!')
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model(
                [state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            # print('critic_loss: ', critic_loss)

        critic_grad = tape.gradient(
            critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            opt_layer = OptLayer(num_actions, num_actions)
            actions = opt_layer(actions)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float64)
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


###############################  DDPG  ####################################

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
# @tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    # '''
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(400, activation="relu", dtype=tf.float64)(inputs)
    out = layers.Dense(300, activation="relu", dtype=tf.float64)(out)
    # outputs = layers.Dense(num_actions, activation="tanh",
    #                        kernel_initializer=last_init)(out)
    outputs = layers.Dense(
        num_actions, activation="tanh", dtype=tf.float64)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    # print(outputs)

    # opt_output = OptLayer(num_actions, num_actions)(outputs)
    # opt_output = opt_layer(outputs)

    model = tf.keras.Model(inputs, outputs)
    # model = tf.keras.Model(inputs, opt_output)
    # '''
    '''
    model = tf.keras.Sequential(
        # layers.Dense(400, activation='relu', input_shape=(num_states,), dtype=tf.float32),
        # layers.Dense(300, activation='relu', dtype=tf.float32),
        # layers.Dense(num_actions, activation='tanh', dtype=tf.float32),
        OptLayer(num_actions, num_actions)
    )
    '''

    dot_img_file = './model_plot/model_1.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    return model


def get_critic():
    '''
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(32, activation="relu")(state_input)
    # state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])
    '''
    state_input = layers.Input(shape=(num_states))
    action_input = layers.Input(shape=(num_actions))
    concat = layers.Concatenate()([state_input, action_input])
    # '''

    out = layers.Dense(400, activation="relu", dtype=tf.float64)(concat)
    out = layers.Dense(300, activation="relu", dtype=tf.float64)(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    dot_img_file = './model_plot/model_2.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    # legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    # legal_action = clipping(sampled_actions)
    opt_layer = OptLayer(num_actions, num_actions)
    legal_action = opt_layer(sampled_actions)

    # return [np.squeeze(legal_action)]
    return np.squeeze(legal_action)


##############################################################
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(
    1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(10000, 64)
################################################################

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # print('action: ', action)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
