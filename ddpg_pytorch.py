'''
torch = 0.41
'''
from cvxpylayers.torch import CvxpyLayer
import torch
import cvxpy as cp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import gym_BSS  # noqa: F401


#####################  hyper parameters  ####################

MAX_EPISODES = 1000  # 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
c = 0.1  # 0.1
BATCH_SIZE = 64  # 32
TAU = 0.01
RENDER = False
# ENV_NAME = 'Pendulum-v0'
ENV_NAME = 'BSSEnvTest-v0'

###############################  DDPG  ####################################


class OptLayer(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(OptLayer, self).__init__()
        self.W = torch.nn.Parameter(1e-3*torch.randn(D_out, D_in))
        self.b = torch.nn.Parameter(1e-3*torch.randn(D_out))

        u = torch.as_tensor(a_bound)

        y = cp.Variable(D_out)
        Wtilde = cp.Variable((D_out, D_in))
        W = cp.Parameter((D_out, D_in))
        b = cp.Parameter(D_out)
        x = cp.Parameter(D_in)
        obj = cp.Minimize(cp.sum_squares(Wtilde @ x - b - y))
        cons = [cp.sum(y) == 90., 0 <= y, y <= u, Wtilde == W]
        prob = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(prob, [W, b, x], [y])

    def forward(self, x):
        # when x is batched, repeat W and b
        if x.ndim == 2:
            batch_size = x.shape[0]
            return self.layer(self.W.repeat(batch_size, 1, 1), self.b.repeat(batch_size, 1), x)[0]
        else:
            return self.layer(self.W, self.b, x)[0]


class ANet(nn.Module):   # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        # actions_value = x*2
        # print('x: ', x)
        # print('a_bound: ', a_bound)
        # actions_value = x * a_bound
        actions_value = x * 35
        # print('actions_value: ', actions_value)
        opt_action = OptLayer(a_dim, a_dim)(x)
        # print('opt_action: ', opt_action)
        return opt_action


class CNet(nn.Module):   # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach()  # ae（s）

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x +
                 '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x +
                 '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        # self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct
        record_range = min(self.pointer, MEMORY_CAPACITY)
        indices = np.random.choice(record_range, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        q = self.Critic_eval(bs, a)
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        # print(q)
        # print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        a_ = self.Actor_target(bs_)
        # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_ = self.Critic_target(bs_, a_)
        q_target = br+GAMMA*q_  # q_target = 负的
        # print(q_target)
        q_v = self.Critic_eval(bs, ba)
        # print(q_v)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1


###############################  training  ####################################
Rs = []
ewma_reward = 0  # EWMA reward for tracking the learning progress
ewma_reward_s = []

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    done = False
    j = 0
    # for j in range(MAX_EP_STEPS):
    while not done:
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        # print('In main: ', a)
        # add randomness to action selection for exploration
        # a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > c*MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            # print('learn!!!!')
            ddpg.learn()

        s = s_
        ep_reward += r
        '''
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' %
                  int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:
                RENDER = True
            break
        j += 1
        '''
    ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward

    print({
        'episode': i,
        'ewma reward': ewma_reward,
        # 'ep reward': R,
        'Explore': var
    })
    Rs.append(ep_reward)
    ewma_reward_s.append(ewma_reward)

Rs = np.array(Rs)
ewma_reward_s = np.array(ewma_reward_s)

print('')
print('---------------------------')
print('Average reward per episode:', np.average(Rs))

"""
Save rewards to file
"""
# np.save('ewma_reward', ewma_reward_s)
# np.save('ep_reward', Rs)

xAxis = np.arange(MAX_EPISODES)
yAxis = ewma_reward_s

plt.plot(xAxis, yAxis)
plt.title('Memory: {}, Batch size: {}, Episode: {}'.format(
    MEMORY_CAPACITY, BATCH_SIZE, MAX_EPISODES))
plt.xlabel('Episode')
plt.ylabel('EWMA Reward')
plt.show()

# print('Running time: ', time.time() - t1)
