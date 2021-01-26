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
import random
import gym_BSS  # noqa: F401


#####################  hyper parameters  ####################

MAX_EPISODES = 2000  # 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01  # 0.01      # soft replacement
MEMORY_CAPACITY = 1000  # 10000
c = 1  # 0.1
BATCH_SIZE = 64  # 32
RENDER = False
random_seed = 0
# ENV_NAME = 'Pendulum-v0'
ENV_NAME = 'BSSEnvTest-v0'

eval_freq = 5000
#####################  global variables  ####################

env = gym.make(ENV_NAME)
env = env.unwrapped
# env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

#####################  random seed  ####################

torch.manual_seed(random_seed)
env.seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
env.action_space.seed(random_seed)

#################### testing part #################################
def evaluation(ddpg, eval_episode=10):
    avg_reward = 0
    avg = []
    eval_env = gym.make(ENV_NAME)
    eval_env.seed(random_seed + 100)
    for eptest in range(eval_episode):
        running_reward = 0
        done = False
        s = eval_env.reset()
        while not done:     
            action = ddpg.choose_action(s, None)
            s_, r, done, info = eval_env.step(action)
            s = s_
            running_reward = running_reward + r
        print('Episode {}\tReward: {} \t AvgReward'.format(eptest, running_reward))
        avg_reward = avg_reward + running_reward
        avg.append(running_reward)
    avg_reward = avg_reward / eval_episode
    print("------------------------------------------------")
    print("Evaluation average reward :",avg_reward)
    print("------------------------------------------------")

    return avg_reward / 100

###############################  DDPG  ####################################


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


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
        cons = [cp.sum(y) == env.nbikes, 0 <= y, y <= u, Wtilde == W]
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
        self.fc1 = nn.Linear(s_dim, 32)
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(32, 32)
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(32, a_dim)
        # self.out.weight.data.normal_(0, 0.1)  # initialization
        self.opt_layer = OptLayer(a_dim, a_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        # actions_value = x*2
        # print('x: ', x)
        # print('a_bound: ', a_bound)
        # actions_value = x * a_bound
        actions_value = x * 35
        # print('actions_value: ', actions_value)
        # opt_action = OptLayer(a_dim, a_dim)(x)
        opt_action = self.opt_layer(actions_value)
        # print('opt_action: ', opt_action)
        return opt_action


class CNet(nn.Module):   # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 32)
        # self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(a_dim, 32)
        # self.fca.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(32, 32)
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(32, 1)
        # self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        z = self.fc2(net)
        z = F.relu(z)
        actions_value = self.out(z)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        # self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim, a_dim)  # .type(torch.IntTensor)
        self.Actor_target = ANet(s_dim, a_dim)  # .type(torch.IntTensor)
        self.Actor_perturbed = ANet(s_dim, a_dim)  # .type(torch.IntTensor)

        self.Critic_eval = CNet(s_dim, a_dim)  # .type(torch.IntTensor)
        self.Critic_target = CNet(s_dim, a_dim)  # .type(torch.IntTensor)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s, para):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        # s = torch.unsqueeze(torch.IntTensor(s), 0)

        # self.Actor_eval.eval()
        # self.Actor_perturbed.eval()

        if para is None:
            return self.Actor_eval(s)[0].detach()  # ae（s）
        else:
            return self.Actor_perturbed(s)[0].detach()

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
        # '''
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])
        # '''
        '''
        bs = torch.IntTensor(bt[:, :self.s_dim])
        ba = torch.IntTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.IntTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.IntTensor(bt[:, -self.s_dim:])
        '''

        a = self.Actor_eval(bs)
        # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        q = self.Critic_eval(bs, a)
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        # print('q: ', q)
        # print('loss_a: ', loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()
        # print('atrain grad: ', self.atrain.grad)
        # for p in self.Actor_eval.parameters():
        #     print(p.name, p.requires_grad, p.grad.norm())

        # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        a_ = self.Actor_target(bs_)
        # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_ = self.Critic_target(bs_, a_)
        q_target = br+GAMMA*q_  # q_target = 负的
        # print('q_target: ', q_target)
        q_v = self.Critic_eval(bs, ba)
        # print('q_v: ', q_v)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # print('td_error: ', td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.Actor_perturbed, self.Actor_eval)
        params = self.Actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            random = torch.randn(param.shape)
            # if use_cuda:
            #     random = random.cuda()
            param += random * param_noise.current_stddev


###############################  training  ####################################
Rs = []
ewma_reward = 0  # EWMA reward for tracking the learning progress
ewma_reward_s = []

eva_reward = []
store_action = []

param_noise = AdaptiveParamNoiseSpec(
    initial_stddev=0.05, desired_action_stddev=0.3, adaptation_coefficient=1.05)
# param_noise = None

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = np.round(env.reset())
    old_s = s
    ep_reward = 0
    done = False
    j = 0
    noise_counter = 0
    if param_noise is not None:
        ddpg.perturb_actor_parameters(param_noise)
    # for j in range(MAX_EP_STEPS):
    while not done:
        if RENDER:
            env.render()

        # Add exploration noise
        a_float = ddpg.choose_action(s, param_noise)
        # Make it int and sum up to nbikes
        a = torch.round(a_float)
        diff = abs(torch.sum(a) - env.nbikes)
        if torch.sum(a) < env.nbikes:
            for a_idx in range(a_dim):
                if a[a_idx] + diff <= a_bound[a_idx]:
                    a[a_idx] += diff
                    break
        elif torch.sum(a) > env.nbikes:
            for a_idx in range(a_dim):
                if a[a_idx] - diff >= 0:
                    a[a_idx] -= diff
                    break
        # print('===========In main: ===============')
        # print('s = ', s)
        # print('old a = ', a_float)
        # print('a = ', a)
        # add randomness to action selection for exploration
        # a = np.clip(np.random.normal(a, var), -2, 2)
        # print('a: ', a)
        # print('store_action: ', store_action)
        store_action.append(a.numpy())

        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > c*MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            # print('learn!!!!')
            ddpg.learn()

        if (ddpg.pointer + 1) % eval_freq == 0:
            eva_reward.append(evaluation(ddpg))

        noise_counter += 1
        old_s = s
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

    # print('===========In main: ===============')
    # print('s = ', old_s)
    # # print('old a = ', a_float)
    # print('a = ', a)

    print({
        'episode': i,
        'ewma reward': ewma_reward,
        # 'ep reward': R,
        'Explore': var
    })
    Rs.append(ep_reward)
    ewma_reward_s.append(ewma_reward)
    np.save('bike_{}_memory{}_ewma'.format(random_seed, MEMORY_CAPACITY), np.array(ewma_reward_s))
    np.save('bike_{}_memory{}_ep_reward'.format(random_seed, MEMORY_CAPACITY), np.array(Rs))
    np.save('bike_{}_memory{}_eval_reward'.format(random_seed, MEMORY_CAPACITY), np.array(eva_reward))
    np.save('bike_{}_memory{}_action'.format(random_seed, MEMORY_CAPACITY), np.array(store_action))

Rs = np.array(Rs)
ewma_reward_s = np.array(ewma_reward_s)

print('')
print('---------------------------')
print('Average reward per episode:', np.average(Rs))

"""
Save model
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
