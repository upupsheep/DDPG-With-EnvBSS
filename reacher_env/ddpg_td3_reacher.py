import gym
import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################  hyper parameters  ####################

MAX_EPISODES = 10000 # 5000
MAX_EP_STEPS = 100
TOTAL_STEPS = 500000
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99    # reward discount
TAU = 0.001  # 0.01      # soft replacement
MEMORY_CAPACITY = 10000  # 10000
c = 0.1  # 0.1
BATCH_SIZE = 64  # 32
RENDER = False
random_seed = 1
# ENV_NAME = 'Pendulum-v0'
arg_env = 'Reacher-v2'

EVAL = True
eval_freq = 5000

SAVE_FILE = True

before_opt = []
after_opt = []
#############################################################

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    eval_action = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), None)
            eval_action.append(action)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    if SAVE_FILE:
        np.save('Reacher_seed{}_eval_action'.format(random_seed), np.array(eval_action))
    return avg_reward


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


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class OptLayer(torch.nn.Module):
    def __init__(self, D_in, D_out, a_bound):
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
        cons = [cp.sum(y) <= 0.1, cp.sum(y) >= -0.1, cp.sum_squares(y) <= 0.02, Wtilde == W]
        prob = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(prob, [W, b, x], [y])

    def forward(self, x):
        # when x is batched, repeat W and b
        if x.ndim == 2:
            batch_size = x.shape[0]
            return self.layer(self.W.repeat(batch_size, 1, 1), self.b.repeat(batch_size, 1), x)[0]
        else:
            return self.layer(self.W, self.b, x)[0]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.opt_layer = OptLayer(action_dim, action_dim, max_action)
        
        self.max_action = max_action

    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        scaled_a = self.max_action * torch.tanh(self.l3(a))
        # before_opt.append(scaled_a)
        opt_a = self.opt_layer(scaled_a)
        # after_opt.append(opt_a)
        return opt_a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)


    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_perturbed = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau
        
        # torch.cuda.empty_cache()


    def select_action(self, state, para):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if para is not None:
            return self.actor_perturbed(state).cpu().data.numpy().flatten()
        else:
            return self.actor(state).cpu().data.numpy().flatten()
        # torch.cuda.empty_cache()


    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            random = torch.randn(param.shape).to(device)
            # if use_cuda:
                # random = random.cuda()
            param += random * param_noise.current_stddev
            # torch.cuda.empty_cache()


    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # torch.cuda.empty_cache()

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


if __name__ == "__main__":
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # if args.save_model and not os.path.exists("./models"):
    #     os.makedirs("./models")

    env = gym.make(arg_env)

    # Set seeds
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": GAMMA,
        "tau": TAU,
    }

    # # Initialize policy
    # if args.policy == "TD3":
    # 	# Target policy smoothing is scaled wrt the action scale
    # 	kwargs["policy_noise"] = args.policy_noise * max_action
    # 	kwargs["noise_clip"] = args.noise_clip * max_action
    # 	kwargs["policy_freq"] = args.policy_freq
    # 	policy = TD3.TD3(**kwargs)
    # elif args.policy == "OurDDPG":
    # 	policy = OurDDPG.DDPG(**kwargs)
    # elif args.policy == "DDPG":
    # 	policy = DDPG.DDPG(**kwargs)
    policy = DDPG(**kwargs)

    # if args.load_model != "":
    # 	policy_file = file_name if args.load_model == "default" else args.load_model
    # 	policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, arg_env, random_seed)]

    state, done = env.reset(), False
    episode_reward = 0
    ewma_r = 0
    episode_timesteps = 0
    episode_num = 0

    store_ewma = []
    # eva_reward = []
    store_action = []
    store_reward = []

    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,desired_action_stddev=0.3, adaptation_coefficient=1.05)

    for t in range(int(MAX_EPISODES)):
        
        episode_timesteps += 1
        noise_counter = 0
        if param_noise is not None:
            policy.perturb_actor_parameters(param_noise)

        # # Select action randomly or according to policy
        action = policy.select_action(state, param_noise)

        store_action.append(action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        noise_counter += 1

        # Train agent after collecting sufficient data
        if t >= c*MEMORY_CAPACITY:
            policy.train(replay_buffer, BATCH_SIZE)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            ewma_r = 0.05 * episode_reward + (1 - 0.05) * ewma_r
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} EWMA: {ewma_r:.3f}")

            # save results
            store_reward.append(episode_reward)
            store_ewma.append(ewma_r)
            if SAVE_FILE:
                np.save('Reacher_seed{}_episode_reward'.format(random_seed), np.array(store_reward))
                np.save('Reacher_seed{}_ewma_reward'.format(random_seed), np.array(store_ewma))
                np.save('Reacher_seed{}_action'.format(random_seed), np.array(store_action))
                np.save('Reacher_seed{}_before_opt'.format(random_seed), before_opt)
                np.save('Reacher_seed{}_after_opt'.format(random_seed), after_opt)

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_policy(policy, arg_env, random_seed))
            if SAVE_FILE:
                np.save('Reacher_seed{}_eval_reward'.format(random_seed), np.array(evaluations))
            # np.save(f"./results/{file_name}", evaluations)
            # if args.save_model: policy.save(f"./models/{file_name}")
    if SAVE_FILE:
            np.save('Reacher_seed{}_episode_reward'.format(random_seed), np.array(store_reward))
            np.save('Reacher_seed{}_ewma_reward'.format(random_seed), np.array(store_ewma))
            np.save('Reacher_seed{}_action'.format(random_seed), np.array(store_action))
            np.save('Reacher_seed{}_before_opt'.format(random_seed), before_opt)
            np.save('Reacher_seed{}_after_opt'.format(random_seed), after_opt)