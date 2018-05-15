import os.path
import sys

import gym
import numpy as np

import gym_BSS  # noqa: F401

name = sys.argv[1] if len(sys.argv) > 1 else 'BSSEnvTest-v0'
env = gym.make(name)  # gym.Env
env.seed(42)
# print(env.observation_space, env.action_space)
print(name)
print(env.metadata)


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
    f1 = open(os.path.join(env.data_dir, "Our_policy",
                           "policy_result{0}.csv".format(scenario)))
    line = f1.readline()
    line = f1.readline()
    while(line != ""):
        line = line.strip(" \n")
        line = line.split(",")
        if(int(line[0]) < 100):
            ypcmu[int(line[0]) + 1][int(line[1])] = float(line[2])
            yncmu[int(line[0]) + 1][int(line[1])] = float(line[3])
        line = f1.readline()
    f1.close()
    return (ypcmu, yncmu)


Rs = []
for ep in range(100):
    R = 0
    ld_pickup = 0
    ld_dropoff = 0
    revenue = 0
    scenario = None
    done = False
    obs = env.reset()
    # policy = read_supriyo_policy_results(env)
    while not done:
        action = None
        # print(obs)
        # action = get_supriyo_policy_action(env, obs, policy)
        action = None
        obs, r, done, info = env.step(action)
        R += r
        ld_pickup += info["lost_demand_pickup"]
        ld_dropoff += info["lost_demand_dropoff"]
        revenue += info["revenue"]
        scenario = info["scenario"]
    print({
        'episode': ep,
        'reward': R,
        'lost_demand_pickup': ld_pickup,
        "lost_demand_dropoff": ld_dropoff,
        "revenue": revenue,
        "scenario": scenario
    })
    Rs.append(R)

print('')
print('---------------------------')
print('Average reward per episode:', np.average(Rs))
