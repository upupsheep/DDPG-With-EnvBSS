# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os.path

import scipy
import numpy
import gym
import mujoco_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #make render not lag
env_name='Reacher-v2'
env=gym.make(env_name)
s_dim = env.observation_space.shape[0] #11 np.cos(theta) 2,np.sin(theta) 2,qpos[2],qpos[3] ,qvel[0],qvel[1]
                                       # self.get_body_com("fingertip")-self.get_body_com("target") 3
a_dim = env.action_space.shape[0]               #2  
a_bound=env.action_space.high
ewma_r=0
arg_seed = 0
#########################seed##############################
tf.compat.v1.reset_default_graph()
random.seed(arg_seed)
np.random.seed(arg_seed)
env.seed(arg_seed)
env.action_space.np_random.seed(arg_seed)
#####################  hyper parameters  ####################
LR_C=0.001
LR_A=0.0001
GAMMA=0.99
TAU=0.001

MEMORY_CAPACITY=10000
BATCH_SIZE=64
eval_freq = 5000
####################testing part#################################
def evaluation(env_name,seed,ddpg,eval_episode=10):
    avgreward=0
    avg=[]
    eval_env=gym.make(env_name)
    eval_env.seed(seed+100)
    for eptest in range(eval_episode):
        running_reward =0
        done=False
        s=eval_env.reset()
        while not done:     
            action= ddpg.choose_action(s)
            s_,r,done,info=eval_env.step(action)
            s=s_
            running_reward=running_reward+r
        print('Episode {}\tReward: {} \t AvgReward'.format(eptest, running_reward))
        avgreward=avgreward+running_reward
        avg.append(running_reward)
    avgreward=avgreward/eval_episode
    print("------------------------------------------------")
    print("Evaluation average reward :",avgreward)
    print("------------------------------------------------")

    return avgreward/100
###############################  DDPG  ####################################
'''
env.reset()
for _ in range(100000):
    env.render()
    a=env.action_space.sample()   
    s,r,done,_=env.step(a)
    #print(s)

    '''
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1+1), dtype=np.float32)
        self.pointer = 0
        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        #self.sess = tf.compat.v1.Session()
        tf.random.set_seed(arg_seed)


        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')
        self.Done=tf.compat.v1.placeholder(tf.float32, [None, 1], 'done')

        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + (1-self.Done)*GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        
       
        a_loss = - tf.reduce_mean(input_tensor=self.q)    # maximize the q
        
             
        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):
       
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br=  bt[:,self.s_dim+self.a_dim:self.s_dim+self.a_dim+1]
        #br = bt[:, -self.s_dim - 1-1: -self.s_dim-1]
        bs_ = bt[:,self.s_dim+self.a_dim+1:self.s_dim+self.a_dim+1+self.s_dim]
        #bs_ = bt[:, -self.s_dim-1:-self.s_dim] 
        bd = bt[:,-1:]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.Done:bd})

    def store_transition(self, s, a, r, s_,done):
        transition = np.hstack((s, a, [r], s_,done))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
           # REGULARIZER = tf.keras.regularizers.l2(0.1)

            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.relu, name='l1', trainable=trainable
                                            )
            net2 = tf.compat.v1.layers.dense(net,300, activation=tf.nn.relu, name='l2', trainable=trainable
                                             )
            a = tf.compat.v1.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable
                                          )
            return tf.multiply(a, self.a_bound, name='scaled_a')
    '''
    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2= tf.compat.v1.layers.dense(net,300,activation=tf.nn.relu, name='cl2', trainable=trainable)
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)
        '''
    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.relu, name='cl1', trainable=trainable)
            #net2 = tf.compat.v1.layers.dense(tf.concat[net,a], 300, activation=tf.nn.relu, name='cl1', trainable=trainable)
            w2_net = tf.compat.v1.get_variable('w2_net', [400, 300], trainable=trainable)
            w2_a = tf.compat.v1.get_variable('w2_a', [self.a_dim, 300], trainable=trainable)
            b2= tf.compat.v1.get_variable('b1', [1, 300], trainable=trainable)
            net2= tf.nn.relu(tf.matmul(a,w2_a)+tf.matmul(net,w2_net)+b2)
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)
        
ddpg = DDPG(a_dim,s_dim,a_bound)    


Net_action=np.zeros((100000,a_dim+2))   
ewma = []
eva_reward=[]
store_action=[]
reward=[]
i=0   
for ep in range(100000000):
    #env.render()

    R=0
    done=False
    s=env.reset()
    
    while not done:
        #env.render()
        '''
        if np.random.random() <= exploration:
            action = env.action_space.sample()
        else:    
            action = ddpg.choose_action(s)
            '''
        if ddpg.pointer<1000:
            action=env.action_space.sample()
        else :
            action=ddpg.choose_action(s)+np.random.normal(0,0.1,a_dim)
            '''
            Net_action[i][0:2]=action 
            if -0.5<=sum(action)<=0.5 and -1<=action[0]<=1 and -1<=action[1]<=1:
                Net_action[i][2]=0
                #print(action)
            else:
                Net_action[i][2]=-1
            Net_action[i][3] = ep
            i=i+1
            '''
        store_action.append(action)
        #print(sum(action))
        s_,r,done,info=env.step(action)
        ddpg.store_transition(s,action,r,s_,done)
        if ddpg.pointer>MEMORY_CAPACITY:
            ddpg.learn()
        if (ddpg.pointer+1)% eval_freq==0:
            eva_reward.append(evaluation(env_name,arg_seed,ddpg))
        s= s_
        R += r
    reward.append(R)
    ewma_r = 0.05 * R + (1 - 0.05) * ewma_r
    print({
        'episode': ep,
        'reward' :R,
        'ewma_reward' :ewma_r
    })
    ewma.append(ewma_r)
    if(ddpg.pointer>=1000000):
        print("done training")
        break
a=[]
for i in range(1000):
    a.append(i*1000)
plt.plot(a,ewma)
plt.title("ewma reward, lr=0.05 fix, final ewma={}".format(ewma[999]))  

#mask = np.isin(Net_action[:,2], -1)
#violate_index=np.where(mask)

np.save("Reacher_{}_DDPG_Reward".format(arg_seed),reward)
np.save("Reacher_{}_DDPG_Action".format(arg_seed),store_action)
np.save("Reacher_{}_DDPG_eval_reward".format(arg_seed),eva_reward)
   

'''
avgreward=0
for ep in range(100):
    R=0
    running_reward =0
    done=False
    s=env.reset()
    while not done:
        env.render()

        action= ddpg.choose_action(s)
        s_,r,done,info=env.step(action)
        s=s_
        running_reward=running_reward+r
    print('Episode {}\tReward: {} \t AvgReward'.format(ep, running_reward))
    avgreward=avgreward+running_reward

print(avgreward/100)  #-10.56
        
#saver = tf.compat.v1.train.Saver()
#save_path=saver.save(ddpg.sess,"/home/johnny/Desktop/DDPG_model/ddpgmodel.ckpt")
    '''
    
############debug######################    
'''  
while True:
    test=env.np_random.uniform(low=-.2, high=.2, size=2)
    if np.linalg.norm(test) < 0.2:
                break
#env.get_body_com("body1")[0]+0.1*np.cos(sum(env.sim.data.qpos.flat[:2]))
#env.get_body_com("body1")[1]+0.1*np.sin(sum(env.sim.data.qpos.flat[:2]))
'''