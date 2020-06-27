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
    #print(current_alloc)
    #print(sum(current_alloc))
    
    current_time = int(obs[-1])
    #print(current_alloc)
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
    #line = f1.readline()
    print(scenario)
    while(line != ""):
        line = line.strip(" \n")
        line = line.split(",")
        if(int(line[0]) < 100):
            ypcmu[int(line[0]) + 1][int(line[1])] = float(line[2])  #移出
            yncmu[int(line[0]) + 1][int(line[1])] = float(line[3])  #移入
        line = f1.readline()
    f1.close()
    return (ypcmu, yncmu)

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,env):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.env=env
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
            print(net,"net")
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            print(a,"a")
            a=  tf.multiply(a, self.a_bound, name='scaled_a')
            print(a,"a")
            a = OptLayer_function(a,self.a_dim,self.a_bound,self.env)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
        
################ Opt layer#####################
            
def OptLayer_function(action,a_dim,a_bound,env):
    #adjust to y
    print(type(action))
    print(action)
    with tf.variable_scope("Optlayer"):
        maxa=tf.Variable(tf.reduce_max(action),name='maxa')
        mina=tf.reduce_min(action)
        lower=tf.zeros(a_dim)
        tfa_bound=tf.convert_to_tensor(a_bound)
        y=tf.zeros(a_dim)
        y=lower+(tfa_bound-lower)*(action-mina)/(maxa-mina)
        print("maxa: ",maxa)
        print("mina: ",mina)
        print("lower: ",lower)
        print("tfa_bound: ",tfa_bound)
        print("y",y)
   # maxa=action[tf.math.argmax(action)]
   # mina=action[np.argmin(action)]
    #lower=np.zeros(a_dim)
    #y=np.zeros(a_dim)
    
    #adjust to z
    z=tf.zeros(a_dim)
    #start algorithm#
    phase=0                          #  lower=0 , upeer=1 , done=2
    C_unclamp=tf.Variable(float(env.nbikes))            # how many left bike to distribute
    set_unclamp=set(range(a_dim))    # unclamp set
    unclamp_num=tf.Variable(float(a_dim))                # unclamp number=n'
    grad_z=tf.zeros([a_dim,a_dim],tf.float64)   # grad_z is 4*4 arrray
    
    while phase != 2 :
        sum_y=tf.Variable(0.)
        cond=np.zeros(a_dim)
        set_clamp_round=set()  # indices clamped in this iteration of the while loop
        #algorithm line 7
        for i in range(a_dim):
            if i in set_unclamp:
                sum_y=sum_y+tf.gather(y,i) #need better way, can change to the tf.where method
        for i in range(a_dim):
            if i in set_unclamp:
                cond[i]=True
            else:
                cond[i]=False #not calculate.
        case_true=y[0]+(C_unclamp-sum_y)/unclamp_num   
        #print("y_QQQQQQQQQ",y[0])
        case_false=z
        z=tf.where(cond,case_true,case_false)
        print(z,"z")
        print(sum_y,"sum_y")
        condxy=np.zeros([a_dim,a_dim])
        grad_operator=tf.zeros([a_dim,a_dim])  #make sure the tensor shape the same to do tf.where
        print("grad_op",grad_operator)
        #algorithm line 8
        for i in range(a_dim):
            if cond[i]==True:
                for j in range(a_dim):
                    if cond[j]==True:
                        condxy[i][j]=True
                    else:
                        condxy[i][j]=False
        case_grad_true=grad_operator-(1.0/unclamp_num)
        case_grad_false=grad_operator+1.0-(1.0/unclamp_num)
        grad_z=tf.where(condxy,case_grad_true,case_grad_false)
        print(grad_z)
        
        #algorithm line 9
        for j in range(a_dim):
            if cond[j]==False:
                for i in range(a_dim):
                    condxy[i][j]=True
            else:
                condxy[i][j]=False
        case_grad_0_true=grad_operator
        case_grad_0_false=grad_z
        grad_z=tf.where(cond,case_grad_0_true,case_grad_0_false)
        print(grad_z,"grad before clamp in this iteration")
        
        '''modify above'''
        
        #algorithm lin 10~20
        for i in range(a_dim):
            if i in set_unclamp:
                if z[i]<lower[i] and phase==0 :
                    z[i]=lower[i]
                    for j in range(a_dim):
                        grad_z[i][j]=0
                    set_clamp_round.add(i)
                elif (z[i]>a_bound[i]) and phase==1:
                    z[i]=a_bound[i]
                    for j in range(a_dim):
                        grad_z[i][j]=0
                    set_clamp_round.add(i)
        print(z,"z_after clamp")
        print(grad_z,"grad after clamp")
        #algorithm 21~25
        unclamp_num=unclamp_num-len(set_clamp_round)
        print(unclamp_num,"unclamp")
        for i in range(a_dim):
            if i in set_clamp_round:
                C_unclamp=C_unclamp-z[i]
        print(C_unclamp,"C")
        set_unclamp= set_unclamp.difference(set_clamp_round)
        print(set_unclamp,"unclamp set")
        if len(set_clamp_round)==0 :
            phase=phase+1
        
    #debug after optlayer
    final_sum=0
    for i in range(a_dim):           
        final_sum=final_sum+z[i]
        assert lower[i]<=z[i]<=a_bound[i]   # make sure not violate the local constraint 
    assert final_sum==env.nbikes     # make sure sum is equal to bike number
    if np.sum(y)==env.nbikes:
        assert z==y
        
    return z

###############################  training  ####################################   
Rs = []
s_dim = env.observation_space.shape[0]  # 2*ZONE+1 , 前面ZONE個是Demand(這個ZONE被拿走幾台),後面ZONE個是number of resource on zone K (dS_) +time
#又等於get_observe function in env
a_dim = env.action_space.shape[0]
#print(a_dim,"YEEEEEEE")
#print(env.action_space.low,"low")
a_bound = env.action_space.high   #最大上限,txt裡面設定的

ddpg = DDPG(a_dim, s_dim, a_bound,env)

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
    #policy = read_supriyo_policy_results(env)
    while not done:
        #action = None
        action=ddpg.choose_action(s)
        print(action,"x")
        OptLayer_function(action,a_dim,a_bound,env)

        # print(obs)
        #action = get_supriyo_policy_action(env, obs, policy)
        
        #action = None
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
