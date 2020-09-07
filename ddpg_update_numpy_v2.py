import os.path
import sys

import gym
import numpy as np
import tensorflow as tf
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
            ypcmu[int(line[0]) + 1][int(line[1])] = float(line[2])  #output
            yncmu[int(line[0]) + 1][int(line[1])] = float(line[3])  #input
        line = f1.readline()
    f1.close()
    return (ypcmu, yncmu)

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
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
        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001)
        a_loss = - tf.reduce_mean(input_tensor=q)    # maximize the q
        self.grads_and_vars = optimizer.compute_gradients(a_loss,self.ae_params)
       # self.optgrad=tf.zeros([a_dim, a_dim])
       # self.gv_opt_fn=[(gv[0]*self.optgrad,gv[1])for gv in self.grads_and_vars]
        self.opttrain=optimizer.apply_gradients(self.grads_and_vars)
        
        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):
        #print(s.shape,"QQQQQQQQQQQQQQQQQQQQQWWWWWW")   #9 
        #print(s[np.newaxis, :].shape) #(1,9)
        #print(self.sess.run(self.a, {self.S: s[np.newaxis, :]}))
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

        a,g=self.sess.run([self.atrain,self.grads_and_vars], {self.S: bs})
        print(g[0][0])   #one more layer 9,2>>11,2  but add (a_loss,self.ae_params) become 4,2
        print(g[0][1])
        print("Q")
        print(g[1][0])
        print("Q")
        print(g[1][1])
        print("QQ")
        print(g[2][0])
        print(g[2][1])
        print(g[3][0])
        print(g[3][1])
        print(g.gg)  #to terminal
        
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.compat.v1.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            #print(a)
            #print(tf.multiply(a, self.a_bound, name='scaled_a'),"QQQ")
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
        
################ Opt layer#####################
            
def OptLayer_function(action,a_dim,a_bound,env):
    #adjust to y
    maxa=action[np.argmax(action)]
    mina=action[np.argmin(action)]
    lower=np.zeros(a_dim)
    y=np.zeros(a_dim)
    #print(env.nbikes,"bike_num")
    
    #print(a_bound,"abound")
    for i in range(a_dim):
        y[i] = lower[i]+(a_bound[i]-lower[i])*(action[i]-mina)/(maxa-mina)
    #print(y,"y")
    #adjust to z
    z=np.zeros(a_dim)
   
    #start algorithm#
    phase=0                          #  lower=0 , upeer=1 , done=2
    C_unclamp=env.nbikes             # how many left bike to distribute
    set_unclamp=set(range(a_dim))    # unclamp set
    unclamp_num=a_dim                # unclamp number=n'
    grad_z=np.zeros((a_dim,a_dim))   # grad_z is 4*4 arrray
    while phase != 2 :
        sum_y=0
        set_clamp_round=set()  # indices clamped in this iteration of the while loop
        #algorithm line 7
        for i in range(a_dim):
            if i in set_unclamp:
                sum_y=sum_y+y[i]
        for i in range(a_dim):
            if i in set_unclamp:
                z[i]=y[i]+(C_unclamp-sum_y)/unclamp_num
        #print(z,"z")
        #print(sum_y,"sum_y")
        #algorithm line8
        for i in range(a_dim):
            if i in set_unclamp:
                for j in range(a_dim):
                    if j in set_unclamp:
                        if (i!=j):
                            grad_z[i][j]= -1/unclamp_num 
                        else :
                            grad_z[i][j]= 1- (1/unclamp_num)
       # print(grad_z)
        #algorithm line 9
        for j in range(a_dim):
            if j not in set_unclamp :
                for i in range(a_dim):
                    grad_z[i][j]=0
      # print(grad_z,"grad before clamp in this iteration")
        
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
       # print(z,"z_after clamp")
       #print(grad_z,"grad after clamp")
        #algorithm 21~25
        unclamp_num=unclamp_num-len(set_clamp_round)
     #   print(unclamp_num,"unclamp")
        for i in range(a_dim):
            if i in set_clamp_round:
                C_unclamp=C_unclamp-z[i]
       # print(C_unclamp,"C")
        set_unclamp= set_unclamp.difference(set_clamp_round)
      #  print(set_unclamp,"unclamp set")
        if len(set_clamp_round)==0 :
            phase=phase+1
        
    #debug after optlayer
    final_sum=0
    for i in range(a_dim):           
        final_sum=final_sum+z[i]
        assert lower[i]<=z[i]<=a_bound[i]   # make sure not violate the local constraint 
    final_sum=round(final_sum,2)
   # print(final_sum)
    assert final_sum==env.nbikes     # make sure sum is equal to bike number
    if np.sum(y)==env.nbikes:
        assert z==y
    return z ,grad_z

###############################  training  ####################################   
Rs = []
s_dim = env.observation_space.shape[0]  # 2*ZONE+1 ZONE's Demand,zone's number of resource on zone K (dS_) +time
#equal to get_observe function in env
a_dim = env.action_space.shape[0]        #4

#print(a_dim,"YEEEEEEE")
#print(env.action_space.low,"low")
a_bound = env.action_space.high   #bound , in txt file

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration

for ep in range(100000):
    R = 0
    ld_pickup = 0
    ld_dropoff = 0
    revenue = 0
    scenario = None
    done = False
    s = env.reset()  # [0,0,0,0,8,7,8,8,0]
    #print(s)
    #policy = read_supriyo_policy_results(env)
    while not done:
        #action = None
        action=ddpg.choose_action(s)
       # print(action,"x")
        action,grad=OptLayer_function(action,a_dim,a_bound,env)
        #print(action,"After_modify")
        # print(obs)
        #action = get_supriyo_policy_action(env, obs, policy)
        
        #action = None
        s_, r, done, info = env.step(action)
        print(done)
        print(ddpg.pointer)
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
