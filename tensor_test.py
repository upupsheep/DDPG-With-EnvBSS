# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:51:07 2020

@author: stitch
"""
import tensorflow as tf
import numpy as np

"""
a = tf.constant([[11,0,13,14],
                 [21,22,23,0]])
condition = tf.equal(a, 11)

case_true = tf.reshape(tf.multiply(tf.ones([8], tf.int32), -9999), [2, 4])
case_false = a
a = tf.where(condition, case_true, case_false)
sess = tf.Session()
print(sess.run(condition))
print(sess.run(a))

x=[1,2,3,4,5,6,7,8,9,10]
set={1,5,7}
y=[0,0,0,0,0,0,0,0,0,0]

for i in range(10):
    if i in set:
        y[i]=x[i]
        
print(y)

cond=[0,0,0,0]
t=tf.constant([1,2,3,4])
y=tf.constant([2,3,4,5])
sets={1,3}
print(sess.run(t))
for i in range(4):
    if i in sets:
        cond[i]=True
    else:
        cond[i]=False
case_true=t+y
case_false=t
t=tf.where(cond,case_true,case_false) #1,5,3,9
print(sess.run(t))
'''
condxy=[]
z=tf.zeros([4,4])
condx=[True,False,False,True]
condy=[False,True,True,True]
for i in range(4):
    for j in range(4):
        if(condx[i]==True and condy[j]==True):
            condxy.append(True)
        else:
            condxy.append(False)
condxy=np.reshape(condxy,(4,4))
print(condxy)
case_1=z+10
case_2=z
z=tf.where(condxy,case_1,case_2)
print(sess.run(z))
'''

condxy=np.zeros([4,4])
z=tf.zeros([4,4])
print(sess.run(z))
condx=[True,False,False,True]
condy=[False,True,True,True]
for i in range(4):
    for j in range(4):
        if(condx[i]==True and condy[j]==True):
            condxy[i][j]=True
        else:
            condxy[i][j]=False
print(condxy)
case_1=z+tf.constant(10.)
case_2=z
z=tf.where(condxy,case_1,case_2)
print(sess.run(z))

'''gather test'''
g=tf.constant([1,2,3,4])
print(sess.run(tf.gather(g,2)))


import tensorflow as tf
x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sets = tf.constant([1, 5, 7])
y = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y2 = tf.tensor_scatter_nd_update(y, tf.expand_dims(sets, 1), tf.gather(x, sets))
print(sess.run(y2))
# [0 2 0 0 0 6 0 8 0 0]

#compare test
acompare = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bcompare= tf.constant([2,1,5,6,7,4,2,7,8,1])

mask = tf.greater(acompare, bcompare)
slices = tf.boolean_mask(acompare, mask)
print(sess.run(slices))

#share the dense layer , using reuse
import tensorflow as tf
x1 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="x1")
x2 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="x2")
print(x1)
with tf.variable_scope("myscope") as scope:
	l1 = tf.layers.Dense(units=3)
	h11 = l1(x1)
with tf.variable_scope("myscope",reuse=True) as scope:
	l2 = tf.layers.Dense(units=3)
	h12 = l2(x2)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([h11, h12], feed_dict={x1: [[1, 2, 3],[10,20,30]] ,x2: [[2, 4, 6],[5,10,15]]}))
"""
def OptLayer_function(action, a_dim, a_bound):
    # adjust to y
    with tf.variable_scope("Optlayer"):
        maxa = tf.reduce_max(action)
        mina = tf.reduce_min(action)
        lower = tf.zeros(a_dim)
        tfa_bound = tf.convert_to_tensor(a_bound)
        y = tf.zeros(a_dim)
        y = lower+(tfa_bound-lower)*(action-mina)/(maxa-mina)
    # maxa=action[tf.math.argmax(action)]
    # mina=action[np.argmin(action)]
    # lower=np.zeros(a_dim)
    # y=np.zeros(a_dim)

    # adjust to z
    z = tf.zeros(a_dim)
    # start algorithm#
    phase = 0  # lower=0 , upeer=1 , done=2
    # how many left bike to distribute
    C_unclamp = tf.Variable(float(30))
    set_unclamp = set(range(a_dim))    # unclamp set
    unclamp_num = tf.Variable(float(a_dim))                # unclamp number=n'
    grad_z = tf.zeros([a_dim, a_dim], tf.float64)   # grad_z is 4*4 arrray

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
        sum_y=tf.reduce_sum(y)
        for i in range(a_dim):
            if i in set_unclamp:
                cond[i] = True
            else:
                cond[i] = False  # not calculate.
        case_true = y+(C_unclamp-sum_y)/unclamp_num
        case_false = z
        z = tf.where(cond, case_true, case_false)
        condxy = np.zeros([a_dim, a_dim])
        # make sure the tensor shape the same to do tf.where
        grad_operator = tf.zeros([a_dim, a_dim])
        
        # algorithm line 8
        for i in range(a_dim):
            if cond[i] == True:
                for j in range(a_dim):
                    if cond[j] == True:
                        condxy[i][j] = True
                    else:
                        condxy[i][j] = False
        case_grad_true = grad_operator-(1.0/unclamp_num)
        case_grad_false = grad_operator+1.0-(1.0/unclamp_num)
        grad_z = tf.where(condxy, case_grad_true, case_grad_false)
        

        # algorithm line 9
        for j in range(a_dim):
            if cond[j] == False:
                for i in range(a_dim):
                    condxy[i][j] = True
            else:
                condxy[i][j] = False
        case_grad_0_true = grad_operator
        case_grad_0_false = grad_z
        grad_z = tf.where(cond, case_grad_0_true, case_grad_0_false)

        # algorithm lin 10~20
        if phase == 0:
            mask = tf.greater(lower, z)
            for i in range(a_dim):
                if i not in set_unclamp:
                    mask[i] = False
            z = tf.where(mask, lower, z)  # true,means i>z
            for i in range(a_dim):
                if mask[i] == True:
                    set_clamp_round.add(i)
                    for j in range(a_dim):
                        condxy[i][j] = True
                else:
                    for j in range(a_dim):
                        condxy[i][j] = False
            grad_z = tf.where(condxy, grad_operator, grad_z)
        elif phase == 1:
            mask2 = tf.greater(z, a_bound)
            for i in range(a_dim):
                if i not in set_unclamp:
                    mask2[i] = False
            z = tf.where(mask2, a_bound, z)
            for i in range(a_dim):
                if mask2[i] == True:
                    set_clamp_round.add(i)
                    for j in range(a_dim):
                        condxy[i][j] = True
                else:
                    for j in range(a_dim):
                        condxy[i][j] = False
            grad_z = tf.where(condxy, grad_operator, grad_z)

        '''modify above'''
        # algorithm 21~25
        unclamp_num = unclamp_num-len(set_clamp_round)
        for i in range(a_dim):
            if i in set_clamp_round:
                C_unclamp = C_unclamp-z[i]
        set_unclamp = set_unclamp.difference(set_clamp_round)
        if len(set_clamp_round) == 0:
            phase = phase+1

    # debug after optlayer
    final_sum = 0
    """
    for i in range(a_dim):
        final_sum=final_sum+z[i]
        # make sure not violate the local constraint
        assert lower[i]<=z[i]<=a_bound[i]
    assert final_sum==env.nbikes     # make sure sum is equal to bike number
    if np.sum(y)==env.nbikes:
        assert z==y
        """
    z_shape = z.shape[0]
    print("z shape: ", z_shape)
    z_reshape = tf.reshape(z, (1, z_shape))
    print("z_reshape: ", z_reshape.shape)
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
    return z_reshape, grad_z

action=tf.Variable([20.,10.,6.,9.],tf.float32)
a_dim=4
a_bound=[11., 19. ,18., 12.]
OptLayer_function(action,a_dim,a_bound)
