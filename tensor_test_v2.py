# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:51:07 2020

@author: stitch
"""
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

a = tf.constant([[1,2,3],[4,5,6]])
proto_tensor = tf.make_tensor_proto(a)  # convert `tensor a` to a proto tensor
print(tf.make_ndarray(proto_tensor))
"""




def Numpy_opt(action,a_dim,a_bound):
    #adjust to y
    maxa=action[np.argmax(action)]
    mina=action[np.argmin(action)]
    lower=np.zeros(a_dim)
    y=np.zeros(a_dim)
    
    print(a_bound,"abound")
    for i in range(a_dim):
        y[i] = lower[i]+(a_bound[i]-lower[i])*(action[i]-mina)/(maxa-mina)
    print(y,"y")
    #adjust to z
    z=np.zeros(a_dim)
   
    #start algorithm#
    phase=0                          #  lower=0 , upeer=1 , done=2
    C_unclamp=30             # how many left bike to distribute
    set_unclamp=set(range(a_dim))    # unclamp set
    unclamp_num=a_dim                # unclamp number=n'
    grad_z=np.zeros((a_dim,a_dim))   # grad_z is 4*4 arrray
    while phase != 2 :
        print("-----first loop----")
        sum_y=0
        set_clamp_round=set()  # indices clamped in this iteration of the while loop
        #algorithm line 7
        for i in range(a_dim):
            if i in set_unclamp:
                sum_y=sum_y+y[i]
        for i in range(a_dim):
            if i in set_unclamp:
                z[i]=y[i]+(C_unclamp-sum_y)/unclamp_num
        print(z,"z")
        print(sum_y,"sum_y")
        #algorithm line8
        for i in range(a_dim):
            if i in set_unclamp:
                for j in range(a_dim):
                    if j in set_unclamp:
                        if (i!=j):
                            grad_z[i][j]= -1/unclamp_num 
                        else :
                            grad_z[i][j]= 1- (1/unclamp_num)
        print(grad_z)
        #algorithm line 9
        for j in range(a_dim):
            if j not in set_unclamp :
                for i in range(a_dim):
                    grad_z[i][j]=0
        print(grad_z,"grad before clamp in this iteration")
        
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
        #print(grad_z,"grad after clamp")
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
        print(grad_z,"grad_z this round")
        
    #debug after optlayer
    final_sum=0
    print(z)
    print(grad_z)
    
def OptLayer_function(action, a_dim, a_bound):
        # adjust to y
    print(action,"action")
    
    maxa = tf.reduce_max(input_tensor=action)
    mina = tf.reduce_min(input_tensor=action)
    lower = tf.zeros(a_dim,dtype=tf.float64)
    tfa_bound = tf.convert_to_tensor(value=a_bound,dtype=tf.float64)
    y = tf.zeros(a_dim,tf.float64)
    y = lower+(tfa_bound-lower)*(action-mina)/(maxa-mina)    
    
    print(y,"y")
    # maxa=action[tf.math.argmax(action)]
    # mina=action[np.argmin(action)]
    # lower=np.zeros(a_dim)
    # y=np.zeros(a_dim)

    # adjust to z
    z = tf.zeros(a_dim,dtype=tf.float64)
    # start algorithm#
    phase = 0  # lower=0 , upeer=1 , done=2
    # how many left bike to distribute
    C_unclamp = tf.Variable(float(30),dtype=tf.float64)
    set_unclamp = set(range(a_dim))    # unclamp set
    unclamp_num = tf.Variable(float(a_dim),dtype=tf.float64)                # unclamp number=n'
    grad_z = tf.zeros([a_dim, a_dim], dtype=tf.float64)   # grad_z is 4*4 arrray
    first=True
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
        
        for i in range(a_dim):
            if i in set_unclamp:
                cond[i] = True
            else:
                cond[i] = False  # not calculate.
        case_sum_true=y
        case_sum_false=tf.zeros(a_dim,dtype=tf.float64)
        sum_y=tf.compat.v1.where(cond,case_sum_true,case_sum_false)
        sum_y=tf.reduce_sum(input_tensor=sum_y)
        print(sum_y)
        print(cond,"cond test")
        case_true = y+(C_unclamp-sum_y)/unclamp_num
        case_false = z
        z = tf.compat.v1.where(cond, case_true, case_false)
        condxy = np.zeros([a_dim, a_dim])
        # make sure the tensor shape the same to do tf.where
        grad_operator = tf.zeros([a_dim, a_dim],dtype=tf.float64)
        # algorithm line 8  3 phase to change 
        for i in range(a_dim): 
            for j in range(a_dim):
                if i not in set_unclamp:
                    condxy[i][j]=False
                elif j not in set_unclamp:
                    condxy[i][j]=False
                else :
                    condxy[i][j]=True
        case_grad_false=grad_z
        case_grad_true=grad_operator+1.0-(1.0/unclamp_num)
        grad_z=tf.compat.v1.where(condxy,case_grad_true,case_grad_false)
        
        for i in range(a_dim):
            if cond[i] == True:
                for j in range(a_dim):
                    if cond[j] == True and i==j:
                        condxy[i][j] = False
                    else:
                        condxy[i][j] = True
        case_grad_true = grad_operator-(1.0/unclamp_num)
        case_grad_false = grad_z
        grad_z = tf.compat.v1.where(condxy, case_grad_true, case_grad_false)
        
        # algorithm line 9
        for j in range(a_dim):
            if cond[j] == False:
                for i in range(a_dim):
                    condxy[i][j] = True
            else:
                for i in range(a_dim):
                    condxy[i][j] = False
        print(condxy,"BUFFFFF")
        case_grad_0_true = grad_operator
        case_grad_0_false = grad_z
        grad_z = tf.compat.v1.where(condxy, case_grad_0_true, case_grad_0_false)
        # algorithm lin 10~20
        if phase == 0:
            mask = tf.greater(lower, z)
            proto_tensor=tf.make_tensor_proto(mask)
            ndarry=tf.make_ndarray(proto_tensor)
            for i in range(a_dim):
                if i not in set_unclamp:
                    ndarry[i] = False
            z = tf.compat.v1.where(mask, lower, z)  # true,means i>z
            for i in range(a_dim):
                if ndarry[i] == True:
                    set_clamp_round.add(i)
                    for j in range(a_dim):
                        condxy[i][j] = True
                else:
                    for j in range(a_dim):
                        condxy[i][j] = False
            grad_z = tf.compat.v1.where(condxy, grad_operator, grad_z)
            temp_z=grad_z
        elif phase == 1:
            mask2 = tf.greater(z, tfa_bound)
            print(mask2,"maske_type")
            proto_tensor=tf.make_tensor_proto(mask2)
            ndarry=tf.make_ndarray(proto_tensor)
            
            for i in range(a_dim):
                if i not in set_unclamp:
                    ndarry[i] = False
            print(ndarry,"change to arrray")
            z = tf.compat.v1.where(mask2, tfa_bound, z)
            for i in range(a_dim):
                if ndarry[i] == True:
                    set_clamp_round.add(i)
                    for j in range(a_dim):
                        condxy[i][j] = True
                else:
                    for j in range(a_dim):
                        condxy[i][j] = False
            
            grad_z = tf.compat.v1.where(condxy, grad_operator, grad_z)
            temp_z=grad_z
            print(set_clamp_round,"IME here")
        ''''''
        # algorithm 21~25
        unclamp_num = unclamp_num-len(set_clamp_round)
        for i in range(a_dim):
            if i in set_clamp_round:
                C_unclamp = C_unclamp-z[i]
        set_unclamp = set_unclamp.difference(set_clamp_round)
        if len(set_clamp_round) == 0:
            phase = phase+1
       # if(first==True):
      #      sess=tf.Session()
     #       sess.run(tf.global_variables_initializer())
        first=False
    #    print(sess.run([y,tempmask]))
        print(z,"Z in this round")
        print(grad_z,"grad_z this round")

    # debug after optlayer
    final_sum =tf.reduce_sum(input_tensor=z)
    assert final_sum==30
    mask=tf.greater(lower, z)
    mask2=tf.greater(z,a_bound)
    proto_tensor=tf.make_tensor_proto(mask)
    ndarry=tf.make_ndarray(proto_tensor)
    proto_tensor=tf.make_tensor_proto(mask2)
    ndarry2=tf.make_ndarray(proto_tensor)
    assert (ndarry==ndarry2).all() and (ndarry==False).all()

    z_shape = z.shape[0]
    print("z shape: ", z_shape)
    z_reshape = tf.reshape(z, (1, z_shape))
    print("z_reshape: ", z_reshape.shape)
    
    print(z)
    print(grad_z)
    print(z_reshape)
    return z_reshape, grad_z

action=tf.Variable([20.,10.,6.,9.],dtype=tf.float64)
a_dim=4
a_bound=[11., 19. ,18., 12.]

OptLayer_function(action,a_dim,a_bound)
print("--------------------------------------------------------")
Numpy_opt([20,10,6,9],4,[11,19,18,12])
