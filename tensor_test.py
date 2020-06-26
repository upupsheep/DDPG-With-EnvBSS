# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:51:07 2020

@author: stitch
"""
import tensorflow as tf
import numpy as np


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
