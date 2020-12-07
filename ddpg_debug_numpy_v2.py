import numpy as np
import tensorflow as tf
import math

tf.compat.v1.disable_eager_execution()


# x = tf.Variable(initial_value=50., dtype='float32', trainable=True)
# w = tf.Variable(initial_value=10., dtype='float32', trainable=True)
# y = w * x
# z = y * y

x = tf.Variable([[50.]], dtype='float32', name='x', trainable=True)
w = tf.Variable([[10.]], dtype='float32', name='w', trainable=True)
# v = tf.Variable([[40.]], dtype='float32', name='v', trainable=True)

# y = w @ x
# z = 2 * y

print("x: ", x.get_shape())
print("w: ", w.get_shape())

w_init = tf.constant_initializer(np.ones((1, 1)))
# w_init = tf.constant_initializer(np.array([[3.]]))
b_init = tf.constant_initializer(np.array([[2.]]))

# '''
net_1 = tf.compat.v1.layers.dense(
    w @ x, 1, activation=None, name='l1', use_bias=False, kernel_initializer=w_init, trainable=True)
net_2 = tf.compat.v1.layers.dense(
    2*net_1, 1, activation=None, name='l2', use_bias=False, kernel_initializer=w_init, trainable=True)
# '''


opt = tf.compat.v1.train.GradientDescentOptimizer(0.1)
grad = opt.compute_gradients(net_2)
with tf.compat.v1.Session() as sess:
    writer = tf.compat.v1.summary.FileWriter("TensorBoard/", graph=sess.graph)

    sess.run(tf.compat.v1.global_variables_initializer())
    print("net_1: ", sess.run(net_1))
    print("net_2: ", sess.run(net_2))
    g = sess.run(grad)
    print(g)
    g = np.array(g)
    print(g.shape)
