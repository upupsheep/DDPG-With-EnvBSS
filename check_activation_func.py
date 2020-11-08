from __future__ import absolute_import, division, print_function
import os.path
import sys
import math

import gym
import numpy as np
import tensorflow as tf
import gym_BSS  # noqa: F401

tf.compat.v1.disable_eager_execution()

### Writing activation function ###


def clipping_acti_func(x):
    upper = np.array([3, 1, 5, 5], dtype=np.float32)
    lower = np.zeros(4, dtype=np.float32)

    y = np.zeros(4)
    print("bound: ", upper)
    print("acti input x: ", x)
    # print(((x[0] - np.min(x)) / (np.max(x) - np.min(x))))
    for i in range(4):
        if x[i] <= upper[i] and x[i] >= lower[i]:
            y[i] = x[i]
        else:
            y[i] = lower[i] + (upper[i] - lower[i]) * \
                ((x[i] - np.min(x)) / (np.max(x) - np.min(x)))
    print("acti output y:", y)
    return y


np_clipping_acti_func = np.vectorize(
    clipping_acti_func)  # vectorize the python function

#############################################################################

### Gradient of Activation ###


def d_clipping_acti_func(x):
    upper = np.array([3, 1, 5, 5], dtype=np.float32)
    lower = np.zeros(4, dtype=np.float32)

    y = np.zeros(4)
    print("bound: ", upper)
    print("acti input x: ", x)
    # print(((x[0] - np.min(x)) / (np.max(x) - np.min(x))))
    for i in range(4):
        if x[i] <= upper[i] and x[i] >= lower[i]:
            y[i] = x[i]
        else:
            y[i] = lower[i] + (upper[i] - lower[i]) * \
                ((x[i] - np.min(x)) / (np.max(x) - np.min(x)))
    print("acti output y:", y)
    return y


np_d_clipping_acti_func = np.vectorize(d_clipping_acti_func)


def np_d_clipping_acti_func_32(x):
    return np_d_clipping_acti_func(x).astype(np.float32)


def tf_d_clipping_acti_func(x, name=None):
    with tf.compat.v1.name_scope(name, "d_clipping_acti_func", [x]) as name:
        y = py_func(np_d_clipping_acti_func_32,  # forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    stateful=False)

        # when using with the code, it is used to specify the rank of the input.
        y[0].set_shape(x.get_shape())
        return y[0]


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.compat.v1.py_func(func, inp, Tout, stateful=stateful, name=name)

#############################################################################

### Gradient Function ###


def clipping_grad(op, grad):
    # print("--------In clipping_grad-----------\n")
    # print("op: ", op.inputs[0])
    x = op.inputs[0]
    n_gr = tf_d_clipping_acti_func(x)  # defining the gradient
    # print("grad: ", grad * n_gr)
    # print("---------------------------------\n")
    return grad * n_gr

#############################################################################

### Combinaing it all together ###


def np_clipping_acti_func_32(x):
    return np_clipping_acti_func(x).astype(np.float32)


def tf_clipping_acti_func(x, name=None):
    with tf.compat.v1.name_scope(name, "clipping_acti_func", [x]) as name:
        y = py_func(np_clipping_acti_func_32,  # forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    grad=clipping_grad)  # <-- here's the call to the gradient

        # when using with the code, it is used to specify the rank of the input.
        y[0].set_shape(x.get_shape())
        return y[0]


if __name__ == "__main__":
    x = tf.compat.v1.placeholder(tf.float32, shape=(4))
    # x = tf.constant([1., 2., 4., 8.], dtype=tf.float32)
    y = tf_clipping_acti_func(x)
    # tf.compat.v1.initialize_all_variables().run()
    # y = -3 * x
    var_grad = tf.gradients(y, x)
    with tf.compat.v1.Session() as sess:
        #     print("x: ", x)
        var_grad_val = sess.run(var_grad, feed_dict={x: [1, 2, 4, 8]})
    print("grad: ", var_grad_val)
    # print(var_grad[0].eval())
