from __future__ import absolute_import, division, print_function
import os.path
import sys
import math

import gym
import numpy as np
import tensorflow as tf
import gym_BSS  # noqa: F401

tf.compat.v1.disable_eager_execution()


def clipping_acti_func(x):
    print("x: ", x)
    if x > 3:
        # print("here!")
        return 2*x
    elif x < 3:
        # print("there!")
        return -3*x
    else:
        # print("x=0")
        return 0


np_clipping_acti_func = np.vectorize(
    clipping_acti_func)  # vectorize the python function


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.compat.v1.py_func(func, inp, Tout, stateful=stateful, name=name)


def clipping_grad(op, grad):
    # print("--------In clipping_grad-----------\n")
    # print("op: ", op.inputs[0])
    x = op.inputs[0]
    n_gr = tf_clipping_acti_func(x)  # defining the gradient
    # print("grad: ", grad * n_gr)
    # print("---------------------------------\n")
    return grad * n_gr


def np_clipping_acti_func_32(x):
    return clipping_acti_func(x).astype(np.float32)


def tf_clipping_acti_func(x, name=None):
    with tf.compat.v1.name_scope(name, "clipping_acti_func", [x]) as name:
        y = py_func(np_clipping_acti_func_32,  # forward pass function
                    [x],
                    [tf.float32],
                    name=name,
                    grad=clipping_grad)  # the function that overrides gradient

        # when using with the code, it is used to specify the rank of the input.
        y[0].set_shape(x.get_shape())
        return y[0]


if __name__ == "__main__":
    x = tf.compat.v1.placeholder(tf.float32)
    y = tf_clipping_acti_func(x)
    # y = -3 * x
    var_grad = tf.gradients(y, x)
    with tf.compat.v1.Session() as sess:
        print("x: ", x)
        var_grad_val = sess.run(var_grad, feed_dict={x: 1})
    print("grad: ", var_grad_val)
