import numpy as np
import tensorflow as tf
import math

nbikes = 20

num_actions = 4
upper_bound = np.array([10, 10, 10, 10])


def OptLayer_op(y):
    # print("y: ", y[0])
    # y = y.numpy()
    # z = tf.zeros(num_actions)
    lower = np.zeros(num_actions)
    lower = tf.convert_to_tensor(lower, dtype=tf.float32)
    tf_upper_bound = tf.convert_to_tensor(upper_bound, dtype=tf.float32)

    z = [tf.Variable(0, dtype=tf.float32) for i in range(num_actions)]
    # z = tf.zeros(num_actions, dtype=tf.float32)

    ### start algorithm ###
    phase = 0  # lower=0, upper=1, done=2
    C_unclamp = nbikes  # how many left bike to distribute
    set_unclamp = set(range(num_actions))  # unclamp set
    unclamp_num = num_actions  # unclamp number=n'
    # grad_z = tf.zeros((num_actions, num_actions))   # grad_z is 4*4 arrray
    grad_z = np.zeros((num_actions, num_actions))

    while phase != 2:
        sum_y = tf.Variable(0, dtype=tf.float32)
        set_clamp_round = set()  # indices clamped in this iteration of the while loop
        # algorithm line 7
        for i in range(num_actions):
            if i in set_unclamp:
                # sum_y = sum_y+y[i]
                # print(y[0][i])
                y_i = tf.gather(y, i)
                print("y[i]: ", tf.keras.backend.get_value(y_i))
                sum_y.assign_add(y_i)
        for i in range(num_actions):
            if i in set_unclamp:
                z[i] = tf.gather(y, i)+(C_unclamp-sum_y)/unclamp_num
        # print(z,"z")
        # print(sum_y,"sum_y")
        # algorithm line8
        for i in range(num_actions):
            if i in set_unclamp:
                for j in range(num_actions):
                    if j in set_unclamp:
                        if (i != j):
                            grad_z[i][j] = -1/unclamp_num
                        else:
                            grad_z[i][j] = 1 - (1/unclamp_num)
       # print(grad_z)
        # algorithm line 9
        for j in range(num_actions):
            if j not in set_unclamp:
                for i in range(num_actions):
                    grad_z[i][j] = 0
      # print(grad_z,"grad before clamp in this iteration")

        # algorithm lin 10~20
        for i in range(num_actions):
            if i in set_unclamp:
                # if z[i] < lower[i] and phase == 0:
                z_i = z[i]
                # print("z_i: ", z_i)
                lower_i = tf.gather(lower, i)
                upper_i = tf.gather(tf_upper_bound, i)
                # print("lower_i: ", lower_i)
                # print(tf.math.less(z_i, lower_i).numpy())
                if (tf.math.less(z_i, lower_i)).numpy() and (phase == 0):
                    # z[i] = lower[i]
                    z[i] = lower_i
                    for j in range(num_actions):
                        grad_z[i][j] = 0
                    set_clamp_round.add(i)
                # elif (z[i] > upper_bound[i]) and phase == 1:
                elif (tf.math.greater(z_i, upper_i)).numpy() and (phase == 1):
                    # z[i] = upper_bound[i]
                    z[i] = upper_i
                    for j in range(num_actions):
                        grad_z[i][j] = 0
                    set_clamp_round.add(i)
       # print(z,"z_after clamp")
       # print(grad_z,"grad after clamp")
        # algorithm 21~25
        unclamp_num = unclamp_num-len(set_clamp_round)
     #   print(unclamp_num,"unclamp")
        for i in range(num_actions):
            if i in set_clamp_round:
                C_unclamp = C_unclamp-z[i]
       # print(C_unclamp,"C")
        set_unclamp = set_unclamp.difference(set_clamp_round)
      #  print(set_unclamp,"unclamp set")
        if len(set_clamp_round) == 0:
            phase = phase+1

    print("z: ", z)
    # debug after optlayer
    final_sum = 0
    for i in range(num_actions):
        z_i = z[i].numpy()
        # final_sum = final_sum+z[i]
        final_sum = final_sum + z_i
        # make sure not violate the local constraint
        # assert lower[i] <= z_i <= upper_bound[i]
        assert tf.gather(lower, i).numpy() <= z_i <= upper_bound[i]
    final_sum = round(final_sum, 2)
   # print(final_sum)
    assert final_sum == nbikes     # make sure sum is equal to bike number
    # if np.sum(y) == nbikes:
    #     assert z == y
    return tf.stack(z), grad_z


def OptLayer_function(y):
    lower = np.zeros(num_actions)
    z = np.zeros(num_actions)

    #start algorithm#
    phase = 0  # lower=0 , upeer=1 , done=2
    C_unclamp = nbikes             # how many left bike to distribute
    set_unclamp = set(range(num_actions))    # unclamp set
    unclamp_num = num_actions                # unclamp number=n'
    grad_z = np.zeros((num_actions, num_actions))   # grad_z is 4*4 arrray
    while phase != 2:
        sum_y = 0
        set_clamp_round = set()  # indices clamped in this iteration of the while loop
        # algorithm line 7
        for i in range(num_actions):
            if i in set_unclamp:
                sum_y = sum_y+y[i]
        for i in range(num_actions):
            if i in set_unclamp:
                z[i] = y[i]+(C_unclamp-sum_y)/unclamp_num
        # print(z,"z")
        # print(sum_y,"sum_y")
        # algorithm line8
        for i in range(num_actions):
            if i in set_unclamp:
                for j in range(num_actions):
                    if j in set_unclamp:
                        if (i != j):
                            grad_z[i][j] = -1/unclamp_num
                        else:
                            grad_z[i][j] = 1 - (1/unclamp_num)
       # print(grad_z)
        # algorithm line 9
        for j in range(num_actions):
            if j not in set_unclamp:
                for i in range(num_actions):
                    grad_z[i][j] = 0
      # print(grad_z,"grad before clamp in this iteration")

        # algorithm lin 10~20
        for i in range(num_actions):
            if i in set_unclamp:
                if z[i] < lower[i] and phase == 0:
                    z[i] = lower[i]
                    for j in range(num_actions):
                        grad_z[i][j] = 0
                    set_clamp_round.add(i)
                elif (z[i] > upper_bound[i]) and phase == 1:
                    z[i] = upper_bound[i]
                    for j in range(num_actions):
                        grad_z[i][j] = 0
                    set_clamp_round.add(i)
       # print(z,"z_after clamp")
       # print(grad_z,"grad after clamp")
        # algorithm 21~25
        unclamp_num = unclamp_num-len(set_clamp_round)
     #   print(unclamp_num,"unclamp")
        for i in range(num_actions):
            if i in set_clamp_round:
                C_unclamp = C_unclamp-z[i]
       # print(C_unclamp,"C")
        set_unclamp = set_unclamp.difference(set_clamp_round)
      #  print(set_unclamp,"unclamp set")
        if len(set_clamp_round) == 0:
            phase = phase+1

    # debug after optlayer
    final_sum = 0
    for i in range(num_actions):
        final_sum = final_sum+z[i]
        # make sure not violate the local constraint
        assert lower[i] <= z[i] <= upper_bound[i]
    final_sum = round(final_sum, 2)
   # print(final_sum)
    assert final_sum == nbikes     # make sure sum is equal to bike number
    if np.sum(y) == nbikes:
        assert z == y
    return z, grad_z


np_a = np.array([5, 6, 7, 8])
z, grad_z = OptLayer_function(np_a)
print(z)


tf_a = tf.Variable([5, 6, 7, 8], dtype=tf.float32)
tf_z, tf_grad_z = OptLayer_op(tf_a)
print(tf_z)
