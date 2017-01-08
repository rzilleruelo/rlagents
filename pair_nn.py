import functools

import numpy as np
import tensorflow as tf


STATE_SIZE = 4
ACTIONS_SIZE = 2
SAMPLE_SIZE = 5000


def build_model():
    s = tf.placeholder(tf.float32, [None, STATE_SIZE])
    next_state_q = tf.placeholder(shape=[None, 1, ACTIONS_SIZE], dtype=tf.float32)

    w = tf.Variable(tf.random_uniform([STATE_SIZE, ACTIONS_SIZE], -1.0, 1.0))
    q = tf.matmul(s, w)
    a = tf.argmax(q, 1)

    loss = tf.reduce_sum(tf.square(next_state_q - q))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    return s, q, a, next_state_q, train_step


def reset():
    return np.random.randint(2, size=[STATE_SIZE])


def step(state, a):
    expected_a = functools.reduce(lambda result, v: (2 ** v[0]) * v[1] + result, enumerate(state), 0) % 2
    r = 1.0 if expected_a == a[0] else -1.0
    return np.random.randint(2, size=[STATE_SIZE]), r


def main():
    s, q, a, next_state_q, train_step = build_model()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        e = 0.9
        gamma = 0.99
        current_s = reset()
        while True:
            current_a, current_q = sess.run([a, q], feed_dict={s: [current_s]})
            if np.random.random() <= e:
                current_a = [np.random.randint(2)]
            next_s, r = step(current_s, current_a)
            print(r)
            next_q = sess.run(q, feed_dict={s: [next_s]})
            max_next_q = np.max(next_q)
            target_q = current_q
            target_q[0, current_a[0]] = r + gamma * max_next_q
            sess.run(train_step, feed_dict={s: [current_s], next_state_q: [target_q]})
            current_s = next_s
            e -= 0.001



if __name__ == '__main__':
    main()
