import gym
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()
inputs1 = tf.placeholder(shape=[1, 16],dtype=tf.float32)
w = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
q_out = tf.matmul(inputs1, w)
predict = tf.argmax(q_out, 1)

next_q = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_q - q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()
y = .9
e = 0.9
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    s = env.reset()
    while True:
        a, all_q = sess.run([predict, q_out], feed_dict={inputs1: np.identity(16)[s:s + 1]})
        if np.random.rand(1) < e:
            a[0] = env.action_space.sample()
        s1, r, d, _ = env.step(a[0])
        all_q_1 = sess.run(q_out, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})
        max_all_q_1 = np.max(all_q_1)
        target_q = all_q
        target_q[0, a[0]] = r + y * max_all_q_1
        sess.run([updateModel, w], feed_dict={inputs1: np.identity(16)[s:s + 1], next_q: target_q})
        print(w.eval(), loss.eval({inputs1: np.identity(16)[s:s + 1], next_q: target_q}), e)
        s = s1
        if d is True:
            e -= 0.001
            s = env.reset()
