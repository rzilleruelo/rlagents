import gym
import numpy as np

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = .85
y = .99
num_episodes = 2000

rList = []
for i in range(num_episodes):
    s = env.reset()
    r_all = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        a = np.argmax(Q[s, :] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        s1, r, d, x = env.step(a)
        Q[s, a] += lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        r_all += r
        s = s1
        if d is True:
            break
    rList.append(r_all)

print('Score over time: ' + str(sum(rList)/num_episodes))
print('Final Q-Table Values')
print(Q)

s = env.reset()
while True:
    a = np.argmax(Q[s, :] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
    s1, r, d, x = env.step(a)
    print(a, s1, r, d, x)
    s = s1
    if d is True:
        break
    env.render()
