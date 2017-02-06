import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

GAMMA = 0.7
STATE_SIZE = 2
GRID_SIZE = 4
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'NOTHING']
EXPLORE_RATE = 0.1


def build_model():
    s = tf.placeholder(tf.float32, [None, STATE_SIZE])
    next_state_q = tf.placeholder(shape=[None, 1, len(ACTIONS)], dtype=tf.float32)

    w = tf.Variable(tf.random_uniform([STATE_SIZE, len(ACTIONS)], -1.0, 1.0))
    q = tf.nn.relu(tf.matmul(s, w))

    a = tf.argmax(q, 1)

    loss = tf.reduce_sum(tf.square(next_state_q - q))
    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    return {'s': s, 'q': q, 'a': a, 'next_state_q': next_state_q, 'train_step': train_step}, (w)


def reset():
    internal_state = {
        'mouse': np.random.randint(GRID_SIZE, size=2, dtype=np.int32),
        'cheese': np.random.randint(GRID_SIZE, size=2, dtype=np.int32),
        'plays': 0
    }
    state = {'mouse': (internal_state['cheese']-internal_state['mouse'])}
    return internal_state, state


def step(agents_a, internal_state):
    mouse_previous_distance = np.sum(np.square(internal_state['cheese']-internal_state['mouse']))
    for agent, a in agents_a.items():
        a_label = ACTIONS[a]
        agent_internal_state = internal_state[agent]
        if a_label == 'UP':
            if agent_internal_state[1] < GRID_SIZE-1:
                agent_internal_state[1] += 1
        elif a_label == 'RIGHT':
            if agent_internal_state[0] < GRID_SIZE-1:
                agent_internal_state[0] += 1
        elif a_label == 'DOWN':
            if agent_internal_state[1] > 0:
                agent_internal_state[1] -= 1
        elif a_label == 'LEFT':
            if agent_internal_state[0] > 0:
                agent_internal_state[0] -= 1
    rewards = {'mouse': 0.0}
    mouse_current_distance = np.sum(np.square(internal_state['cheese'] - internal_state['mouse']))
    if np.all(internal_state['mouse'] == internal_state['cheese']):
        rewards['mouse'] = 1.0
        internal_state['cheese'] = np.random.randint(GRID_SIZE, size=2)
        internal_state['plays'] = 0
    elif internal_state['plays'] >= GRID_SIZE * GRID_SIZE:
        rewards['mouse'] = -1.0
        internal_state['plays'] = 0
    elif mouse_current_distance < mouse_previous_distance:
        rewards['mouse'] = 0.01

    state = {'mouse': (internal_state['mouse'] - internal_state['cheese'])}
    internal_state['plays'] += 1
    return internal_state, state, rewards


def draw(internal_state, ephocs):
    size = 256
    scale = size // GRID_SIZE
    width = scale//2
    im = Image.new('RGBA', (size, size), 'white')
    canvas = ImageDraw.Draw(im)
    x, y = internal_state['mouse'] * scale
    canvas.rectangle((x - width, y - width, x + width, y + width), fill='blue')
    x, y = internal_state['cheese'] * scale
    canvas.rectangle((x - width, y - width, x + width, y + width), fill='green')
    im.save('/tmp/openai-universe/drawing-%06d.png' % ephocs)


def main():
    agents = {'mouse': build_model()}
    w = agents['mouse'][1]
    agents['mouse'] = agents['mouse'][0]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        agents_acc_r = {'mouse': 0.0}
        ephocs = 1
        counter = 1
        internal_state, agents_current_s = reset()
        while True:
            draw(internal_state, ephocs)
            agents_actions = {}
            for agent, params in agents.items():
                current_a, current_q = sess.run(
                    [params['a'], params['q']],
                    feed_dict={params['s']: [agents_current_s[agent]]}
                )
                if np.random.random() <= EXPLORE_RATE:
                    current_a = np.array([1])
                    current_a[0] = np.random.randint(len(ACTIONS))
                agents_actions[agent] = {'current_a': current_a, 'current_q': current_q}
            agents_a = {agent: action['current_a'][0] for agent, action in agents_actions.items()}
            internal_state, agents_next_s, agents_r = step(agents_a, internal_state)

            for agent, r in agents_r.items():
                agents_acc_r[agent] += r
            for agent, acc_r in agents_acc_r.items():
                print('e: %d a: %s r: %0.3f w:' % (ephocs, agent, acc_r/counter))
                for w_i in sess.run(w):
                    print(w_i)

            for agent, next_s in agents_next_s.items():
                params = agents[agent]
                actions = agents_actions[agent]
                next_q = sess.run(params['q'], feed_dict={params['s']: [next_s]})
                max_next_q = np.max(next_q)
                target_q = actions['current_q']
                target_q[0, actions['current_a'][0]] = agents_r[agent] + GAMMA * max_next_q
                sess.run(
                    params['train_step'],
                    feed_dict={params['s']: [agents_current_s[agent]], params['next_state_q']: [target_q]}
                )
            agents_current_s = agents_next_s
            ephocs += 1
            counter += 1
            if counter >= 10000:
                counter = 1
                agents_acc_r = {'mouse': 0.0}


if __name__ == '__main__':
    main()
