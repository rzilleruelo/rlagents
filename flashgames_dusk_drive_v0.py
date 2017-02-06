import subprocess

import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import universe

ENVIRONMENTS_SIZE = 1
MEMORY_SIZE = 1000
BATCH_SIZE = 50
BUFFER_SIZE = 1
FFMPEG_CMD = ('ffmpeg', '-y', '-r', '5', '-i', '-', 'video.avi')
ACTIONS = [
    (),
    ('KeyEvent', 'ArrowUp', True),
    ('KeyEvent', 'ArrowUp', False),
    ('KeyEvent', 'ArrowDown', True),
    ('KeyEvent', 'ArrowDown', False),
    ('KeyEvent', 'ArrowRight', True),
    ('KeyEvent', 'ArrowRight', False),
    ('KeyEvent', 'ArrowLeft', True),
    ('KeyEvent', 'ArrowLeft', False)
]
INPUT_HEIGHT = 768
INPUT_WIDTH = 1024
INPUT_DEPTH = 3


class Plotter(object):
    def __init__(self, file_name):
        self._ffmpeg = subprocess.Popen(('ffmpeg', '-y', '-r', '5', '-i', '-', file_name), stdin=subprocess.PIPE)
        plt.figure()

    def add_frame(self, handler):
        handler(plt)
        plt.axis('off')
        plt.savefig(self._ffmpeg.stdin, bbox_inches='tight', format='png')
        plt.clf()


def set_buffer(src_buffer, dst_buffer, t):
    dst_buffer[0:BUFFER_SIZE - t] = src_buffer[t:BUFFER_SIZE]
    dst_buffer[BUFFER_SIZE - t: BUFFER_SIZE] = src_buffer[0:t]


def train(
    environments_size=ENVIRONMENTS_SIZE,
    memory_size=MEMORY_SIZE,
    buffer_size=BUFFER_SIZE,
    input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH),
    actions=ACTIONS
):
    plotter = Plotter('video.avi')
    memory = np.zeros([memory_size, buffer_size, *input_shape], dtype=np.uint8)
    memory_buffer = np.zeros([environments_size, buffer_size, *input_shape], dtype=np.uint8)
    t = np.zeros(environments_size, dtype=np.int8)
    memory_index = 0

    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(remotes=environments_size)
    observation_n = env.reset()
    while True:
        action_n = [[actions[1]] for ob in observation_n]
        observation_n, reward_n, done_n, info = env.step(action_n)
        for ob_index, ob in enumerate(observation_n):
            if ob:
                memory_buffer[ob_index, t[ob_index]] = ob['vision']
                set_buffer(memory_buffer[ob_index], memory[memory_index], t[ob_index])
                for image in memory[memory_index]:
                    plotter.add_frame(lambda x: x.imshow(image))
                t[ob_index] = (t[ob_index] + 1) % buffer_size
                memory_index = (memory_index + 1) % memory_size
        env.render()


if __name__ == '__main__':
    train()
