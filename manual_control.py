#!/usr/bin/env python3

import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_minigrid

def reset():
    env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        plt.xlabel(env.mission)

    img = env.render('rgb_array')
    imshow_obj.set_data(img)
    fig.canvas.draw()

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()

    img = env.render('rgb_array')
    imshow_obj.set_data(img)
    fig.canvas.draw()

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        plt.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--env-name",
    dest="env_name",
    help="gym environment to load",
    #default='MiniGrid-MultiRoom-N6-v0'
    default='MiniGrid-Empty-8x8-v0'
)
args = parser.parse_args()

env = gym.make(args.env_name)

fig, ax = plt.subplots()

# Keyboard handler
fig.canvas.mpl_connect('key_press_event', key_handler)

# Show the env name in the window title
fig.canvas.set_window_title('gym_minigrid - ' + args.env_name)

# Turn off x/y axis numbering/ticks
ax.set_xticks([], [])
ax.set_yticks([], [])

# Show the first image of the environment
img = env.render('rgb_array')
imshow_obj = ax.imshow(img, interpolation='bilinear')

reset()

# Show the plot, enter the matplotlib event loop
plt.show()
