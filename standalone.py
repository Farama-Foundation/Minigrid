#!/usr/bin/env python3

from __future__ import division, print_function

import numpy
import gym
import time

import gym_minigrid
from gym_minigrid.envs import MiniGridEnv

def main():

    env = gym.make('MiniGrid-Multi-Room-N6-v0')
    env.reset()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        action = 0
        if keyName == 'LEFT':
            action = MiniGridEnv.ACTION_LEFT
        elif keyName == 'RIGHT':
            action = MiniGridEnv.ACTION_RIGHT
        elif keyName == 'UP':
            action = MiniGridEnv.ACTION_FORWARD
        elif keyName == 'SPACE':
            action = MiniGridEnv.ACTION_TOGGLE
        else:
            "unknown key"

        obs, reward, done, info = env.step(action)

        print('reward=%s' % reward)

        if done:
            print('done!')
            env.reset()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
