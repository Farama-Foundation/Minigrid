#!/usr/bin/env python3

from __future__ import division, print_function

import numpy
import gym

import gym_minigrid

def main():

    env = gym.make('MiniGrid-Multi-Room-N6-v0')
    env.reset()

    # Create a window to render into
    renderer = env.render('human')

    while True:

        env.render('human')

        action = 0

        obs, reward, done, info = env.step(action)

        print('reward=%s' % reward)

        if done:
            print('done!')
            env.reset()

        # If the window was closed
        if not renderer.window:
            break

if __name__ == "__main__":
    main()
