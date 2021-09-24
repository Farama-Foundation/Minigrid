#!/usr/bin/env python3
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

VEC_TO_DIR = {
    (1, 0): 0,
    (0, 1): 1,
    (-1, 0): 2,
    (0, -1): 3,
}

class OracleAgent:
    def __init__(self, env, visualize=True, seed=-1):
        self.env = env
        self.visualize = visualize
        self.seed = seed

    def redraw(self, img):
        # if not args.agent_view:
        img = self.env.render('rgb_array', tile_size=32)
        self.window.show_img(img)

    def reset(self):
        if self.seed != -1:
            self.env.seed(self.seed)

        obs = self.env.reset()

        if hasattr(self.env, 'mission') and self.visualize:
            print('Mission: %s' % self.env.mission)
            self.window.set_caption(self.env.mission)
        if self.visualize:
            self.redraw(obs)
        return obs, self.env.target_cell


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            print(f'step={self.env.step_count}, reward={reward:.2f}, done={done}')

        if self.visualize:
            self.redraw(obs)

        return obs, reward, done, info

    def get_sequence(self, goal):
        initial_states = [(*self.env.agent_pos, *self.env.dir_vec)]
        accept_fn = lambda i, j: [i, j] == list(goal)
        path, finish, previous_pos = self.breadth_first_search(self.env.grid, initial_states, accept_fn)
        for cell in path:
            cell = np.array(cell)
            print("curr_pos", self.env.agent_pos, cell)
            while not (self.env.agent_pos == cell).all():
                yield self.next_action(cell, next_cell_is_goal= (cell == goal).all() )

    def next_action(self, next_cell, next_cell_is_goal=False):
        curr_pos, curr_dir = self.env.agent_pos, self.env.agent_dir
        required_dir = VEC_TO_DIR[tuple(next_cell - curr_pos)]
        if required_dir == curr_dir:
            if next_cell_is_goal:
                return self.env.actions.pickup
            else:
                return self.env.actions.forward
        elif abs(required_dir - curr_dir) == 2:
            # facing 180 degrees. either right or left are fine
            return self.env.actions.right
        elif (curr_dir + 1) % 4 == required_dir:
            return self.env.actions.right
        else:
            return self.env.actions.left


    def breadth_first_search(self, grid, initial_states, accept_fn):
        """Performs breadth first search.
        This is pretty much your textbook BFS. The state space is agent's locations,
        but the current direction is also added to the queue to slightly prioritize
        going straight over turning.
        """
        bfs_counter = 0

        queue = [(state, None) for state in initial_states]
        previous_pos = dict()

        while len(queue) > 0:
            state, prev_pos = queue[0]
            queue = queue[1:]
            i, j, di, dj = state

            if (i, j) in previous_pos:
                continue

            bfs_counter += 1

            cell = grid.get(i, j)
            previous_pos[(i, j)] = prev_pos

            # If we reached a position satisfying the acceptance condition
            if accept_fn(i, j):
                path = []
                pos = (i, j)
                while pos:
                    path.append(pos)
                    pos = previous_pos[pos]
                path = path[::-1]
                return path, (i, j), previous_pos

            if not (cell is None or cell.can_overlap()):
                continue

            # # If this cell was not visually observed, don't expand from it
            # if not self.vis_mask[i, j]:
            #     continue

            if cell:
                if cell.type == 'wall':
                    continue
                # If this is a door
                elif cell.type == 'door':
                    # If the door is closed, don't visit neighbors
                    if not cell.is_open:
                        continue

            # Location to which the bot can get without turning
            # are put in the queue first
            for k, l in [(di, dj), (dj, di), (-dj, -di), (-di, -dj)]:
                next_pos = (i + k, j + l)
                next_dir_vec = (k, l)
                next_state = (*next_pos, *next_dir_vec)
                queue.append((next_state, (i, j)))

        # Path not found
        print("path not found")
        return None, None, previous_pos

    def generate_demos(self, num_demos=1):
        if self.visualize:
            self.window = Window('gym_minigrid')
        demos = []
        for demo in range(num_demos):
            obss, rewards, actions = [], [], []
            obs, target = self.reset()
            mission = obs["mission"]
            obss.append(obs)
            rewards.append(0)
            if self.visualize:
                # Blocking event loop
                self.window.show(block=False)

            for action in self.get_sequence(target):
                actions.append(action)
                obs, reward, done, info = self.step(action)
                obss.append(obs)
                rewards.append(0)
                if self.visualize:
                    time.sleep(0.5)
                if done:
                    break

            actions.append(self.env.actions.done)

            demos.append( (mission, obss, actions, rewards) )

        if self.visualize:
            self.window.close()

        return demos



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )
    parser.add_argument(
        '--agent_view',
        default=False,
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )

    args = parser.parse_args()

    env = gym.make(args.env)
    env = FullyObsWrapper(env)

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window('gym_minigrid - ' + args.env)
    # window.reg_key_handler(key_handler)
    obs, target = reset()
    # Blocking event loop
    window.show(block=False)

    while True:
        time.sleep(1)
        for action in get_sequence(env, target):
            print("ACTION", action)
            obs, reward, done, info = step(action)
            time.sleep(1)
            if done:
                print('done!')
                break


        # print(obs['image'][:, :, 0])
        # print(obs['image'][:, :, 1])
        # print(obs['image'][:, :, 2])
        # time.sleep(10)
        obs, target = reset()
