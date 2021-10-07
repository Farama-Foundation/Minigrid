import random

import numpy as np


class SarsaLambdaAgent:
    """Sarsa Lambda Agent exploits and explores an environment
    in order to converge to the optimal action-value function
    """

    def __init__(self, env, discount_rate=0.9, learning_rate=0.1, epsilon=0.5):
        self.action_size = env.action_space.n

        # TODO: decide together in group what we will store as state
        # the observation space seems very big, view env.observation_space
        # there are 10 possible value for objects, 6 colors, 3 door states (view minigrid.py for more details)
        # the default agent view size is 7x7
        # plus 4 directions
        # the total number of combinations of observation is gigantic
        # my suggestion is to use as state the distance to walls, obstacles, the goal, and other objects of interest
        self.state_size = 10
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.init_q_value_table()

    def init_q_value_table(self):
        self.q_value_table = np.zeros((self.state_size, self.action_size), np.int)

    def init_eligibility_table(self):
        """Initialise eligibility trace table with zeros. Must be invoked before each episode."""
        self.eligibility_table = np.zeros((self.state_size, self.action_size))

    def get_new_action_e_greedly(self, state):
        """With probability 1 - epsilon choose the greedy action.
        With probability epsilon choose random action.
        """
        q_state = self.q_value_table[state["direction"]]
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            return np.argmax(q_state)

    def train(self, state, action, reward, new_state, new_action, done):
        """Updates the action value for every pair state, action
        in proportion to TD-error and eligibility trace

        """
        q_value_state_s = self.q_value_table[state["direction"]]
        q_value_new_state = self.q_value_table[new_state["direction"]]
        td_error = reward + self.discount_rate * q_value_new_state[new_action] - q_value_state_s[action]

        # TODO: update tables
        # for all state s and action a,
        # update q_value_table , if td_error != 0:
        # apply decay to eligibility_table
