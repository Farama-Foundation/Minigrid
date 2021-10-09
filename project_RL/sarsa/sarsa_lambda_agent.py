import random

import numpy as np


class SarsaLambda:
    """Sarsa Lambda Algorithm exploits and explores an environment.
    It learns the state action value of each state and action pair.
    It converges to the optimal action-value function.
    """

    def __init__(self, env, discount_rate=0.9, learning_rate=0.1, epsilon=0.5):
        self.action_size = env.action_space.n
        # TODO: update README to explain the reasoning behind the state decision
        # the state size is: dlw x drw x duw x dbw x dxg x dyg
        # 7 x 7 x 7 x 7 x 7 x 7 x 4 = 823543
        self.state_size = 823543
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.init_q_value_table()

    def init_q_value_table(self):
        """Initialise q value table with zeros.
        Its first dimension is the state size and the second dimension is the action size.
        """
        self.q_value_table = np.zeros((self.state_size, self.action_size), np.int)

    def init_eligibility_table(self):
        """Initialise eligibility trace table with zeros. Must be invoked before each episode.
        Its first dimension is the state size and the second dimension is the action size.
        """
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

    def update(self, state, action, reward, new_state, new_action, done):
        """Updates the state action value for every pair state and action
        in proportion to TD-error and eligibility trace

        """
        q_value_state_s = self.q_value_table[state["direction"]]
        q_value_new_state = self.q_value_table[new_state["direction"]]
        td_error = reward + self.discount_rate * q_value_new_state[new_action] - q_value_state_s[action]

        # TODO: update tables
        # for all state s and action a,
        # update q_value_table , if td_error != 0:
        # apply decay to eligibility_table
