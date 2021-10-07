from project_RL.sarsa.sarsa_lambda_agent import SarsaLambdaAgent
from gym_minigrid.wrappers import *


def train(hyperparameters):
    """ Train a sarsa lambda agent in the requested environment

    Arguments:
        hyperparameters dictionary containing:
            - env_name
            - discount_rate
            - learning_rate
            - epsilon
    """
    env = gym.make(hyperparameters['env_name'])
    agent = SarsaLambdaAgent(env, hyperparameters['discount_rate'],
                             hyperparameters['learning_rate'], hyperparameters['epsilon'])

    # initialize variables for plotting purpose
    step = 0
    total_reward = 0.0

    for episode in range(100):
        # reset environment before each episode
        agent.init_eligibility_table()
        observation = env.reset()
        state = observation  # TODO: change it after decision regarding state
        action = agent.get_new_action_e_greedly(state)
        done = False

        env.render()
        while not done:
            observation, reward, done, info = env.step(action)
            next_state = observation  # TODO: change it after decision regarding state
            total_reward += reward
            next_action = agent.get_new_action_e_greedly(next_state)

            agent.train(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            env.render()
            print("a:", action, "i:", step, "reward:", reward, "info:", info)
            if done:
                print("done?", done, "total reward:", total_reward, "info:", info)
            step += 1

    env.close()


hyperparameters = {
    'env_name': 'MiniGrid-Empty-8x8-v0',
    'discount_rate': 0.9,
    'learning_rate': 0.1,
    'epsilon': 0.3
}

train(hyperparameters)
