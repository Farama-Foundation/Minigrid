from project_RL.sarsa.sarsa_lambda_agent import SarsaLambda
from gym_minigrid.wrappers import *
from time import time


def train(env, hyperparameters):
    """ Train a sarsa lambda agent in the requested environment

    Arguments:
        hyperparameters dictionary containing:
            - env_name
            - discount_rate
            - learning_rate
            - epsilon
    """
    agent = SarsaLambda(env, hyperparameters['discount_rate'],
                        hyperparameters['learning_rate'], hyperparameters['epsilon'])

    # initialise variables for plotting purpose
    step = 0
    total_reward = 0.0
    prev_total_reward = 0.0

    for episode in range(int(1e4)):
        # reset environment before each episode
        agent.init_eligibility_table()
        observation = env.reset()
        state = observation  # TODO: change it after decision regarding state
        action = agent.get_new_action_e_greedly(state)
        done = False

        # env.render()
        while not done:
            observation, reward, done, info = env.step(action)
            next_state = observation  # TODO: change it after decision regarding state
            total_reward += reward
            next_action = agent.get_new_action_e_greedly(next_state)

            # t0 = time()
            agent.update(state, action, reward, next_state, next_action, done)
            # print(f'{time() - t0} elapsed.')

            state = next_state
            action = next_action
            # env.render()
            # print("a:", action, "i:", step, "reward:", reward, "info:", info)
            if done:
                # env.render()
                if total_reward > prev_total_reward:
                    print("done?", done, "total reward:", total_reward, "info:", info)
                    play(env, agent)
                    prev_total_reward = total_reward
            step += 1
    env.close()
    return agent


def play(env, agent, episodes=1):
    for episode in range(episodes):
        # reset environment before each episode
        observation = env.reset()
        state = observation  # TODO: change it after decision regarding state
        action = agent.get_new_action_e_greedly(state)
        done = False
        total_reward = 0

        env.render()
        while not done:
            observation, reward, done, info = env.step(action)
            env.render()
            next_state = observation  # TODO: change it after decision regarding state
            total_reward += reward
            action = agent.get_new_action_e_greedly(next_state)
        print(f'Total reward: {total_reward}')


if __name__ == '__main__':
    hyperparameters = {
        'env_name': 'MiniGrid-Empty-8x8-v0',
        'discount_rate': 0.9,
        'learning_rate': 0.1,
        'epsilon': 0.3
    }

    env = gym.make(hyperparameters['env_name'])
    agent = train(env, hyperparameters)
