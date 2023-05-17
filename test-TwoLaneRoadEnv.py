import time
import gym
import gym_minigrid
import logging
env = gym.make('TwoLaneRoadEnv-20x80-v0')
env.reset()

for i in range(110):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    if i % 10 == 0:
        logging.info(f"Completed step {i+1}")

    time.sleep(0.5)