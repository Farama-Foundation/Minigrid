import time
import gym
import gym_minigrid
from gym_minigrid.agents import *
import logging

from gym_minigrid.agents.SimpleVehicle import SimpleVehicle
env = gym.make('TwoLaneRoadEnv30x80-v0')       
env.reset()

v1 = SimpleVehicle(1, (14, 20), (20, 29), 1, 5, 5, 1, 1)
p1 = SimplePedAgent(id=1, position=(0, 42), direction=Direction.East, maxSpeed=4, speed = 3)

env.addVehicleAgent(v1)
env.addPedAgent(p1)

for i in range(110):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    if i % 10 == 0:
        logging.info(f"Completed step {i+1}")

    time.sleep(0.5)