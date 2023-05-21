import time
import gym
import gym_minigrid
from gym_minigrid.agents import *
import logging
env = gym.make('TwoLaneRoadEnv60x80-v0')       
env.reset()

env.addVehicleAgent(Vehicle(1, (30, 30), (40, 35), 1, 2, 2, 1, 1))
p1 = BlueAdlerPedAgent(id=1, position=(1,1), direction=2, maxSpeed=4, speed = 3)
p2 = BlueAdlerPedAgent(id=1, position=(9,3), direction=Direction.West, maxSpeed=4, speed = 3)
env.addPedAgent(p1)
env.addPedAgent(p2)

for i in range(110):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    if i % 10 == 0:
        logging.info(f"Completed step {i+1}")

    time.sleep(0.5)