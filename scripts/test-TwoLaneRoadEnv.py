import time
import gym
import pedgrid
from pedgrid.agents import *
import logging

env = gym.make('TwoLaneRoadEnv60x80-v0')
env.reset()

v1 = SimpleVehicle(1, (14, 20), (20, 29), 1, 5, 5, 1, 1)
v2 = SimpleVehicle(2, (39, 60), (45, 69), 3, 5, 5, 1, 2)
p1 = BlueAdlerPedAgent(id=1, position=(0, 42), direction=Direction.East, maxSpeed=4, speed = 3)
p2 = BlueAdlerPedAgent(id=2, position=(59,44), direction=Direction.West, maxSpeed=4, speed = 3)
p3 = BlueAdlerPedAgent(id=3, position=(5,5), direction=Direction.South, maxSpeed=4, speed = 3)
p4 = BlueAdlerPedAgent(id=4, position=(55,60), direction=Direction.North, maxSpeed=4, speed = 3)

env.addVehicleAgent(v1)
env.addVehicleAgent(v2)
env.addPedAgent(p1)
env.addPedAgent(p2)
env.addPedAgent(p3)
env.addPedAgent(p4)

for i in range(110):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    if i % 10 == 0:
        logging.info(f"Completed step {i+1}")

    time.sleep(0.5)