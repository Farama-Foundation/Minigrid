import time
import gym
from pedgrid.agents import *
import logging

from pedgrid.agents.StopGoModel.stopGoPed import StopGoPed
from pedgrid.agents.StopGoModel.stopGoVehicle import StopGoVehicle
env = gym.make('TwoLaneRoadEnv60x80-v0')       
env.reset()

v1 = StopGoVehicle(1, (14, 20), (20, 29), Direction.South, 5, 2, 1, 1)
v2 = StopGoVehicle(2, (14, 35), (20, 40), Direction.South, 5, 4, 1, 1)
v3 = StopGoVehicle(3, (39, 60), (45, 69), Direction.North, 5, 5, 1, 2)

p1 = StopGoPed(id=1, position=(0, 42), direction=Direction.East, minTimeToCross= 5, speed = 3)
p2 = StopGoPed(id=2, position=(59,44), direction=Direction.West, minTimeToCross=6, speed = 3)
p3 = StopGoPed(id=3, position=(0, 43), direction=Direction.East, minTimeToCross= 7, speed = 3)
p4 = StopGoPed(id=4, position=(0,44), direction=Direction.East, minTimeToCross = 4, speed = 4)

# p3 = BlueAdlerPedAgent(id=3, position=(5,5), direction=Direction.South, maxSpeed=4, speed = 3)
# p4 = BlueAdlerPedAgent(id=4, position=(55,60), direction=Direction.North, maxSpeed=4, speed = 3)

env.addVehicleAgent(v1)
env.addVehicleAgent(v2)
env.addVehicleAgent(v3)
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

    time.sleep(0.05)