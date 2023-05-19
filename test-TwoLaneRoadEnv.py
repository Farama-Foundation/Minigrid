import time
import gym
import gym_minigrid
import logging
from gym_minigrid.agents.BlueAdlerPedAgent import BlueAdlerPedAgent
from gym_minigrid.agents.PedAgent import PedAgent
from gym_minigrid.agents.SimpleVehicle import SimpleVehicle
from gym_minigrid.agents.Vehicle import Vehicle
from gym_minigrid.lib.Direction import Direction
env = gym.make('TwoLaneRoadEnv-10x20-v0')
env.reset()

v1 = SimpleVehicle(id = 1, topLeft= (10, 5), bottomRight=(13, 6), direction=Direction.East, maxSpeed=1, speed = 1, inRoad=1, inLane=1)
p1 = BlueAdlerPedAgent(id=1, position=(1,1), direction=2, maxSpeed=4, speed = 3)
p2 = BlueAdlerPedAgent(id=1, position=(9,3), direction=Direction.West, maxSpeed=4, speed = 3)
env.addVehicleAgent(v1)
env.addPedAgent(p1)
env.addPedAgent(p2)
logging.basicConfig(level=logging.INFO)
for i in range(15):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    if i % 10 == 0:
        logging.info(f"Completed step {i+1}")

    time.sleep(0.5)