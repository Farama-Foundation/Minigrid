exec(open("sys_path_hack.py").read())
import gym
import pedgrid
from pedgrid.agents import BlueAdlerPedAgent, SimpleVehicle
from pedgrid.agents import TutorialPedAgent
from pedgrid.lib.Direction import Direction
import time

# Load the gym environment
env = gym.make('TwoLaneRoadEnv60x80-v0')
env.reset()

# create your actors. Each actor needs some configuration
pedestrians = []
vehicles = []

firstPed = BlueAdlerPedAgent(
                    id=1,
                    position=(0,2),
                    direction=Direction.West,
                )

secondPed = TutorialPedAgent (
                    id=2,
                    position=(0,1),
                    direction=Direction.West,

)

v1 = SimpleVehicle(1, (14, 20), (20, 29), 1, 5, 5, 1, 1)
v2 = SimpleVehicle(2, (39, 60), (45, 69), 3, 5, 5, 1, 2)
pedestrians.append(firstPed)
pedestrians.append(secondPed)
vehicles.append(v1)
vehicles.append(v2)


# attach agents to the environment
env.addPedAgents(pedestrians)
env.addVehicleAgents(vehicles)

for i in range(40):
    env.step(None)
    env.render()
    time.sleep(1)
env.close()