exec(open("sys_path_hack.py").read())
import gym
import pedgrid
from pedgrid.agents import BlueAdlerPedAgent
from pedgrid.agents import TutorialPedAgent
from pedgrid.lib.Direction import Direction
import time

# Load the gym environment
env = gym.make('MultiPedestrian-Empty-5x20-v0')
env.reset()

# create your actors. Each actor needs some configuration
pedestrians = []

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
pedestrians.append(firstPed)
pedestrians.append(secondPed)

# attach agents to the environment
env.addPedAgents(pedestrians)

for i in range(40):
    env.step(None)
    env.render()
    time.sleep(1)
env.close()