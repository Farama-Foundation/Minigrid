import gym
import gym_minigrid
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.lib.Direction import Direction
import time

# Load the gym environment
env = gym.make('MultiPedestrian-Empty-100x100-v0')
env.reset()

# create your actors. Each actor needs some configuration
pedestrians = []

firstPed = BlueAdlerPedAgent(
                    id=1,
                    position=(50,50),
                    direction=Direction.West,
                )
pedestrians.append(firstPed)

# attach agents to the environment
env.addPedAgents(pedestrians)

for i in range(40):
    env.step(None)
    env.render()
    time.sleep(1)
env.close()