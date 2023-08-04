import gym
import gym_minigrid
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.agents import PedAgent
from gym_minigrid.lib.Direction import Direction
import time

env = gym.make('TwoLaneRoadEnv60x80-v0')
env.reset()

pedestrians = []

firstPed = BlueAdlerPedAgent(
                    id=1,
                    position=(5,50),
                    direction=Direction.North,
                )

secondPed = BlueAdlerPedAgent(
                    id=2,
                    position=(90,90),
                    direction=Direction.South,
                )

thirdPed = BlueAdlerPedAgent(
                    id=3,
                    position=(30,90),
                    direction=Direction.South,
                )               

fourthPed = BlueAdlerPedAgent(
                    id = 4, 
                    position = (15, 43), 
                    direction = Direction.West,
                )
pedestrians.append(firstPed)
pedestrians.append(secondPed)
pedestrians.append(thirdPed)
pedestrians.append(fourthPed)



# attach agents to the environment
env.addPedAgents(pedestrians)

for i in range(40):
    env.step(None)
    env.render()
    time.sleep(1)
env.close()