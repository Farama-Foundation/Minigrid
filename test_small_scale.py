import time
import random
import gym
import numpy as np
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.lib.MetricCollector import MetricCollector
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)

den = []
vols = []
p = []
split = []
speeds = []
speedy = []

values = np.zeros((6, 19, 3)) # 6 directional split starting from 50/50 to 100/0, 19 densities, 2 values/trial
for p_exchg in [0, 0.25, 0.5, 0.75, 1]:
    for dirSplitInt in range(5, 11):
        for densityInt in range(1, 40):
        
        # Load the gym environment
            env = gym.make('MultiPedestrian-Empty-5x20-v0')
            metricCollector = MetricCollector(env, 0, 100)
            agents = []

            density = round(0.025 * densityInt, ndigits=3)

            # density = 0.04
            DML = False
            # p_exchg = 1 # 0.5 for 3rd graph, 1.0 for 1st and 2nd graphs
            dirSplit = round(dirSplitInt/10, ndigits=1)
            # dirSplit = 0.9

            print("Density: " + str(density) + " Directional Split: " + str(dirSplit))

            possibleX = list(range(0, env.width))
            possibleY = list(range(1, env.height - 1))
            possibleCoordinates = []
            for i in possibleX:
                for j in possibleY:
                    possibleCoordinates.append((i, j))

            # logging.info(f"Number of possible coordinates is {len(possibleCoordinates)}")

            for i in range(int(density * env.width * (env.height - 2))): # -2 from height to account for top and bottom
                randomIndex = np.random.randint(0, len(possibleCoordinates) - 1)
                pos = possibleCoordinates[randomIndex]
                direction = 2 if np.random.random() > dirSplit else 0
                toss = np.random.random()
                speed = 0
                if toss > .1:
                    speed = 3
                elif toss >0.05:
                    speed = 2
                else:
                    speed = 4
                agents.append(BlueAdlerPedAgent(i, pos, direction, speed, speed, DML, p_exchg, speed))
                del possibleCoordinates[randomIndex]
            env.addAgents(agents)

            env.reset()

            for i in range(120):

                obs, reward, done, info = env.step(None)
                
                if done:
                    "Reached the goal"
                    break

                # env.render()


                if i % 10 == 0:
                    logging.info(f"Completed step {i+1}")

                # time.sleep(2)

            # logging.info(env.getAverageSpeed())

            stepStats = metricCollector.getStatistics()[0]
            avgSpeed = sum(stepStats["xSpeed"]) / len(stepStats["xSpeed"])
            avgSpeedy = sum(stepStats["ySpeed"]) / len(stepStats["ySpeed"])
            # logging.info("Average speed: " + str(avgSpeed))
            volumeStats = metricCollector.getStatistics()[1]
            avgVolume = sum(volumeStats) / len(volumeStats)
            # logging.info("Average volume: " + str(avgVolume))
            vols.append(avgVolume)
            den.append(density)
            p.append(p_exchg)
            split.append(dirSplit)
            speeds.append(env.getAverageSpeed())
            # speeds.append(avgSpeed) 
            speedy.append(avgSpeedy)

            # Test the close method

            env.close()
data = {"Density":den, "Volume":vols, "Probability": p, "Direction Split":split, "SpeedX":speeds, "SpeedY":speedy}
data = pd.DataFrame(data)
data.to_csv("data2")
# print(data)
# sns.lineplot(data=data, x="Density", y="SpeedX", hue="Direction Split", palette = "flag", markers=True)
# plt.show()
# with open(f"testing.pickle", "wb") as f:
#     pickle.dump(values, f)