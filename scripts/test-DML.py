import time
import random
import gym
import numpy as np
import pedgrid
from pedgrid.wrappers import *
from pedgrid.agents import PedAgent
from pedgrid.lib.MetricCollector import MetricCollector
import logging
import pickle5 as pickle

logging.basicConfig(level=logging.INFO)

# from pedgrid.envs import MultiPedestrianEnv
# %matplotlib auto
# %load_ext autoreload
# %autoreload 2

values = np.zeros((6, 19, 2)) # 6 directional split starting from 50/50 to 100/0, 19 densities, 2 values/trial
for dirSplitInt in range(5, 11):
    for densityInt in range(1, 20):

        # Load the gym environment
        env = gym.make('MultiPedestrian-Empty-20x80-v0')
        metricCollector = MetricCollector(env)
        agents = []

        density = round(0.05 * densityInt, ndigits=2)
        DML = True
        p_exchg = 0.5 # 0.5 for 6th graph, 1.0 for 5th graph
        dirSplit = round(dirSplitInt/10, ndigits=1)

        print("Density: " + str(density) + " Directional Split: " + str(dirSplit))

        possibleX = list(range(0, env.width))
        possibleY = list(range(1, env.height - 1))
        possibleCoordinates = []
        for i in possibleX:
            for j in possibleY:
                possibleCoordinates.append((i, j))

        logging.info(f"Number of possible coordinates is {len(possibleCoordinates)}")

        for i in range(int(density * env.width * (env.height - 2))): # -2 from height to account for top and bottom
            randomIndex = np.random.randint(0, len(possibleCoordinates) - 1)
            pos = possibleCoordinates[randomIndex]
            direction = 2 if np.random.random() > dirSplit else 0
            agents.append(PedAgent(i, pos, direction, DML, p_exchg))
            del possibleCoordinates[randomIndex]
        env.addAgents(agents)

        env.reset()

        for i in range(1100):

            obs, reward, done, info = env.step(None)
            
            if done:
                "Reached the goal"
                break

            # env.render()

            if i % 100 == 0:
                logging.info(f"Completed step {i+1}")

            # time.sleep(0.05)

        logging.info(env.getAverageSpeed())

        stepStats = metricCollector.getStatistics()[0]
        avgSpeed = sum(stepStats["xSpeed"]) / len(stepStats["xSpeed"])
        logging.info("Average speed: " + str(avgSpeed))
        volumeStats = metricCollector.getStatistics()[1]
        avgVolume = sum(volumeStats) / len(volumeStats)
        logging.info("Average volume: " + str(avgVolume))

        values[dirSplitInt - 5][densityInt - 1][0] = avgSpeed
        values[dirSplitInt - 5][densityInt - 1][1] = avgVolume

        # Test the close method
        env.close()

with open(f"DML_p=0.5.pickle", "wb") as f: # already set to correct name for the next dataset to run (graph 6)
    pickle.dump(values, f)