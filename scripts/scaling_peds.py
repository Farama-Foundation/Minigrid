import time
import random
import gym
import numpy as np
import pedgrid
from pedgrid.wrappers import *
from pedgrid.agents import BlueAdlerPedAgent
from pedgrid.lib.MetricCollector import MetricCollector
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO)

den = []
vols = []
p = []
split = []
speeds = []
speedy = []
ped_count = []
run_time = []

# values = np.zeros((6, 19, 3)) # 6 directional split starting from 50/50 to 100/0, 19 densities, 2 values/trial
# for p_exchg in [0.5]:
#     for dirSplitInt in range(5, 6):
#         # for densityInt in range(30, 31):
#         for numPeds in range(450, 501, 50):
        
#         # Load the gym environment
#             env = gym.make('PedestrianEnv-20x80-v0')
#             metricCollector = MetricCollector(env, 0, 100)
#             agents = []

#             # density = round(0.025 * densityInt, ndigits=3)

#             # density = 0.04
#             DML = False
#             # p_exchg = 1 # 0.5 for 3rd graph, 1.0 for 1st and 2nd graphs
#             dirSplit = round(dirSplitInt/10, ndigits=1)
#             # dirSplit = 0.9

#             # print("Density: " + str(density) + " Directional Split: " + str(dirSplit))

#             possibleX = list(range(0, env.width))
#             possibleY = list(range(1, env.height - 1))
#             possibleCoordinates = []
#             for i in possibleX:
#                 for j in possibleY:
#                     possibleCoordinates.append((i, j))

#             # logging.info(f"Number of possible coordinates is {len(possibleCoordinates)}")

#             # for i in range(int(density * env.width * (env.height - 2))): # -2 from height to account for top and bottom
#             for i in range(numPeds):
#                 randomIndex = np.random.randint(0, len(possibleCoordinates) - 1)
#                 pos = possibleCoordinates[randomIndex]
#                 direction = 2 if np.random.random() > dirSplit else 0
#                 toss = np.random.random()
#                 speed = 0
#                 if toss > .1:
#                     speed = 3
#                 elif toss >0.05:
#                     speed = 2
#                 else:
#                     speed = 4
#                 agents.append(BlueAdlerPedAgent(i, pos, direction, speed, speed, DML, p_exchg, speed))
#                 del possibleCoordinates[randomIndex]
#             env.addPedAgents(agents)

#             env.reset()
#             start = time.time()
#             for i in range(1000):

#                 obs, reward, done, info = env.step(None)
                
#                 if done:
#                     "Reached the goal"
#                     break

#                 # env.render()


#                 if i % 10 == 0:
#                     logging.info(f"Completed step {i+1}")

#                 # time.sleep(2)
#             end = time.time()
#             logging.info(f"Elapsed time {end - start}s")

#             ped_count.append(numPeds)
#             run_time.append(end - start)

#             # Test the close method

#             env.close()

pedVStime = pd.DataFrame({'pedCount': ped_count, 'runTime': run_time})
# pedVStime.to_csv('outputs/pedVStime1000steps.csv', index=False)
pedVStime.to_csv('outputs/pedVStime1000steps.csv', mode='a', header=False, index=False)
pedVStime = pd.read_csv('outputs/pedVStime1000steps.csv')

plt.plot(pedVStime['pedCount'], pedVStime['runTime']/60, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Pedestrians')
plt.ylabel('Run Time (in minutes)')
plt.title('Performance Analysis (Bi-directional Pedestrian Flow):\nInterspersed, 1000 Steps, $p_{exchg}$ = 0.5, Direction Split 50/50')
plt.grid(True)

# # Define the quadratic function to fit the data (y = ax^2 + bx + c)
# def quadratic_function(x, a, b, c):
#     return a * x**2 + b * x + c

# # Perform curve fitting to find the coefficients (a, b, c)
# popt, _ = curve_fit(quadratic_function, pedVStime['pedCount'], pedVStime['runTime']/60)

# # Extract the coefficients (a, b, c)
# a_fit, b_fit, c_fit = popt

# # Generate points for the fitted curve using the quadratic function
# x_fit = np.linspace(min(pedVStime['pedCount']), max(pedVStime['pedCount']), 100)
# y_fit = quadratic_function(x_fit, a_fit, b_fit, c_fit)

# # Plot the fitted quadratic curve
# plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')

# # Add labels and title
# plt.title('Quadratic Curve Fitting')

plt.savefig('outputs/scaling1000steps.png')

plt.show()