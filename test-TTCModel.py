import time
import gym
import gym_minigrid
from gym_minigrid.agents import *
import logging
from gym_minigrid.agents.SimpleVehicle import SimpleVehicle
from gym_minigrid.lib.MetricCollector import MetricCollector
import pickle
import pandas as pd

for trials in range(100):
    env = gym.make('TwoLaneRoadEnv900x270-v0')       
    env.reset()
    metricCollector = MetricCollector(env)

    mean = 40
    std_dev = 12
    lower_bound = 5
    upper_bound = 75

    random_variable = np.random.normal(mean, std_dev)  # Generates a random number from a normal distribution
    random_dist_ahead = max(lower_bound, min(upper_bound, random_variable))  # Ensure the value is within the desired range
    dist_ahead_inTiles = random_dist_ahead * 10 # 0.1 m/tile

    mean = 6.3
    std_dev = 1.9
    lower_bound = 0.6
    upper_bound = 12

    random_variable = np.random.normal(mean, std_dev)  # Generates a random number from a normal distribution
    random_dist_LR = max(lower_bound, min(upper_bound, random_variable))  # Ensure the value is within the desired range
    dist_LR_inTiles = random_dist_LR * 10 # 0.1 m/tile

    vehicle_speed = math.ceil(dist_ahead_inTiles/3)
    ped_speed = math.ceil(dist_LR_inTiles/3)

    possiblePedPos = [(850, 0), (850, 269)]
    pedPos = possiblePedPos[random.randint(0, 1)]
    if pedPos == (850, 0):
        vehiclePos = (pedPos[0] - dist_ahead_inTiles, pedPos[1] + dist_LR_inTiles)
    else:
        vehiclePos = (pedPos[0] - dist_ahead_inTiles, pedPos[1] - dist_LR_inTiles)

    p1 = SimplePedAgent(id=1, position=pedPos, direction=Direction.South if pedPos == (850, 0) else Direction.North, maxSpeed=1000, speed=ped_speed)
    v1 = SimpleVehicle(1, (vehiclePos[0] - 100, vehiclePos[1] - 25), (vehiclePos[0], vehiclePos[1] + 25), direction=Direction.East, maxSpeed=1000, speed=vehicle_speed, inRoad=1, inLane=1)

    env.addVehicleAgent(v1)
    env.addPedAgent(p1)

    for i in range(2):

        obs, reward, done, info = env.step(None)
        
        if done:
            "Reached the goal"
            break

        # env.render()

        if i % 10 == 0:
            logging.info(f"Completed step {i+1}")

        # time.sleep(0.5)
    
    pedPosX = []
    pedPosY = []
    vehPosX = []
    vehPosY = []

    pedPositions, vehPositions = metricCollector.getPositions()
    ped = env.getPedAgents()[0]
    veh = env.getVehicleAgents()[0]
    pedPosX = [pos[0] for pos in pedPositions[ped]]
    pedPosY = [pos[1] for pos in pedPositions[ped]]
    vehPosX = [pos[0] for pos in vehPositions[veh]["bottomRight"]]
    vehPosY = [(pos[1] - 25) for pos in vehPositions[veh]["bottomRight"]]

    pedDict = {'pedX' : pedPosX, 'pedY' : pedPosY, 'vehX' : vehPosX, 'vehY' : vehPosY, 'TTC' : [3, 2, 1]}
    data = pd.DataFrame(pedDict)
    data.to_csv('TTC.csv', index = False)