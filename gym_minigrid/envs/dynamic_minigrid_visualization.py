import numpy as np
import time

from gym_minigrid.envs.dynamic_minigrid import DynamicMiniGrid

dyn_grid=DynamicMiniGrid()
prob_array = np.array([0.2, 0.2, 0.15, 0.15, 0.3])
n_alterations = 100
sleep_time = 0.5

for _ in range(n_alterations):
    is_solvable = dyn_grid.alter(prob_array)
    dyn_grid.render()
    print(is_solvable)
    time.sleep(sleep_time)

a = 1