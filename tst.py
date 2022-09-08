import gym 

import gym_minigrid

env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode="human")
env.reset() 

for _ in range(1000):
    _, _, terminated, _, _ = env.step(env.action_space.sample())
    # env.render()
    while terminated:
        env.reset()
