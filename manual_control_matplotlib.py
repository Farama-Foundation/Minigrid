import time
import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_minigrid

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        plt.close()
        return

    if event.key == 'left':
        env.step(env.actions.left)
        img = env.render('rgb_array')

        #img = np.zeros(shape=(256,256,3), dtype=np.uint8)
        imshow_obj.set_data(img)
        fig.canvas.draw()
        #plt.show()

        return

env = gym.make('MiniGrid-Empty-8x8-v0')

#env.step(env.actions.left)


t0 = time.time()

for i in range(1000):
    img = env.render('rgb_array')

t1 = time.time()
dt = int(1000 * (t1-t0))

print(dt)

print(img.shape)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', key_handler)

#plt.figure(num='gym-minigrid')
imshow_obj = ax.imshow(img)




plt.show()
