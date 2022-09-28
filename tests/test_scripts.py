import gymnasium as gym
import numpy as np

from minigrid.benchmark import benchmark
from minigrid.manual_control import key_handler, reset
from minigrid.utils.window import Window


def test_benchmark():
    "Test that the benchmark function works for a specific environment"
    env_id = "MiniGrid-Empty-16x16-v0"
    benchmark(env_id, num_resets=10, num_frames=100)


def test_window():
    "Testing the class functions of window.Window. This should locally open a window !"
    title = "testing window"
    window = Window(title)

    img = np.random.rand(100, 100, 3)
    window.show_img(img)

    caption = "testing caption"
    window.set_caption(caption)

    window.show(block=False)

    window.close()


def test_manual_control():
    class FakeRandomKeyboardEvent:
        active_actions = ["left", "right", "up", " ", "pageup", "pagedown"]
        reset_action = "backspace"
        close_action = "escape"

        def __init__(self, active_actions=True, reset_action=False) -> None:
            if active_actions:
                self.key = np.random.choice(self.active_actions)
            elif reset_action:
                self.key = self.reset_action
            else:
                self.key = self.close_action

    env_id = "MiniGrid-Empty-16x16-v0"
    env = gym.make(env_id)
    window = Window(env_id)

    reset(env, window)

    for i in range(3):  # 3 resets
        for j in range(20):  # Do 20 steps
            key_handler(env, window, FakeRandomKeyboardEvent())

        key_handler(
            env,
            window,
            FakeRandomKeyboardEvent(active_actions=False, reset_action=True),
        )

    # Close the environment
    key_handler(
        env, window, FakeRandomKeyboardEvent(active_actions=False, reset_action=False)
    )
