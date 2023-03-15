from __future__ import annotations

import gymnasium as gym
import numpy as np
from pytest_mock import MockerFixture

from minigrid.benchmark import benchmark
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


def test_benchmark():
    "Test that the benchmark function works for a specific environment"
    env_id = "MiniGrid-Empty-16x16-v0"
    benchmark(env_id, num_resets=10, num_frames=100)


def test_manual_control(mocker: MockerFixture):
    class FakeRandomKeyboardEvent:
        active_actions = ["left", "right", "up", " ", "pageup", "pagedown"]
        reset_action = "backspace"
        close_action = "escape"

        def __init__(self, reset: bool = False, close: bool = False) -> None:
            if reset:
                self.key = self.reset_action
                return
            if close:
                self.key = self.close_action
                return
            self.key = np.random.choice(self.active_actions)

    env_id = "MiniGrid-Empty-16x16-v0"
    env: MiniGridEnv = gym.make(env_id, render_mode="human")
    manual_control = ManualControl(env)

    for i in range(3):  # 3 resets
        manual_control.reset()
        for j in range(20):  # Do 20 steps
            manual_control.key_handler(FakeRandomKeyboardEvent())

        fake_event = FakeRandomKeyboardEvent(reset=True)
        manual_control.key_handler(fake_event)

    # Close the environment
    mocked_quit = mocker.patch("pygame.quit")
    fake_event = FakeRandomKeyboardEvent(close=True)
    manual_control.key_handler(fake_event)
    mocked_quit.assert_called_once()
