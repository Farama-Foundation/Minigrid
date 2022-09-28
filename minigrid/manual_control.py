#!/usr/bin/env python3

import gymnasium as gym

from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


def redraw(window, img):
    window.show_img(img)


def reset(env, window, seed=None):
    env.reset(seed=seed)

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    img = env.get_frame()

    redraw(window, img)


def step(env, window, action):
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={env.step_count}, reward={reward:.2f}")

    if terminated:
        print("terminated!")
        reset(env, window)
    elif truncated:
        print("truncated!")
        reset(env, window)
    else:
        img = env.get_frame()
        redraw(window, img)


def key_handler(env, window, event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(env, window)
        return

    if event.key == "left":
        step(env, window, env.actions.left)
        return
    if event.key == "right":
        step(env, window, env.actions.right)
        return
    if event.key == "up":
        step(env, window, env.actions.forward)
        return

    # Spacebar
    if event.key == " ":
        step(env, window, env.actions.toggle)
        return
    if event.key == "pageup":
        step(env, window, env.actions.pickup)
        return
    if event.key == "pagedown":
        step(env, window, env.actions.drop)
        return

    if event.key == "enter":
        step(env, window, env.actions.done)
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-MultiRoom-N6-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    env = gym.make(
        args.env,
        tile_size=args.tile_size,
    )

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window("minigrid - " + args.env)
    window.reg_key_handler(lambda event: key_handler(env, window, event))

    seed = None if args.seed == -1 else args.seed
    reset(env, window, seed)

    # Blocking event loop
    window.show(block=True)
