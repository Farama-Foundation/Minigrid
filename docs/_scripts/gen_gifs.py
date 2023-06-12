from __future__ import annotations

import os
import re

import gymnasium
from PIL import Image
from tqdm import tqdm

# snake to camel case: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case # noqa: E501
pattern = re.compile(r"(?<!^)(?=[A-Z])")

# how many steps to record an env for
LENGTH = 300

output_dir = os.path.join(os.path.dirname(__file__), "..", "_static", "videos")
os.makedirs(output_dir, exist_ok=True)

# Some environments have multiple versions
# For example, KeyCorridorEnv -> KeyCorridorS3R1, KeyCorridorS3R2, KeyCorridorS3R3, etc
# We only want one as an example
envs_completed = []

# iterate through all envspecs
for env_spec in tqdm(gymnasium.envs.registry.values()):
    # minigrid.envs:Env or minigrid.envs.babyai:Env
    split = env_spec.entry_point.split(".")
    # ignore minigrid.envs.env_type:Env
    env_module = split[0]
    env_name = split[-1].split(":")[-1]
    env_type = env_module if len(split) == 2 else split[-1].split(":")[0]

    if env_module == "minigrid" and env_name not in envs_completed:
        os.makedirs(os.path.join(output_dir, env_type), exist_ok=True)
        path = os.path.join(output_dir, env_type, env_name + ".gif")
        envs_completed.append(env_name)

        # try catch in case missing some installs
        try:
            env = gymnasium.make(env_spec.id, render_mode="rgb_array")
            # the gymnasium needs to be rgb renderable
            if not ("rgb_array" in env.metadata["render_modes"]):
                continue

            # obtain and save LENGTH frames worth of steps
            frames = []
            t = 0
            while True:
                state, info = env.reset()
                terminated, truncated = False, False
                while not (terminated or truncated) and len(frames) <= LENGTH:

                    frame = env.render()
                    frames.append(Image.fromarray(frame))
                    action = env.action_space.sample()

                    # Avoid to much movement
                    if t % 10 == 0:
                        state_next, reward, terminated, truncated, info = env.step(
                            action
                        )
                    t += 1

                if len(frames) > LENGTH:
                    break

            env.close()

            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
            )
            print("Saved: " + env_name)

        except BaseException as e:
            print("ERROR", e)
            continue
