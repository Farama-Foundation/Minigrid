from gym.envs.registration import register as gym_register

env_list = []

def register(
    id,
    entry_point,
    reward_threshold=0.95
):
    assert id.startswith("MiniGrid-")
    assert id not in env_list

    # Register the environment with OpenAI gym
    gym_register(
        id=id,
        entry_point=entry_point,
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    env_list.append(id)
