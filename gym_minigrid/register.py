from gym.envs.registration import register as gymRegister

envList = []

def register(
    id,
    entry_point,
    reward_threshold=900
):
    assert id.startswith("MiniGrid-")
    assert id not in envList

    # Register the environment with OpenAI gym
    gymRegister(
        id=id,
        entry_point=entry_point,
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    envList.append(id)
