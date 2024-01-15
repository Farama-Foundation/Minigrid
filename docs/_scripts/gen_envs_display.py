from __future__ import annotations

import os

import gymnasium


def create_grid_cell(type_id, env_id, base_path):
    return f"""
            <a href="{base_path}{env_id}">
                <div class="env-grid__cell">
                    <div class="cell__image-container">
                        <img src="/_static/videos/{type_id}/{env_id}.gif">
                    </div>
                    <div class="cell__title">
                        <span>{' '.join(env_id.split('_')).title()}</span>
                    </div>
                </div>
            </a>
    """


def generate_page(env, limit=-1, base_path=""):
    env_type_id = env["id"]
    env_list = env["list"]
    cells = [create_grid_cell(env_type_id, env_id, base_path) for env_id in env_list]
    non_limited_page = limit == -1 or limit >= len(cells)
    if non_limited_page:
        cells = "\n".join(cells)
    else:
        cells = "\n".join(cells[:limit])

    more_btn = (
        """
<a href="./complete_list">
    <button class="more-btn">
        See More Environments
    </button>
</a>
"""
        if not non_limited_page
        else ""
    )
    return f"""
<div class="env-grid">
    {cells}
</div>
{more_btn}
    """


if __name__ == "__main__":
    """
    python gen_envs_display
    """
    type_dict = {}

    for env_spec in gymnasium.envs.registry.values():
        if isinstance(env_spec.entry_point, str):
            # minigrid.envs:Env or minigrid.envs.babyai:Env
            split = env_spec.entry_point.split(".")
            # ignore minigrid.envs.env_type:Env
            env_module = split[0]
            env_name = split[-1].split(":")[-1]
            env_type = env_module if len(split) == 2 else split[-1].split(":")[0]

            if env_module == "minigrid":
                if env_type not in type_dict.keys():
                    type_dict[env_type] = []

                if env_name not in type_dict[env_type]:
                    type_dict[env_type].append(env_name)
        else:
            continue

    for key, value in type_dict.items():
        env_type = key

        page = generate_page({"id": key, "list": value})
        fp = open(
            os.path.join(
                os.path.dirname(__file__), "..", "environments", env_type, "list.html"
            ),
            "w",
            encoding="utf-8",
        )
        fp.write(page)
        fp.close()
