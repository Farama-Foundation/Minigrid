"""Observation options for multi-agent environments, isolated from MiniGrid."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from gymnasium import spaces

from minigrid.core.constants import DIR_TO_VEC, OBJECT_TO_IDX


@dataclass(frozen=True)
class ObservationSettings:
    style: str = "symbolic"
    view_size: int = 7
    tile_size: int = 5
    view_offset: int = 0
    see_through_walls: bool = False
    observe_rewards: bool = False
    observe_position: bool = False
    observe_orientation: bool = False
    hide_item_types: tuple[str, ...] = ()

    def __post_init__(self):
        if self.style not in ("symbolic", "image", "rich"):
            raise ValueError("observation_style must be symbolic, image or rich")
        for name, value, minimum in (
            ("view_tile_size", self.tile_size, 1),
            ("view_offset", self.view_offset, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.view_offset >= self.view_size:
            raise ValueError("view_offset must be smaller than agent_view_size")
        for name in (
            "see_through_walls",
            "observe_rewards",
            "observe_position",
            "observe_orientation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if isinstance(self.hide_item_types, str) or any(
            name not in OBJECT_TO_IDX for name in self.hide_item_types
        ):
            raise ValueError("hide_item_types must contain MiniGrid object type names")

    def space(self, symbolic_space):
        if self.style == "symbolic":
            return copy.deepcopy(symbolic_space)
        image = spaces.Box(
            0, 255, (self.view_size * self.tile_size,) * 2 + (3,), np.uint8
        )
        if self.style == "image":
            return image
        fields = {"pov": image}
        if self.observe_rewards:
            fields["reward"] = spaces.Box(-np.inf, np.inf, (), np.float32)
        if self.observe_position:
            fields["position"] = spaces.Box(0, 1, (2,), np.float32)
        if self.observe_orientation:
            fields["orientation"] = spaces.Discrete(4)
        return spaces.Dict(fields)

    def observe(self, env, agent):
        state = env.agent_states[agent]
        active = agent in env._visible_agents and state.position is not None
        if active:
            world = env._observation_grid(agent)
            view = env._set_view(agent, world)
            # Shift the viewing window backwards without changing world state.
            dx, dy = DIR_TO_VEC[state.direction]
            view.agent_pos = (
                state.position[0] - self.view_offset * int(dx),
                state.position[1] - self.view_offset * int(dy),
            )
            x, y, _, _ = view.get_view_exts()
            grid = world.slice(x, y, self.view_size, self.view_size)
            for _ in range(state.direction + 1):
                grid = grid.rotate_left()
            own = (self.view_size // 2, self.view_size - 1 - self.view_offset)
            visible = (
                np.ones((self.view_size, self.view_size), dtype=bool)
                if self.see_through_walls
                else grid.process_vis(own)
            )
            # Hiding objects changes observations, not collision/visibility rules.
            for i, cell in enumerate(grid.grid):
                if cell is not None and cell.type in self.hide_item_types:
                    grid.grid[i] = None
            grid.set(*own, state.carrying)
            encoded = grid.encode(visible)
            if self.style != "symbolic":
                # Paint hidden tiles black; highlight_mask alone would leak them.
                for x in range(grid.width):
                    for y in range(grid.height):
                        if not visible[x, y]:
                            grid.set(x, y, None)
                pixels = grid.render(self.tile_size, agent_pos=own, agent_dir=3)
                pixel_mask = np.repeat(
                    np.repeat(visible.T, self.tile_size, 0), self.tile_size, 1
                )
                pixels[~pixel_mask] = 0
        else:
            encoded = np.zeros((self.view_size, self.view_size, 3), dtype=np.uint8)
            pixels = np.zeros(
                (self.view_size * self.tile_size,) * 2 + (3,), dtype=np.uint8
            )
        if self.style == "symbolic":
            return {
                "image": encoded,
                "direction": state.direction,
                "mission": env._mission(),
            }
        if self.style == "image":
            return pixels
        result = {"pov": pixels}
        if self.observe_rewards:
            result["reward"] = np.asarray(
                env._last_rewards.get(agent, 0), dtype=np.float32
            )
        if self.observe_position:
            result["position"] = (
                np.asarray(state.position or (0, 0), dtype=np.float32) / env.size
            )
        if self.observe_orientation:
            result["orientation"] = state.direction
        return result
