from __future__ import annotations

import numpy as np
import pytest

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, WorldObj


@pytest.mark.parametrize("color", COLOR_NAMES)
def test_goal_encode_decode_preserves_color(color):
    goal = Goal(color)
    decoded = WorldObj.decode(*goal.encode())

    assert isinstance(decoded, Goal)
    assert decoded.color == color
    assert decoded.encode() == goal.encode()


@pytest.mark.parametrize("color", COLOR_NAMES)
@pytest.mark.parametrize("partially_visible", [False, True])
def test_grid_encode_decode_preserves_goal_color(color, partially_visible):
    grid = Grid(5, 4)
    grid.set(2, 1, Goal(color))
    visibility = np.ones((5, 4), dtype=bool)
    if partially_visible:
        visibility[0, :] = False

    encoded = grid.encode(visibility)
    decoded, decoded_visibility = Grid.decode(encoded)

    np.testing.assert_array_equal(decoded_visibility, visibility)
    np.testing.assert_array_equal(decoded.encode(decoded_visibility), encoded)
    np.testing.assert_array_equal(
        decoded.render(8, (1, 2), agent_dir=0),
        grid.render(8, (1, 2), agent_dir=0),
    )
