from __future__ import annotations

import pytest
from gymnasium import spaces

from minigrid.core.mission import MissionSpace


@pytest.mark.parametrize("container", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_different_placeholder_modes_compare_unequal(container, reverse):
    left = MissionSpace(lambda color: f"get {color}", [["red", "green"]])
    right = MissionSpace(lambda: "get red")
    if container:
        left, right = spaces.Dict({"mission": left}), spaces.Dict({"mission": right})
    if reverse:
        left, right = right, left
    assert (left == right) is False
    assert (left != right) is True


@pytest.mark.parametrize("placeholders", [None, [["red", "green"]]])
def test_equivalent_mission_spaces_compare_equal(placeholders):
    mission = (
        (lambda: "get red") if placeholders is None else (lambda color: f"get {color}")
    )
    left = MissionSpace(mission, placeholders)
    right = MissionSpace(mission, placeholders)
    assert left == right
    assert left != object()
