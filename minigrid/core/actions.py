# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5

    # Done completing task
    done = 6
