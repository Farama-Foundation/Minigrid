"""Convert input data to adjacency information. Implementation based on https://github.com/ikarth/wfc_2019f"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def adjacency_extraction(
    pattern_grid: NDArray[np.int64],
    pattern_catalog: dict[int, NDArray[np.int64]],
    direction_offsets: list[tuple[int, tuple[int, int]]],
    pattern_size: tuple[int, int] = (2, 2),
) -> list[tuple[tuple[int, int], int, int]]:
    """Takes a pattern grid and returns a list of all of the legal adjacencies found in it."""

    def is_valid_overlap_xy(
        adjacency_direction: tuple[int, int], pattern_1: int, pattern_2: int
    ) -> bool:
        """Given a direction and two patterns, find the overlap of the two patterns
        and return True if the intersection matches."""
        dimensions = (1, 0)
        not_a_number = -1

        # TODO: can probably speed this up by using the right slices, rather than rolling the whole pattern...
        shifted = np.roll(
            np.pad(
                pattern_catalog[pattern_2],
                max(pattern_size),
                mode="constant",
                constant_values=not_a_number,
            ),
            adjacency_direction,
            dimensions,
        )
        compare = shifted[
            pattern_size[0] : pattern_size[0] + pattern_size[0],
            pattern_size[1] : pattern_size[1] + pattern_size[1],
        ]

        left = max(0, 0, +adjacency_direction[0])
        right = min(pattern_size[0], pattern_size[0] + adjacency_direction[0])
        top = max(0, 0 + adjacency_direction[1])
        bottom = min(pattern_size[1], pattern_size[1] + adjacency_direction[1])
        a = pattern_catalog[pattern_1][top:bottom, left:right]
        b = compare[top:bottom, left:right]
        res = np.array_equal(a, b)
        return res

    pattern_list = list(pattern_catalog.keys())
    legal = []
    for pattern_1 in pattern_list:
        for pattern_2 in pattern_list:
            for _direction_index, direction in direction_offsets:
                if is_valid_overlap_xy(direction, pattern_1, pattern_2):
                    legal.append((direction, pattern_1, pattern_2))
    return legal
