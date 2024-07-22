"""Utility data and functions for WFC. Implementation based on https://github.com/ikarth/wfc_2019f"""

from __future__ import annotations

import collections
import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

CoordXY = collections.namedtuple("CoordXY", ["x", "y"])
CoordRC = collections.namedtuple("CoordRC", ["row", "column"])


def hash_downto(a: NDArray[np.integer], rank: int, seed=0) -> NDArray[np.int64]:
    state = np.random.RandomState(seed)
    # np_random = np.random.default_rng(seed)
    assert rank < len(a.shape)

    u: NDArray[np.integer] = a.reshape((np.prod(a.shape[:rank], dtype=np.int64), -1))
    v = state.randint(1 - (1 << 63), 1 << 63, np.prod(a.shape[rank:]), dtype=np.int64)
    # v = np_random.integers(1 - (1 << 63), 1 << 63, np.prod(a.shape[rank:]), dtype=np.int64)
    return np.asarray(np.inner(u, v).reshape(a.shape[:rank]), dtype=np.int64)


def find_pattern_center(wfc_ns):
    # wfc_ns.pattern_center = (math.floor((wfc_ns.pattern_width - 1) / 2), math.floor((wfc_ns.pattern_width - 1) / 2))
    wfc_ns.pattern_center = (0, 0)
    return wfc_ns


def tile_grid_to_image(
    tile_grid: NDArray[np.int64],
    tile_catalog: dict[int, NDArray[np.integer]],
    tile_size: tuple[int, int],
    partial: bool = False,
    color_channels: int = 3,
) -> NDArray[np.integer]:
    """
    Takes a tile_grid and transforms it into an image, using the information
    in tile_catalog. We use tile_size to figure out the size the new image
    should be.
    """
    tile_dtype = next(iter(tile_catalog.values())).dtype
    new_img = np.zeros(
        (
            tile_grid.shape[0] * tile_size[0],
            tile_grid.shape[1] * tile_size[1],
            color_channels,
        ),
        dtype=tile_dtype,
    )
    if partial and (len(tile_grid.shape)) > 2:
        # TODO: implement rendering partially completed solution
        # Call tile_grid_to_average() instead.
        assert False
    else:
        for i in range(tile_grid.shape[0]):
            for j in range(tile_grid.shape[1]):
                tile = tile_grid[i, j]
                for u in range(tile_size[0]):
                    for v in range(tile_size[1]):
                        pixel = [200, 0, 200]
                        # If we want to display a partial pattern, it is helpful to
                        # be able to show empty cells.
                        pixel = tile_catalog[tile][u, v]
                        # TODO: will need to change if using an image with more than 3 channels
                        new_img[
                            (i * tile_size[0]) + u, (j * tile_size[1]) + v
                        ] = np.resize(
                            pixel,
                            new_img[
                                (i * tile_size[0]) + u, (j * tile_size[1]) + v
                            ].shape,
                        )
    return new_img
