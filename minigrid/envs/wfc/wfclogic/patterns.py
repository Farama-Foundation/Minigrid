"Extract patterns from grids of tiles. Implementation based on https://github.com/ikarth/wfc_2019f"
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from minigrid.envs.wfc.wfclogic.utilities import hash_downto

logger = logging.getLogger(__name__)


def unique_patterns_2d(
    agrid: NDArray[np.int64], ksize: int, periodic_input: bool
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    assert ksize >= 1
    if periodic_input:
        agrid = np.pad(
            agrid,
            ((0, ksize - 1), (0, ksize - 1), *(((0, 0),) * (len(agrid.shape) - 2))),
            mode="wrap",
        )
    else:
        # TODO: implement non-wrapped image handling
        # a = np.pad(a, ((0,k-1),(0,k-1),*(((0,0),)*(len(a.shape)-2))), mode='constant', constant_values=None)
        agrid = np.pad(
            agrid,
            ((0, ksize - 1), (0, ksize - 1), *(((0, 0),) * (len(agrid.shape) - 2))),
            mode="wrap",
        )

    patches: NDArray[np.int64] = np.lib.stride_tricks.as_strided(
        agrid,
        (
            agrid.shape[0] - ksize + 1,
            agrid.shape[1] - ksize + 1,
            ksize,
            ksize,
            *agrid.shape[2:],
        ),
        agrid.strides[:2] + agrid.strides[:2] + agrid.strides[2:],
        writeable=False,
    )
    patch_codes = hash_downto(patches, 2)
    uc, ui = np.unique(patch_codes, return_index=True)
    locs = np.unravel_index(ui, patch_codes.shape)
    up: NDArray[np.int64] = patches[locs[0], locs[1]]
    ids: NDArray[np.int64] = np.vectorize(
        {code: ind for ind, code in enumerate(uc)}.get
    )(patch_codes)
    return ids, up, patch_codes


def unique_patterns_brute_force(grid, size, periodic_input):
    padded_grid = np.pad(
        grid,
        ((0, size - 1), (0, size - 1), *(((0, 0),) * (len(grid.shape) - 2))),
        mode="wrap",
    )
    patches = []
    for x in range(grid.shape[0]):
        row_patches = []
        for y in range(grid.shape[1]):
            row_patches.append(
                np.ndarray.tolist(padded_grid[x : x + size, y : y + size])
            )
        patches.append(row_patches)
    patches = np.array(patches)
    patch_codes = hash_downto(patches, 2)
    uc, ui = np.unique(patch_codes, return_index=True)
    locs = np.unravel_index(ui, patch_codes.shape)
    up = patches[locs[0], locs[1]]
    ids = np.vectorize({c: i for i, c in enumerate(uc)}.get)(patch_codes)
    return ids, up


def make_pattern_catalog(
    tile_grid: NDArray[np.int64], pattern_width: int, input_is_periodic: bool = True
) -> tuple[dict[int, NDArray[np.int64]], Counter, NDArray[np.int64], NDArray[np.int64]]:
    """Returns a pattern catalog (dictionary of pattern hashes to constituent tiles),
    an ordered list of pattern weights, and an ordered list of pattern contents."""
    _patterns_in_grid, pattern_contents_list, patch_codes = unique_patterns_2d(
        tile_grid, pattern_width, input_is_periodic
    )
    dict_of_pattern_contents: dict[int, NDArray[np.int64]] = {}
    for pat_idx in range(pattern_contents_list.shape[0]):
        p_hash = hash_downto(pattern_contents_list[pat_idx], 0)
        dict_of_pattern_contents.update({p_hash.item(): pattern_contents_list[pat_idx]})
    pattern_frequency = Counter(hash_downto(pattern_contents_list, 1))
    return (
        dict_of_pattern_contents,
        pattern_frequency,
        hash_downto(pattern_contents_list, 1),
        patch_codes,
    )


def identity_grid(grid):
    """Do nothing to the grid"""
    # return np.array([[7,5,5,5],[5,0,0,0],[5,0,1,0],[5,0,0,0]])
    return grid


def reflect_grid(grid):
    """Reflect the grid left/right"""
    return np.fliplr(grid)


def rotate_grid(grid):
    """Rotate the grid"""
    return np.rot90(grid, axes=(1, 0))


def make_pattern_catalog_with_rotations(
    tile_grid: NDArray[np.int64],
    pattern_width: int,
    rotations: int = 7,
    input_is_periodic: bool = True,
) -> tuple[dict[int, NDArray[np.int64]], Counter, NDArray[np.int64], NDArray[np.int64]]:
    rotated_tile_grid = tile_grid.copy()
    merged_dict_of_pattern_contents: dict[int, NDArray[np.int64]] = {}
    merged_pattern_frequency: Counter = Counter()
    merged_pattern_contents_list: NDArray[np.int64] | None = None
    merged_patch_codes: NDArray[np.int64] | None = None

    def _make_catalog() -> None:
        nonlocal rotated_tile_grid, merged_dict_of_pattern_contents, merged_pattern_contents_list, merged_pattern_frequency, merged_patch_codes
        (
            dict_of_pattern_contents,
            pattern_frequency,
            pattern_contents_list,
            patch_codes,
        ) = make_pattern_catalog(rotated_tile_grid, pattern_width, input_is_periodic)
        merged_dict_of_pattern_contents.update(dict_of_pattern_contents)
        merged_pattern_frequency.update(pattern_frequency)
        if merged_pattern_contents_list is None:
            merged_pattern_contents_list = pattern_contents_list.copy()
        else:
            merged_pattern_contents_list = np.unique(
                np.concatenate((merged_pattern_contents_list, pattern_contents_list))
            )
        if merged_patch_codes is None:
            merged_patch_codes = patch_codes.copy()

    counter = 0
    grid_ops = [
        identity_grid,
        reflect_grid,
        rotate_grid,
        reflect_grid,
        rotate_grid,
        reflect_grid,
        rotate_grid,
        reflect_grid,
    ]
    while counter <= (rotations):
        # logger.debug(rotated_tile_grid.shape)
        # logger.debug(np.array_equiv(reflect_grid(rotated_tile_grid.copy()), rotate_grid(rotated_tile_grid.copy())))

        # logger.debug(counter)
        # logger.debug(grid_ops[counter].__name__)
        rotated_tile_grid = grid_ops[counter](rotated_tile_grid.copy())
        # logger.debug(rotated_tile_grid)
        # logger.debug("---")
        _make_catalog()
        counter += 1

    # assert False
    assert merged_pattern_contents_list is not None
    assert merged_patch_codes is not None
    return (
        merged_dict_of_pattern_contents,
        merged_pattern_frequency,
        merged_pattern_contents_list,
        merged_patch_codes,
    )


def pattern_grid_to_tiles(
    pattern_grid: NDArray[np.int64], pattern_catalog: Mapping[int, NDArray[np.int64]]
) -> NDArray[np.int64]:
    anchor_x = 0
    anchor_y = 0

    def pattern_to_tile(pattern: int) -> Any:
        # if isinstance(pattern, list):
        #     ptrns = []
        #     for p in pattern:
        #         logger.debug(p)
        #         ptrns.push(pattern_to_tile(p))
        #     logger.debug(ptrns)
        #     assert False
        #     return ptrns
        return pattern_catalog[pattern][anchor_x][anchor_y]

    return np.vectorize(pattern_to_tile)(pattern_grid)
