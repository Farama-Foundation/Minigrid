from __future__ import annotations

import numpy as np

from minigrid.envs.wfc.wfclogic import patterns as wfc_patterns
from minigrid.envs.wfc.wfclogic import tiles as wfc_tiles


def test_unique_patterns_2d(img_redmaze) -> None:
    img = img_redmaze
    tile_size = 1
    pattern_width = 2
    _tile_catalog, tile_grid, _code_list, _unique_tiles = wfc_tiles.make_tile_catalog(
        img, tile_size
    )

    (
        _patterns_in_grid,
        pattern_contents_list,
        patch_codes,
    ) = wfc_patterns.unique_patterns_2d(tile_grid, pattern_width, True)
    assert patch_codes[1][2] == 4867810695119132864
    assert pattern_contents_list[7][1][1] == 8253868773529191888


def test_make_pattern_catalog(img_redmaze) -> None:
    img = img_redmaze
    tile_size = 1
    pattern_width = 2
    _tile_catalog, tile_grid, _code_list, _unique_tiles = wfc_tiles.make_tile_catalog(
        img, tile_size
    )

    (
        pattern_catalog,
        pattern_weights,
        pattern_list,
        _pattern_grid,
    ) = wfc_patterns.make_pattern_catalog(tile_grid, pattern_width)
    assert pattern_weights[-6150964001204120324] == 1
    assert pattern_list[3] == 2800765426490226432
    assert pattern_catalog[5177878755649963747][0][1] == -8754995591521426669


def test_pattern_to_tile(img_redmaze) -> None:
    img = img_redmaze
    tile_size = 1
    pattern_width = 2
    _tile_catalog, tile_grid, _code_list, _unique_tiles = wfc_tiles.make_tile_catalog(
        img, tile_size
    )

    (
        pattern_catalog,
        _pattern_weights,
        _pattern_list,
        pattern_grid,
    ) = wfc_patterns.make_pattern_catalog(tile_grid, pattern_width)
    new_tile_grid = wfc_patterns.pattern_grid_to_tiles(pattern_grid, pattern_catalog)
    assert np.array_equal(tile_grid, new_tile_grid)
