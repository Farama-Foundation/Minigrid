from __future__ import annotations

import imageio  # type: ignore
import numpy as np
from tests.conftest import Resources
from wfc import wfc_patterns
from wfc import wfc_tiles


def test_unique_patterns_2d(resources: Resources) -> None:
    filename = resources.get_image("samples/Red Maze.png")
    img = imageio.imread(filename)
    tile_size = 1
    pattern_width = 2
    _tile_catalog, tile_grid, _code_list, _unique_tiles = wfc_tiles.make_tile_catalog(img, tile_size)

    _patterns_in_grid, pattern_contents_list, patch_codes = wfc_patterns.unique_patterns_2d(
        tile_grid, pattern_width, True
    )
    assert patch_codes[1][2] == 4867810695119132864
    assert pattern_contents_list[7][1][1] == 8253868773529191888


def test_make_pattern_catalog(resources: Resources) -> None:
    filename = resources.get_image("samples/Red Maze.png")
    img = imageio.imread(filename)
    tile_size = 1
    pattern_width = 2
    _tile_catalog, tile_grid, _code_list, _unique_tiles = wfc_tiles.make_tile_catalog(img, tile_size)

    pattern_catalog, pattern_weights, pattern_list, _pattern_grid = wfc_patterns.make_pattern_catalog(
        tile_grid, pattern_width
    )
    assert pattern_weights[-6150964001204120324] == 1
    assert pattern_list[3] == 2800765426490226432
    assert pattern_catalog[5177878755649963747][0][1] == -8754995591521426669


def test_pattern_to_tile(resources: Resources) -> None:
    filename = resources.get_image("samples/Red Maze.png")
    img = imageio.imread(filename)
    tile_size = 1
    pattern_width = 2
    _tile_catalog, tile_grid, _code_list, _unique_tiles = wfc_tiles.make_tile_catalog(img, tile_size)

    pattern_catalog, _pattern_weights, _pattern_list, pattern_grid = wfc_patterns.make_pattern_catalog(
        tile_grid, pattern_width
    )
    new_tile_grid = wfc_patterns.pattern_grid_to_tiles(pattern_grid, pattern_catalog)
    assert np.array_equal(tile_grid, new_tile_grid)
