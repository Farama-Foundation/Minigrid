"""Convert input data to adjacency information"""

import imageio
from wfc import wfc_tiles
from wfc import wfc_patterns
from wfc import wfc_adjacency


def test_adjacency_extraction(resources):
    # TODO: generalize this to more than the four cardinal directions
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))

    filename = resources.get_image("samples/Red Maze.png")
    img = imageio.imread(filename)
    tile_size = 1
    pattern_width = 2
    rotations = 0
    _tile_catalog, tile_grid, _code_list, _unique_tiles = wfc_tiles.make_tile_catalog(img, tile_size)
    pattern_catalog, _pattern_weights, _pattern_list, pattern_grid = wfc_patterns.make_pattern_catalog(
        tile_grid, pattern_width, rotations
    )
    adjacency_relations = wfc_adjacency.adjacency_extraction(
        pattern_grid, pattern_catalog, direction_offsets
    )
    assert ((0, -1), -6150964001204120324, -4042134092912931260) in adjacency_relations
    assert ((-1, 0), -4042134092912931260, 3069048847358774683) in adjacency_relations
    assert ((1, 0), -3950451988873469076, -3950451988873469076) in adjacency_relations
    assert ((-1, 0), -3950451988873469076, -3950451988873469076) in adjacency_relations
    assert ((0, 1), -3950451988873469076, 3336256675067683735) in adjacency_relations
    assert (
        not ((0, -1), -3950451988873469076, -3950451988873469076) in adjacency_relations
    )
    assert (
        not ((0, 1), -3950451988873469076, -3950451988873469076) in adjacency_relations
    )
