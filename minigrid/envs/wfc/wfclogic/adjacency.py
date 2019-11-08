"""Convert input data to adjacency information"""
from collections import Counter
import numpy as np

def adjacency_extraction(pattern_grid, pattern_catalog, direction_offsets, pattern_size=[2, 2]):
    """Takes a pattern grid and returns a list of all of the legal adjacencies found in it."""
    def is_valid_overlap_xy(adjacency_direction, pattern_1, pattern_2):
        """Given a direction and two patterns, find the overlap of the two patterns 
        and return True if the intersection matches."""
        dimensions = (1,0)
        not_a_number = -1

        #TODO: can probably speed this up by using the right slices, rather than rolling the whole pattern...
        shifted = np.roll(np.pad(pattern_catalog[pattern_2], max(pattern_size), mode='constant', constant_values = not_a_number), adjacency_direction, dimensions)
        compare = shifted[pattern_size[0] : pattern_size[0] + pattern_size[0], pattern_size[1] : pattern_size[1] + pattern_size[1]]
        
        left = max(0, 0, + adjacency_direction[0])
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
            for direction_index, direction in direction_offsets:
                if is_valid_overlap_xy(direction, pattern_1, pattern_2):
                    legal.append((direction, pattern_1, pattern_2))
    return legal
    


def test_adjacency_extraction():
    from wfc_tiles import make_tile_catalog
    from wfc_patterns import make_pattern_catalog
    import imageio

    # TODO: generalize this to more than the four cardinal directions
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))


    filename = "images/samples/Red Maze.png"
    img = imageio.imread(filename)
    tile_size = 1
    pattern_width = 2
    rotations = 0
    tile_catalog, tile_grid, code_list, unique_tiles = make_tile_catalog(img, tile_size)
    pattern_catalog, pattern_weights, pattern_list, pattern_grid = make_pattern_catalog(tile_grid, pattern_width, rotations)
    adjacency_relations = adjacency_extraction(pattern_grid, pattern_catalog, direction_offsets)
    assert(((0, -1), -6150964001204120324, -4042134092912931260) in adjacency_relations)
    assert(((-1, 0), -4042134092912931260, 3069048847358774683) in adjacency_relations)
    assert(((1, 0), -3950451988873469076, -3950451988873469076) in adjacency_relations)
    assert(((-1, 0), -3950451988873469076, -3950451988873469076) in adjacency_relations)
    assert(((0, 1), -3950451988873469076, 3336256675067683735) in adjacency_relations)
    assert(not ((0, -1), -3950451988873469076, -3950451988873469076) in adjacency_relations)
    assert(not ((0, 1), -3950451988873469076, -3950451988873469076) in adjacency_relations)


if __name__ == "__main__":
    test_adjacency_extraction()
