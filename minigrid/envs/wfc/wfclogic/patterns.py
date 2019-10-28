"Extract patterns from grids of tiles."
from wfc_utilities import hash_downto
from collections import Counter
import numpy as np

def unique_patterns_2d(agrid, ksize, periodic_input):
    assert ksize >= 1
    if periodic_input:
        agrid = np.pad(agrid, ((0, ksize - 1), (0, ksize - 1), *(((0, 0), )*(len(agrid.shape) - 2))), mode='wrap')
    else:
        # TODO: implement non-wrapped image handling
        #a = np.pad(a, ((0,k-1),(0,k-1),*(((0,0),)*(len(a.shape)-2))), mode='constant', constant_values=None)
        agrid = np.pad(agrid, ((0, ksize - 1), (0, ksize - 1), *(((0, 0), )*(len(agrid.shape) - 2))), mode='wrap')
    
    patches = np.lib.stride_tricks.as_strided(
        agrid,
        (agrid.shape[0] - ksize + 1, agrid.shape[1] - ksize + 1, ksize, ksize, *agrid.shape[2:]),
        agrid.strides[:2] + agrid.strides[:2] + agrid.strides[2:],
        writeable=False)
    patch_codes = hash_downto(patches, 2)
    uc, ui = np.unique(patch_codes, return_index=True)
    locs = np.unravel_index(ui, patch_codes.shape)
    up = patches[locs[0], locs[1]]
    ids = np.vectorize({code: ind for ind, code in enumerate(uc)}.get)(patch_codes)
    return ids, up, patch_codes

def unique_patterns_brute_force(grid, size, periodic_input):
    padded_grid = np.pad(grid, ((0,size-1),(0,size-1),*(((0,0),)*(len(grid.shape)-2))), mode='wrap')
    patches = []
    for x in range(grid.shape[0]):
        row_patches = []
        for y in range(grid.shape[1]):
            row_patches.append(np.ndarray.tolist(padded_grid[x:x+size, y:y+size]))
        patches.append(row_patches)
    patches = np.array(patches)
    patch_codes = hash_downto(patches,2)
    uc, ui = np.unique(patch_codes, return_index=True)
    locs = np.unravel_index(ui, patch_codes.shape)
    up = patches[locs[0],locs[1]]
    ids = np.vectorize({c: i for i,c in enumerate(uc)}.get)(patch_codes)
    return ids, up


def make_pattern_catalog(tile_grid, pattern_width, rotations=8, input_is_periodic=True):
    """Returns a pattern catalog (dictionary of pattern hashes to consituent tiles), 
an ordered list of pattern weights, and an ordered list of pattern contents."""
    patterns_in_grid, pattern_contents_list, patch_codes = unique_patterns_2d(tile_grid, pattern_width, input_is_periodic)
    dict_of_pattern_contents = {}
    for pat_idx in range(pattern_contents_list.shape[0]):
        p_hash = hash_downto(pattern_contents_list[pat_idx], 0)
        dict_of_pattern_contents.update({np.asscalar(p_hash) : pattern_contents_list[pat_idx]})
    pattern_frequency = Counter(hash_downto(pattern_contents_list, 1))
    return dict_of_pattern_contents, pattern_frequency, hash_downto(pattern_contents_list, 1)


def test_unique_patterns_2d():
    from wfc_tiles import make_tile_catalog
    import imageio
    filename = "images/samples/Red Maze.png"
    img = imageio.imread(filename)
    tile_size = 1
    pattern_width = 2
    rotations = 0
    tile_catalog, tile_grid, code_list, unique_tiles = make_tile_catalog(img, tile_size)

    patterns_in_grid, pattern_contents_list, patch_codes = unique_patterns_2d(tile_grid, pattern_width, True)
    #print(patterns_in_grid)
    #print(pattern_contents_list)
    #print(patch_codes)
    assert(patch_codes[1][2] == 4867810695119132864)
    assert(pattern_contents_list[7][1][1] == 8253868773529191888)
    
    
def test_make_pattern_catalog():
    from wfc_tiles import make_tile_catalog
    import imageio
    filename = "images/samples/Red Maze.png"
    img = imageio.imread(filename)
    tile_size = 1
    pattern_width = 2
    rotations = 0
    tile_catalog, tile_grid, code_list, unique_tiles = make_tile_catalog(img, tile_size)

    pattern_catalog, pattern_weights, pattern_list = make_pattern_catalog(tile_grid, pattern_width, rotations)
    #print("---")
    #print(pattern_catalog)
    #print(pattern_weights)
    #print(pattern_list)
    assert(pattern_weights[-6150964001204120324] == 1)
    assert(pattern_list[3] == 2800765426490226432)
    assert(pattern_catalog[5177878755649963747][0][1] == -8754995591521426669)

if __name__ == "__main__":
    test_unique_patterns_2d()
    test_make_pattern_catalog()
