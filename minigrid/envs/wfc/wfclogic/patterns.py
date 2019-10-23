from wfc_utilities import hash_downto
from wfc_tiles import make_tile_catalog
import numpy as np

def unique_patterns_2d(a, k, periodic_input):
    assert k >= 1
    if periodic_input:
        a = np.pad(a, ((0,k-1),(0,k-1),*(((0,0),)*(len(a.shape)-2))), mode='wrap')
    else:
        # TODO: implement non-wrapped image handling
        #a = np.pad(a, ((0,k-1),(0,k-1),*(((0,0),)*(len(a.shape)-2))), mode='constant', constant_values=None)
        a = np.pad(a, ((0,k-1),(0,k-1),*(((0,0),)*(len(a.shape)-2))), mode='wrap')
    
    patches = np.lib.stride_tricks.as_strided(
        a,
        (a.shape[0]-k+1,a.shape[1]-k+1,k,k,*a.shape[2:]),
        a.strides[:2] + a.strides[:2] + a.strides[2:],
        writeable=False)
    patch_codes = hash_downto(patches,2)
    uc, ui = np.unique(patch_codes, return_index=True)
    locs = np.unravel_index(ui, patch_codes.shape)
    up = patches[locs[0],locs[1]]
    ids = np.vectorize({c: i for i,c in enumerate(uc)}.get)(patch_codes)
    return ids, up, patch_codes

def make_pattern_catalog(tile_grid, pattern_width, rotations=8, input_is_periodic=True):
    """Returns a pattern catalog (dictionary of pattern hashes to consituent tiles), 
an ordered list of pattern weights, and an ordered list of pattern contents."""
    pattern_overall_id_count = 0
    
    patterns_in_grid, pattern_contents_list, patch_codes = unique_patterns_2d(tile_grid, pattern_width, input_is_periodic)

    ordered_list_of_pattern_hashes = hash_downto(pattern_contents_list, 1)
    dict_of_pattern_contents = {}
    dict_of_pattern_ids = {}
    for pat_idx in range(pattern_contents_list.shape[0]):
        p_hash = hash_downto(pattern_contents_list[pat_idx], 0)
        dict_of_pattern_contents.update({p_hash : pattern_contents_list[pat_idx]})
        dict_of_pattern_ids.update({pattern_overall_id_count: p_hash})
        
        
        
        
    print(patch_codes)
    print("patterns_in_grid")
    print(patterns_in_grid)
    print("pattern_list")
    print(pattern_contents_list)
    print("##")
    print(hash_downto(pattern_contents_list, 1))

    pattern_catalog = {}
    pattern_weights = []
    pattern_list = []
    return pattern_catalog, pattern_weights, pattern_list


import imageio
filename = "images/samples/Red Maze.png"
img = imageio.imread(filename)
tile_size = 1
pattern_width = 2
rotations = 0
tile_catalog, tile_grid, code_list, unique_tiles = make_tile_catalog(img, tile_size)

pattern_catalog, pattern_weights, pattern_list = make_pattern_catalog(tile_grid, pattern_width, rotations)
print("---")
print(pattern_catalog)
