from wfc_tiles import make_tile_catalog
from wfc_patterns import make_pattern_catalog, pattern_grid_to_tiles, make_pattern_catalog_with_rotations
from wfc_adjacency import adjacency_extraction
from wfc_solver import run, makeWave, makeAdj, lexicalLocationHeuristic, lexicalPatternHeuristic, makeWeightedPatternHeuristic
from wfc_visualize import figure_list_of_tiles, figure_false_color_tile_grid, figure_pattern_catalog, render_tiles_to_output, figure_adjacencies
import imageio
import numpy as np



filename = "../images/samples/Red Maze.png"
img = imageio.imread(filename)
tile_size = 1
pattern_width = 2
rotations = 7
output_size = [32, 32]
ground = None

# TODO: generalize this to more than the four cardinal directions
direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))

tile_catalog, tile_grid, code_list, unique_tiles = make_tile_catalog(img, tile_size)
pattern_catalog, pattern_weights, pattern_list, pattern_grid = make_pattern_catalog_with_rotations(tile_grid, pattern_width, rotations)

figure_list_of_tiles(unique_tiles, tile_catalog)
figure_false_color_tile_grid(tile_grid)
figure_pattern_catalog(pattern_catalog, tile_catalog, pattern_weights, pattern_width)

adjacency_relations = adjacency_extraction(pattern_grid, pattern_catalog, direction_offsets)

#print(adjacency_relations)
figure_adjacencies(adjacency_relations, direction_offsets, tile_catalog, pattern_catalog, pattern_width, [tile_size, tile_size])


number_of_patterns = len(pattern_weights)
decode_patterns = dict(enumerate(pattern_list))
encode_patterns = {x: i for i, x in enumerate(pattern_list)}
encode_directions = {j:i for i,j in direction_offsets}

adjacency_list = {}
for i,d in direction_offsets:
    adjacency_list[d] = [set() for i in pattern_weights]
#print(adjacency_list)
for i in adjacency_relations:
    #print(i)
    #print(decode_patterns[i[1]])
    adjacency_list[i[0]][encode_patterns[i[1]]].add(encode_patterns[i[2]])


wave = makeWave(number_of_patterns, output_size[0], output_size[1])
adjacency_matrix = makeAdj(adjacency_list)

#print(adjacency_matrix)

encoded_weights = np.zeros((number_of_patterns), dtype=np.float64)
for w_id, w_val in pattern_weights.items():
    encoded_weights[encode_patterns[w_id]] = w_val

pattern_heuristic =  lexicalPatternHeuristic
pattern_heuristic = makeWeightedPatternHeuristic(encoded_weights)

solution = run(wave.copy(),
               adjacency_matrix,
               locationHeuristic=lexicalLocationHeuristic,
               patternHeuristic=pattern_heuristic,
               periodic=True,
               backtracking=False,
               onChoice=None,
               onBacktrack=None)

#print(solution)
solution_as_ids = np.vectorize(lambda x : decode_patterns[x])(solution)
solution_tile_grid = pattern_grid_to_tiles(solution_as_ids, pattern_catalog)

print(solution_tile_grid)
render_tiles_to_output(solution_tile_grid, tile_catalog, [tile_size, tile_size], "result.png")



def wfc_execute():
    #wave = wfc.wfc_solver.makeWave(3, 3, 4)
    #adjLists = {}
    #adjLists[(+1, 0)] = adjLists[(-1, 0)] = adjLists[(0, +1)] = adjLists[(0, -1)] = [[1], [0], [2]]
    #adjacencies = wfc.wfc_solver.makeAdj(adjLists)

    #result = run(wave,
    #             adjacencies,
    #             locationHeuristic=wfc.wfc_solver.lexicalLocationHeuristic,
    #             patternHeuristic=wfc.wfc_solver.lexicalPatternHeuristic,
    #             periodic=False)
    #print(result)

    import imageio
    filename = "images/samples/Red Maze.png"
    img = imageio.imread(filename)




