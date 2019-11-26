from .wfc_tiles import make_tile_catalog
from .wfc_patterns import make_pattern_catalog, pattern_grid_to_tiles, make_pattern_catalog_with_rotations
from .wfc_adjacency import adjacency_extraction
from .wfc_solver import run, makeWave, makeAdj, lexicalLocationHeuristic, lexicalPatternHeuristic, makeWeightedPatternHeuristic, Contradiction, StopEarly
from .wfc_visualize import figure_list_of_tiles, figure_false_color_tile_grid, figure_pattern_catalog, render_tiles_to_output, figure_adjacencies, visualize_solver, make_solver_visualizers
import imageio
import numpy as np
import time
import os

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import pprofile

def visualize_tiles(unique_tiles, tile_catalog, tile_grid):
    if False:
        figure_list_of_tiles(unique_tiles, tile_catalog)
        figure_false_color_tile_grid(tile_grid)

def visualize_patterns(pattern_catalog, tile_catalog, pattern_weights, pattern_width):
    if False:
        figure_pattern_catalog(pattern_catalog, tile_catalog, pattern_weights, pattern_width)

    

def execute_wfc(filename, tile_size=0, pattern_width=2, rotations=8, output_size=[48,48], ground=None, attempt_limit=1, output_periodic=True, input_periodic=True):
    timecode = f"{time.time()}"
    output_destination = r"./output/"
    input_folder = r"./images/samples/"

    rotations -= 1 # change to zero-based
    
    # Load the image
    img = imageio.imread(input_folder + filename + ".png")
    img = img[:,:,:3] # TODO: handle alpha channels


    # TODO: generalize this to more than the four cardinal directions
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))

    tile_catalog, tile_grid, code_list, unique_tiles = make_tile_catalog(img, tile_size)
    pattern_catalog, pattern_weights, pattern_list, pattern_grid = make_pattern_catalog_with_rotations(tile_grid, pattern_width, input_is_periodic=input_periodic, rotations=rotations)

    #visualize_tiles(unique_tiles, tile_catalog, tile_grid)
    #visualize_patterns(pattern_catalog, tile_catalog, pattern_weights, pattern_width)
    #figure_list_of_tiles(unique_tiles, tile_catalog, output_filename=f"visualization/tilelist_{filename}_{timecode}")
    #figure_false_color_tile_grid(tile_grid, output_filename=f"visualization/tile_falsecolor_{filename}_{timecode}")
    figure_pattern_catalog(pattern_catalog, tile_catalog, pattern_weights, pattern_width, output_filename=f"visualization/pattern_catalog_{filename}_{timecode}")

    profiler = pprofile.Profile()
    with profiler:
        adjacency_relations = adjacency_extraction(pattern_grid, pattern_catalog, direction_offsets, [pattern_width, pattern_width])

    profiler.dump_stats(f"logs/profile_adj_{filename}_{timecode}.txt")



    #print(adjacency_relations)
    figure_adjacencies(adjacency_relations, direction_offsets, tile_catalog, pattern_catalog, pattern_width, [tile_size, tile_size], output_filename=f"visualization/adjacency_{filename}_{timecode}_A")
    #figure_adjacencies(adjacency_relations, direction_offsets, tile_catalog, pattern_catalog, pattern_width, [tile_size, tile_size], output_filename=f"visualization/adjacency_{filename}_{timecode}_B", render_b_first=True)

    print(f"output size: {output_size}\noutput periodic: {output_periodic}")
    number_of_patterns = len(pattern_weights)
    print(f"# patterns: {number_of_patterns}")
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

    print("adjacency")

    ground_list = []
    if not (ground is 0):
        for g in range(abs(ground)):
            ground_list.append(encode_patterns[pattern_grid.flat[-g]])
    if len(ground_list) < 1:
        ground_list = None
        
    
    wave = makeWave(number_of_patterns, output_size[0], output_size[1], ground=ground_list)
    adjacency_matrix = makeAdj(adjacency_list)

    
    #print(adjacency_matrix)

    encoded_weights = np.zeros((number_of_patterns), dtype=np.float64)
    for w_id, w_val in pattern_weights.items():
        encoded_weights[encode_patterns[w_id]] = w_val

    pattern_heuristic =  lexicalPatternHeuristic
    pattern_heuristic = makeWeightedPatternHeuristic(encoded_weights)

    visualize_choice, visualize_wave = make_solver_visualizers(f"{filename}_{timecode}", wave, decode_patterns=decode_patterns, pattern_catalog=pattern_catalog, tile_catalog=tile_catalog, tile_size=[tile_size, tile_size])
    
    
    print("solving...")
    attempts = 0
    while attempts < attempt_limit:
        attempts += 1
        try:
            #profiler = pprofile.Profile()
            #with profiler:
                #with PyCallGraph(output=GraphvizOutput(output_file=f"visualization/pycallgraph_{filename}_{timecode}.png")):
            solution = run(wave.copy(),
                                   adjacency_matrix,
                                   locationHeuristic=lexicalLocationHeuristic,
                                   patternHeuristic=pattern_heuristic,
                                   periodic=output_periodic,
                                   backtracking=False,
                                   onChoice=visualize_choice,
                                   onBacktrack=None,
                                   onObserve=visualize_wave,
                                   onPropagate=None
            )
            #profiler.dump_stats(f"logs/profile_{filename}_{timecode}.txt")
    

            #print(solution)
            solution_as_ids = np.vectorize(lambda x : decode_patterns[x])(solution)
            solution_tile_grid = pattern_grid_to_tiles(solution_as_ids, pattern_catalog)

            print("Solution:")
            #print(solution_tile_grid)
            render_tiles_to_output(solution_tile_grid, tile_catalog, [tile_size, tile_size], output_destination + filename + "_" + timecode + ".png")
            return solution_tile_grid
        except StopEarly:
            print("Skipping...")
            return None
        except Contradiction as e_c:
            print("Contradiction")
        assert False
    return None
