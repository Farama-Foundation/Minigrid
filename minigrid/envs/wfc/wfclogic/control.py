from .wfc_tiles import make_tile_catalog
from .wfc_patterns import make_pattern_catalog, pattern_grid_to_tiles, make_pattern_catalog_with_rotations
from .wfc_adjacency import adjacency_extraction
from .wfc_solver import run, makeWave, makeAdj, lexicalLocationHeuristic, lexicalPatternHeuristic, makeWeightedPatternHeuristic, Contradiction, StopEarly, makeEntropyLocationHeuristic, make_global_use_all_patterns, makeRandomLocationHeuristic, makeRandomPatternHeuristic, TimedOut
from .wfc_visualize import figure_list_of_tiles, figure_false_color_tile_grid, figure_pattern_catalog, render_tiles_to_output, figure_adjacencies, visualize_solver, make_solver_visualizers, make_solver_loggers
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


def make_log_stats():
    log_line = 0
    def log_stats(stats, filename):
        nonlocal log_line
        if stats:
            log_line += 1
            with open(filename, 'a', encoding='utf_8') as logf:
                if log_line < 2:
                    for s in stats.keys():
                        print(str(s), end='\t', file=logf)
                    print("", file=logf)
                for s in stats.keys():
                    print(str(stats[s]), end='\t', file=logf)
                print("", file=logf)
    return log_stats


def execute_wfc(filename, tile_size=0, pattern_width=2, rotations=8, output_size=[48,48], ground=None, attempt_limit=1, output_periodic=True, input_periodic=True, loc_heuristic="lexical", choice_heuristic="lexical", visualize=True, global_constraint=False, backtracking=False, log_filename="log", logging=True, global_constraints=None, log_stats_to_output=None):
    timecode = f"{time.time()}"
    time_begin = time.time()
    output_destination = r"./output/"
    input_folder = r"./images/samples/"

    rotations -= 1 # change to zero-based

    input_stats = {"filename": filename, "tile_size": tile_size, "pattern_width": pattern_width, "rotations": rotations, "output_size": output_size, "ground": ground, "attempt_limit": attempt_limit, "output_periodic": output_periodic, "input_periodic": input_periodic, "location heuristic": loc_heuristic, "choice heurisitic": choice_heuristic, "global constraint": global_constraint, "backtracking":backtracking}
    
    # Load the image
    img = imageio.imread(input_folder + filename + ".png")
    img = img[:,:,:3] # TODO: handle alpha channels


    # TODO: generalize this to more than the four cardinal directions
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))

    tile_catalog, tile_grid, code_list, unique_tiles = make_tile_catalog(img, tile_size)
    pattern_catalog, pattern_weights, pattern_list, pattern_grid = make_pattern_catalog_with_rotations(tile_grid, pattern_width, input_is_periodic=input_periodic, rotations=rotations)

    print("pattern catalog")

    #visualize_tiles(unique_tiles, tile_catalog, tile_grid)
    #visualize_patterns(pattern_catalog, tile_catalog, pattern_weights, pattern_width)
    #figure_list_of_tiles(unique_tiles, tile_catalog, output_filename=f"visualization/tilelist_{filename}_{timecode}")
    #figure_false_color_tile_grid(tile_grid, output_filename=f"visualization/tile_falsecolor_{filename}_{timecode}")
    if visualize:
        figure_pattern_catalog(pattern_catalog, tile_catalog, pattern_weights, pattern_width, output_filename=f"visualization/pattern_catalog_{filename}_{timecode}")

    print("profiling adjacency relations")
    adjacency_relations = None

    if False:
        profiler = pprofile.Profile()
        with profiler:
            adjacency_relations = adjacency_extraction(pattern_grid, pattern_catalog, direction_offsets, [pattern_width, pattern_width])
        profiler.dump_stats(f"logs/profile_adj_{filename}_{timecode}.txt")
    else:
        adjacency_relations = adjacency_extraction(pattern_grid, pattern_catalog, direction_offsets, [pattern_width, pattern_width])

    print("adjacency_relations")

    if visualize:
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

    print(f"adjacency: {len(adjacency_list)}")

    time_adjacency = time.time()

    ### Ground ###

    ground_list = []
    if not (ground is 0):
        ground_list = np.vectorize(lambda x : encode_patterns[x])(pattern_grid.flat[(ground - 1):])
    if len(ground_list) < 1:
        ground_list = None

    if not (ground_list is None):
        ground_catalog = {encode_patterns[k]:v for k,v in pattern_catalog.items() if encode_patterns[k] in ground_list}
        if visualize:
            figure_pattern_catalog(ground_catalog, tile_catalog, pattern_weights, pattern_width, output_filename=f"visualization/patterns_ground_{filename}_{timecode}")
        
    wave = makeWave(number_of_patterns, output_size[0], output_size[1], ground=ground_list)
    adjacency_matrix = makeAdj(adjacency_list)

    ### Heuristics ###
    
    encoded_weights = np.zeros((number_of_patterns), dtype=np.float64)
    for w_id, w_val in pattern_weights.items():
        encoded_weights[encode_patterns[w_id]] = w_val
    choice_random_weighting = np.random.random(wave.shape[1:]) * 0.1
    
    pattern_heuristic =  lexicalPatternHeuristic
    if choice_heuristic == "weighted":
        pattern_heuristic = makeWeightedPatternHeuristic(encoded_weights)
    if choice_heuristic == "random":
        pattern_heuristic = makeRandomPatternHeuristic(encoded_weights)
    
    location_heuristic = lexicalLocationHeuristic
    if loc_heuristic == "entropy":
        location_heuristic = makeEntropyLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "random":
        location_heuristic = makeRandomLocationHeuristic(choice_random_weighting)

        
    ### Visualization ###

    visualize_choice, visualize_wave, visualize_backtracking, visualize_propagate, visualize_final, visualize_after = None, None, None, None, None, None
    if visualize:
        visualize_choice, visualize_wave, visualize_backtracking, visualize_propagate, visualize_final, visualize_after = make_solver_visualizers(f"{filename}_{timecode}", wave, decode_patterns=decode_patterns, pattern_catalog=pattern_catalog, tile_catalog=tile_catalog, tile_size=[tile_size, tile_size])
    if logging:
        visualize_choice, visualize_wave, visualize_backtracking, visualize_propagate, visualize_final, visualize_after = make_solver_loggers(f"{filename}_{timecode}", input_stats.copy())
    if logging and visualize:
        vis = make_solver_visualizers(f"{filename}_{timecode}", wave, decode_patterns=decode_patterns, pattern_catalog=pattern_catalog, tile_catalog=tile_catalog, tile_size=[tile_size, tile_size])
        log = make_solver_loggers(f"{filename}_{timecode}", input_stats.copy())

        def visfunc(idx):
            def vf(*args, **kwargs):
                if vis[idx]:
                    vis[idx](*args, **kwargs)
                if log[idx]:
                    return log[idx](*args, **kwargs)
            return vf
        visualize_choice, visualize_wave, visualize_backtracking, visualize_propagate, visualize_final, visualize_after = [visfunc(x) for x in range(len(vis))]


    ### Global Constraints ###
    active_global_constraint = lambda wave: True
    if global_constraint == "allpatterns":
        active_global_constraint = make_global_use_all_patterns()
    print(active_global_constraint)

    ### Search Depth Limit
    def makeSearchLengthLimit(max_limit):
        search_length_counter = 0
        def searchLengthLimit(wave):
            nonlocal search_length_counter
            search_length_counter += 1
            return search_length_counter <= max_limit
        return searchLengthLimit

    combined_constraints = [active_global_constraint, makeSearchLengthLimit(1000)]
    def combinedConstraints(wave):
        print
        return all([fn(wave) for fn in combined_constraints])
            
    ### Solving ###

    time_solve_start = None
    time_solve_end = None
    
    solution_tile_grid = None
    print("solving...")
    attempts = 0
    while attempts < attempt_limit:
        attempts += 1
        end_early = False
        time_solve_start = time.time()
        stats = {}
        #profiler = pprofile.Profile()
        if True:
        #with profiler:
            #with PyCallGraph(output=GraphvizOutput(output_file=f"visualization/pycallgraph_{filename}_{timecode}.png")):
                try:
                    solution = run(wave.copy(),
                                   adjacency_matrix,
                                   locationHeuristic=location_heuristic,
                                   patternHeuristic=pattern_heuristic,
                                   periodic=output_periodic,
                                   backtracking=backtracking,
                                   onChoice=visualize_choice,
                                   onBacktrack=visualize_backtracking,
                                   onObserve=visualize_wave,
                                   onPropagate=visualize_propagate,
                                   onFinal=visualize_final,
                                   checkFeasible=combinedConstraints
                    )
                    if visualize_after:
                        stats = visualize_after()
                    #print(solution)
                    #print(stats)
                    solution_as_ids = np.vectorize(lambda x : decode_patterns[x])(solution)
                    solution_tile_grid = pattern_grid_to_tiles(solution_as_ids, pattern_catalog)

                    print("Solution:")
                    #print(solution_tile_grid)
                    render_tiles_to_output(solution_tile_grid, tile_catalog, [tile_size, tile_size], output_destination + filename + "_" + timecode + ".png")

                    time_solve_end = time.time()
                    stats.update({"outcome":"success"})
                    succeeded = True
                except StopEarly:
                    print("Skipping...")
                    end_early = True
                    stats.update({"outcome":"skipped"})
                except TimedOut as e_c:
                    print("Timed Out")
                    if visualize_after:
                        stats = visualize_after()
                    stats.update({"outcome":"timed_out"})
                except Contradiction as e_c:
                    print("Contradiction")
                    if visualize_after:
                        stats = visualize_after()
                    stats.update({"outcome":"contradiction"})
        #profiler.dump_stats(f"logs/profile_{filename}_{timecode}.txt")
            
        outstats = {}
        outstats.update(input_stats)
        solve_duration = time.time() - time_solve_start
        try:
            solve_duration = (time_solve_end - time_solve_start)
        except TypeError:
            pass
        outstats.update({"attempts": attempts, "time_start": time_begin, "time_adjacency": time_adjacency, "time solve start": time_solve_start, "time solve end": time_solve_end, "solve duration": solve_duration, "pattern count": number_of_patterns})
        outstats.update(stats)
        if not log_stats_to_output is None:
            log_stats_to_output(outstats, output_destination + log_filename + ".tsv")
        if not solution_tile_grid is None:
            return solution_tile_grid
        if end_early:
            return None


    return None
