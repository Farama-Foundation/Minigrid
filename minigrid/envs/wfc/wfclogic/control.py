from __future__ import annotations

import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple
from .wfc_tiles import make_tile_catalog
from .wfc_patterns import (
    pattern_grid_to_tiles,
    make_pattern_catalog_with_rotations,
)
from .wfc_adjacency import adjacency_extraction
from .wfc_solver import (
    run,
    makeWave,
    makeAdj,
    lexicalLocationHeuristic,
    lexicalPatternHeuristic,
    makeWeightedPatternHeuristic,
    Contradiction,
    StopEarly,
    makeEntropyLocationHeuristic,
    make_global_use_all_patterns,
    makeRandomLocationHeuristic,
    makeRandomPatternHeuristic,
    TimedOut,
    simpleLocationHeuristic,
    makeSpiralLocationHeuristic,
    makeHilbertLocationHeuristic,
    makeAntiEntropyLocationHeuristic,
    makeRarestPatternHeuristic,
)
from .wfc_visualize import (
    figure_list_of_tiles,
    figure_false_color_tile_grid,
    figure_pattern_catalog,
    render_tiles_to_output,
    figure_adjacencies,
    make_solver_visualizers,
    make_solver_loggers,
    tile_grid_to_image,
)
import imageio  # type: ignore
import numpy as np
import time
import logging
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def visualize_tiles(unique_tiles, tile_catalog, tile_grid):
    if False:
        figure_list_of_tiles(unique_tiles, tile_catalog)
        figure_false_color_tile_grid(tile_grid)


def visualize_patterns(pattern_catalog, tile_catalog, pattern_weights, pattern_width):
    if False:
        figure_pattern_catalog(
            pattern_catalog, tile_catalog, pattern_weights, pattern_width
        )


def make_log_stats() -> Callable[[Dict[str, Any], str], None]:
    log_line = 0

    def log_stats(stats: Dict[str, Any], filename: str) -> None:
        nonlocal log_line
        if stats:
            log_line += 1
            with open(filename, "a", encoding="utf_8") as logf:
                if log_line < 2:
                    for s in stats.keys():
                        print(str(s), end="\t", file=logf)
                    print("", file=logf)
                for s in stats.keys():
                    print(str(stats[s]), end="\t", file=logf)
                print("", file=logf)

    return log_stats


def execute_wfc(
    filename: Optional[str] = None,
    tile_size: int = 1,
    pattern_width: int = 2,
    rotations: int = 8,
    output_size: Tuple[int, int] = (48, 48),
    ground: Optional[int] = None,
    attempt_limit: int = 10,
    output_periodic: bool = True,
    input_periodic: bool = True,
    loc_heuristic: Literal["lexical", "hilbert", "spiral", "entropy", "anti-entropy", "simple", "random"] = "entropy",
    choice_heuristic: Literal["lexical", "rarest", "weighted", "random"] = "weighted",
    visualize: bool = False,
    global_constraint: Literal[False, "allpatterns"] = False,
    backtracking: bool = False,
    log_filename: str = "log",
    logging: bool = False,
    global_constraints: None = None,
    log_stats_to_output: Optional[Callable[[Dict[str, Any], str], None]] = None,
    *,
    image: Optional[NDArray[np.integer]] = None,
) -> NDArray[np.integer]:
    timecode = datetime.datetime.now().isoformat().replace(":", ".")
    time_begin = time.perf_counter()
    output_destination = r"./output/"
    input_folder = r"./images/samples/"

    rotations -= 1  # change to zero-based

    input_stats = {
        "filename": str(filename),
        "tile_size": tile_size,
        "pattern_width": pattern_width,
        "rotations": rotations,
        "output_size": output_size,
        "ground": ground,
        "attempt_limit": attempt_limit,
        "output_periodic": output_periodic,
        "input_periodic": input_periodic,
        "location heuristic": loc_heuristic,
        "choice heuristic": choice_heuristic,
        "global constraint": global_constraint,
        "backtracking": backtracking,
    }

    # Load the image
    if filename:
        if image is not None:
            raise TypeError("Only filename or image can be provided, not both.")
        image = imageio.imread(input_folder + filename + ".png")[:, :, :3]  # TODO: handle alpha channels

    if image is None:
        raise TypeError("An image must be given.")

    # TODO: generalize this to more than the four cardinal directions
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))

    tile_catalog, tile_grid, _code_list, _unique_tiles = make_tile_catalog(image, tile_size)
    (
        pattern_catalog,
        pattern_weights,
        pattern_list,
        pattern_grid,
    ) = make_pattern_catalog_with_rotations(
        tile_grid, pattern_width, input_is_periodic=input_periodic, rotations=rotations
    )

    logger.debug("pattern catalog")

    # visualize_tiles(unique_tiles, tile_catalog, tile_grid)
    # visualize_patterns(pattern_catalog, tile_catalog, pattern_weights, pattern_width)
    # figure_list_of_tiles(unique_tiles, tile_catalog, output_filename=f"visualization/tilelist_{filename}_{timecode}")
    # figure_false_color_tile_grid(tile_grid, output_filename=f"visualization/tile_falsecolor_{filename}_{timecode}")
    if visualize and filename:
        figure_pattern_catalog(
            pattern_catalog,
            tile_catalog,
            pattern_weights,
            pattern_width,
            output_filename=f"visualization/pattern_catalog_{filename}_{timecode}",
        )

    logger.debug("profiling adjacency relations")
    if False:
        import pprofile  # type: ignore
        profiler = pprofile.Profile()
        with profiler:
            adjacency_relations = adjacency_extraction(
                pattern_grid,
                pattern_catalog,
                direction_offsets,
                [pattern_width, pattern_width],
            )
        profiler.dump_stats(f"logs/profile_adj_{filename}_{timecode}.txt")
    else:
        adjacency_relations = adjacency_extraction(
            pattern_grid,
            pattern_catalog,
            direction_offsets,
            (pattern_width, pattern_width),
        )

    logger.debug("adjacency_relations")

    if visualize:
        figure_adjacencies(
            adjacency_relations,
            direction_offsets,
            tile_catalog,
            pattern_catalog,
            pattern_width,
            [tile_size, tile_size],
            output_filename=f"visualization/adjacency_{filename}_{timecode}_A",
        )
        # figure_adjacencies(adjacency_relations, direction_offsets, tile_catalog, pattern_catalog, pattern_width, [tile_size, tile_size], output_filename=f"visualization/adjacency_{filename}_{timecode}_B", render_b_first=True)

    logger.debug(f"output size: {output_size}\noutput periodic: {output_periodic}")
    number_of_patterns = len(pattern_weights)
    logger.debug(f"# patterns: {number_of_patterns}")
    decode_patterns = dict(enumerate(pattern_list))
    encode_patterns = {x: i for i, x in enumerate(pattern_list)}
    _encode_directions = {j: i for i, j in direction_offsets}

    adjacency_list: Dict[Tuple[int, int], List[Set[int]]] = {}
    for _, adjacency in direction_offsets:
        adjacency_list[adjacency] = [set() for _ in pattern_weights]
    # logger.debug(adjacency_list)
    for adjacency, pattern1, pattern2 in adjacency_relations:
        # logger.debug(adjacency)
        # logger.debug(decode_patterns[pattern1])
        adjacency_list[adjacency][encode_patterns[pattern1]].add(encode_patterns[pattern2])

    logger.debug(f"adjacency: {len(adjacency_list)}")

    time_adjacency = time.perf_counter()

    ### Ground ###

    ground_list: Optional[NDArray[np.int64]] = None
    if ground:
        ground_list = np.vectorize(lambda x: encode_patterns[x])(
            pattern_grid.flat[(ground - 1) :]
        )
    if ground_list is None or ground_list.size == 0:
        ground_list = None

    if ground_list is not None:
        ground_catalog = {
            encode_patterns[k]: v
            for k, v in pattern_catalog.items()
            if encode_patterns[k] in ground_list
        }
        if visualize:
            figure_pattern_catalog(
                ground_catalog,
                tile_catalog,
                pattern_weights,
                pattern_width,
                output_filename=f"visualization/patterns_ground_{filename}_{timecode}",
            )

    wave = makeWave(
        number_of_patterns, output_size[0], output_size[1], ground=ground_list
    )
    adjacency_matrix = makeAdj(adjacency_list)

    ### Heuristics ###

    encoded_weights: NDArray[np.float64] = np.zeros((number_of_patterns), dtype=np.float64)
    for w_id, w_val in pattern_weights.items():
        encoded_weights[encode_patterns[w_id]] = w_val
    choice_random_weighting: NDArray[np.float64] = np.random.random_sample(wave.shape[1:]) * 0.1

    pattern_heuristic: Callable[[NDArray[np.bool_], NDArray[np.bool_]], int] = lexicalPatternHeuristic
    if choice_heuristic == "rarest":
        pattern_heuristic = makeRarestPatternHeuristic(encoded_weights)
    if choice_heuristic == "weighted":
        pattern_heuristic = makeWeightedPatternHeuristic(encoded_weights)
    if choice_heuristic == "random":
        pattern_heuristic = makeRandomPatternHeuristic(encoded_weights)

    logger.debug(loc_heuristic)
    location_heuristic: Callable[[NDArray[np.bool_]], Tuple[int, int]] = lexicalLocationHeuristic
    if loc_heuristic == "anti-entropy":
        location_heuristic = makeAntiEntropyLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "entropy":
        location_heuristic = makeEntropyLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "random":
        location_heuristic = makeRandomLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "simple":
        location_heuristic = simpleLocationHeuristic
    if loc_heuristic == "spiral":
        location_heuristic = makeSpiralLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "hilbert":
        location_heuristic = makeHilbertLocationHeuristic(choice_random_weighting)

    ### Visualization ###

    (
        visualize_choice,
        visualize_wave,
        visualize_backtracking,
        visualize_propagate,
        visualize_final,
        visualize_after,
    ) = (None, None, None, None, None, None)
    if filename and visualize:
        (
            visualize_choice,
            visualize_wave,
            visualize_backtracking,
            visualize_propagate,
            visualize_final,
            visualize_after,
        ) = make_solver_visualizers(
            f"{filename}_{timecode}",
            wave,
            decode_patterns=decode_patterns,
            pattern_catalog=pattern_catalog,
            tile_catalog=tile_catalog,
            tile_size=[tile_size, tile_size],
        )
    if filename and logging:
        (
            visualize_choice,
            visualize_wave,
            visualize_backtracking,
            visualize_propagate,
            visualize_final,
            visualize_after,
        ) = make_solver_loggers(f"{filename}_{timecode}", input_stats.copy())
    if filename and logging and visualize:
        vis = make_solver_visualizers(
            f"{filename}_{timecode}",
            wave,
            decode_patterns=decode_patterns,
            pattern_catalog=pattern_catalog,
            tile_catalog=tile_catalog,
            tile_size=[tile_size, tile_size],
        )
        log = make_solver_loggers(f"{filename}_{timecode}", input_stats.copy())

        def visfunc(idx: int):
            def vf(*args, **kwargs):
                if vis[idx]:
                    vis[idx](*args, **kwargs)
                if log[idx]:
                    return log[idx](*args, **kwargs)

            return vf

        (
            visualize_choice,
            visualize_wave,
            visualize_backtracking,
            visualize_propagate,
            visualize_final,
            visualize_after,
        ) = [visfunc(x) for x in range(len(vis))]

    ### Global Constraints ###
    active_global_constraint = lambda wave: True
    if global_constraint == "allpatterns":
        active_global_constraint = make_global_use_all_patterns()
    logger.debug(active_global_constraint)
    combined_constraints = [active_global_constraint]

    def combinedConstraints(wave: NDArray[np.bool_]) -> bool:
        return all(fn(wave) for fn in combined_constraints)

    ### Solving ###

    time_solve_start = None
    time_solve_end = None

    solution_tile_grid = None
    logger.debug("solving...")
    attempts = 0
    while attempts < attempt_limit:
        attempts += 1
        time_solve_start = time.perf_counter()
        stats = {}
        # profiler = pprofile.Profile()
        # with profiler:
        # with PyCallGraph(output=GraphvizOutput(output_file=f"visualization/pycallgraph_{filename}_{timecode}.png")):
        try:
            solution = run(
                wave.copy(),
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
                checkFeasible=combinedConstraints,
            )
            if visualize_after:
                stats = visualize_after()
            # logger.debug(solution)
            # logger.debug(stats)
            solution_as_ids = np.vectorize(lambda x: decode_patterns[x])(solution)
            solution_tile_grid = pattern_grid_to_tiles(
                solution_as_ids, pattern_catalog
            )

            logger.debug("Solution:")
            # logger.debug(solution_tile_grid)
            if filename:
                render_tiles_to_output(
                    solution_tile_grid,
                    tile_catalog,
                    (tile_size, tile_size),
                    output_destination + filename + "_" + timecode + ".png",
                )

            time_solve_end = time.perf_counter()
            stats.update({"outcome": "success"})
        except StopEarly:
            logger.debug("Skipping...")
            stats.update({"outcome": "skipped"})
            raise
        except TimedOut:
            logger.debug("Timed Out")
            if visualize_after:
                stats = visualize_after()
            stats.update({"outcome": "timed_out"})
        except Contradiction as exc:
            logger.warning(f"Contradiction: {exc}")
            if visualize_after:
                stats = visualize_after()
            stats.update({"outcome": "contradiction"})
        finally:
            # profiler.dump_stats(f"logs/profile_{filename}_{timecode}.txt")
            outstats = {}
            outstats.update(input_stats)
            solve_duration = time.perf_counter() - time_solve_start
            if time_solve_end is not None:
                solve_duration = time_solve_end - time_solve_start
            adjacency_duration = time_solve_start - time_adjacency
            outstats.update(
                {
                    "attempts": attempts,
                    "time_start": time_begin,
                    "time_adjacency": time_adjacency,
                    "adjacency_duration": adjacency_duration,
                    "time solve start": time_solve_start,
                    "time solve end": time_solve_end,
                    "solve duration": solve_duration,
                    "pattern count": number_of_patterns,
                }
            )
            outstats.update(stats)
            if log_stats_to_output is not None:
                log_stats_to_output(outstats, output_destination + log_filename + ".tsv")
        if solution_tile_grid is not None:
            return tile_grid_to_image(solution_tile_grid, tile_catalog, (tile_size, tile_size))

    raise TimedOut("Attempt limit exceeded.")
