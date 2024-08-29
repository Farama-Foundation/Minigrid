"""Main WFC execution function. Implementation based on https://github.com/ikarth/wfc_2019f"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal

from minigrid.envs.wfc.wfclogic.adjacency import adjacency_extraction
from minigrid.envs.wfc.wfclogic.patterns import (
    make_pattern_catalog_with_rotations,
    pattern_grid_to_tiles,
)
from minigrid.envs.wfc.wfclogic.solver import (
    Contradiction,
    StopEarly,
    TimedOut,
    lexicalLocationHeuristic,
    lexicalPatternHeuristic,
    make_global_use_all_patterns,
    makeAdj,
    makeAntiEntropyLocationHeuristic,
    makeEntropyLocationHeuristic,
    makeHilbertLocationHeuristic,
    makeRandomLocationHeuristic,
    makeRandomPatternHeuristic,
    makeRarestPatternHeuristic,
    makeSpiralLocationHeuristic,
    makeWave,
    makeWeightedPatternHeuristic,
    run,
    simpleLocationHeuristic,
)

from .tiles import make_tile_catalog
from .utilities import tile_grid_to_image

logger = logging.getLogger(__name__)


def make_log_stats() -> Callable[[dict[str, Any], str], None]:
    log_line = 0

    def log_stats(stats: dict[str, Any], filename: str) -> None:
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
    image: NDArray[np.integer],
    tile_size: int = 1,
    pattern_width: int = 2,
    rotations: int = 8,
    output_size: tuple[int, int] = (48, 48),
    ground: int | None = None,
    attempt_limit: int = 10,
    output_periodic: bool = True,
    input_periodic: bool = True,
    loc_heuristic: Literal[
        "lexical", "hilbert", "spiral", "entropy", "anti-entropy", "simple", "random"
    ] = "entropy",
    choice_heuristic: Literal["lexical", "rarest", "weighted", "random"] = "weighted",
    global_constraint: Literal[False, "allpatterns"] = False,
    backtracking: bool = False,
    log_filename: str = "log",
    logging: bool = False,
    log_stats_to_output: Callable[[dict[str, Any], str], None] | None = None,
    np_random: np.random.Generator | None = None,
) -> NDArray[np.integer]:
    time_begin = time.perf_counter()
    output_destination = r"./output/"
    np_random: np.random.Generator = (
        np.random.default_rng() if np_random is None else np_random
    )

    rotations -= 1  # change to zero-based

    input_stats = {
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
    # TODO: generalize this to more than the four cardinal directions
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))

    tile_catalog, tile_grid, _code_list, _unique_tiles = make_tile_catalog(
        image, tile_size
    )
    (
        pattern_catalog,
        pattern_weights,
        pattern_list,
        pattern_grid,
    ) = make_pattern_catalog_with_rotations(
        tile_grid, pattern_width, input_is_periodic=input_periodic, rotations=rotations
    )

    logger.debug("profiling adjacency relations")

    adjacency_relations = adjacency_extraction(
        pattern_grid,
        pattern_catalog,
        direction_offsets,
        (pattern_width, pattern_width),
    )

    logger.debug("adjacency_relations")

    logger.debug(f"output size: {output_size}\noutput periodic: {output_periodic}")
    number_of_patterns = len(pattern_weights)
    logger.debug(f"# patterns: {number_of_patterns}")
    decode_patterns = dict(enumerate(pattern_list))
    encode_patterns = {x: i for i, x in enumerate(pattern_list)}

    adjacency_list: dict[tuple[int, int], list[set[int]]] = {}
    for _, adjacency in direction_offsets:
        adjacency_list[adjacency] = [set() for _ in pattern_weights]
    # logger.debug(adjacency_list)
    for adjacency, pattern1, pattern2 in adjacency_relations:
        # logger.debug(adjacency)
        # logger.debug(decode_patterns[pattern1])
        adjacency_list[adjacency][encode_patterns[pattern1]].add(
            encode_patterns[pattern2]
        )

    logger.debug(f"adjacency: {len(adjacency_list)}")

    time_adjacency = time.perf_counter()

    # Ground #

    ground_list: NDArray[np.int64] | None = None
    if ground:
        ground_list = np.vectorize(lambda x: encode_patterns[x])(
            pattern_grid.flat[(ground - 1) :]
        )
    if ground_list is None or ground_list.size == 0:
        ground_list = None

    wave = makeWave(
        number_of_patterns, output_size[0], output_size[1], ground=ground_list
    )
    adjacency_matrix = makeAdj(adjacency_list)

    # Heuristics #

    encoded_weights: NDArray[np.float64] = np.zeros(
        (number_of_patterns), dtype=np.float64
    )
    for w_id, w_val in pattern_weights.items():
        encoded_weights[encode_patterns[w_id]] = w_val
    choice_random_weighting: NDArray[np.float64] = (
        np_random.random(wave.shape[1:]) * 0.1
    )

    pattern_heuristic: Callable[
        [NDArray[np.bool_], NDArray[np.bool_]], int
    ] = lexicalPatternHeuristic
    if choice_heuristic == "rarest":
        pattern_heuristic = makeRarestPatternHeuristic(encoded_weights, np_random)
    if choice_heuristic == "weighted":
        pattern_heuristic = makeWeightedPatternHeuristic(encoded_weights, np_random)
    if choice_heuristic == "random":
        pattern_heuristic = makeRandomPatternHeuristic(encoded_weights, np_random)

    logger.debug(loc_heuristic)
    location_heuristic: Callable[
        [NDArray[np.bool_]], tuple[int, int]
    ] = lexicalLocationHeuristic
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
        # This requires hilbert_curve to be installed
        location_heuristic = makeHilbertLocationHeuristic(choice_random_weighting)

    # Global Constraints #

    if global_constraint == "allpatterns":
        active_global_constraint = make_global_use_all_patterns()
    else:

        def active_global_constraint(wave) -> bool:
            return True

    logger.debug(active_global_constraint)
    combined_constraints = [active_global_constraint]

    def combinedConstraints(wave: NDArray[np.bool_]) -> bool:
        return all(fn(wave) for fn in combined_constraints)

    # Solving #

    time_solve_start = None
    time_solve_end = None

    solution_tile_grid = None
    logger.debug("solving...")
    attempts = 0
    while attempts < attempt_limit:
        attempts += 1
        time_solve_start = time.perf_counter()
        stats = {}
        try:
            solution = run(
                wave.copy(),
                adjacency_matrix,
                locationHeuristic=location_heuristic,
                patternHeuristic=pattern_heuristic,
                periodic=output_periodic,
                backtracking=backtracking,
                checkFeasible=combinedConstraints,
            )
            solution_as_ids = np.vectorize(lambda x: decode_patterns[x])(solution)
            solution_tile_grid = pattern_grid_to_tiles(solution_as_ids, pattern_catalog)

            time_solve_end = time.perf_counter()
            stats.update({"outcome": "success"})
        except StopEarly:
            logger.debug("Skipping...")
            stats.update({"outcome": "skipped"})
            raise
        except TimedOut:
            logger.debug("Timed Out")
            stats.update({"outcome": "timed_out"})
        except Contradiction:
            # logger.warning(f"Contradiction: {exc}")
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
                log_stats_to_output(
                    outstats, output_destination + log_filename + ".tsv"
                )
        if solution_tile_grid is not None:
            return (
                tile_grid_to_image(
                    solution_tile_grid, tile_catalog, (tile_size, tile_size)
                ),
                outstats,
            )
        else:
            return None, outstats

    raise TimedOut("Attempt limit exceeded.")
