from __future__ import annotations

from typing import Any, Callable, Collection, Dict, Iterable, Iterator, Mapping, Optional, Tuple, TypeVar
from scipy import sparse  # type: ignore
import numpy
import numpy as np
import sys
import math
import itertools
from numpy.typing import NBitBase, NDArray
from hilbertcurve.hilbertcurve import HilbertCurve  # type: ignore


T = TypeVar("T", bound=NBitBase)

# By default Python has a very low recursion limit.
# Might still be better to rewrite te recursion as a loop, of course
sys.setrecursionlimit(5500)


class Contradiction(Exception):
    """Solving could not proceed without backtracking/restarting."""

    pass


class TimedOut(Exception):
    """Solve timed out."""

    pass


class StopEarly(Exception):
    """Aborting solve early."""

    pass


def makeWave(n: int, w: int, h: int, ground: Optional[Iterable[int]] = None) -> NDArray[numpy.bool_]:
    wave: NDArray[numpy.bool_] = numpy.ones((n, w, h), dtype=numpy.bool_)
    if ground is not None:
        wave[:, :, h - 1] = False
        for g in ground:
            wave[g, :,] = False
            wave[g, :, h - 1] = True
    # print(wave)
    # for i in range(wave.shape[0]):
    #  print(wave[i])
    return wave


def makeAdj(
    adjLists: Mapping[Tuple[int, int], Collection[Iterable[int]]]
) -> Dict[Tuple[int, int], NDArray[numpy.bool_]]:
    adjMatrices = {}
    # print(adjLists)
    num_patterns = len(list(adjLists.values())[0])
    for d in adjLists:
        m = numpy.zeros((num_patterns, num_patterns), dtype=bool)
        for i, js in enumerate(adjLists[d]):
            # print(js)
            for j in js:
                m[i, j] = 1
        adjMatrices[d] = sparse.csr_matrix(m)
    return adjMatrices


######################################
# Location Heuristics


def makeRandomLocationHeuristic(preferences: NDArray[np.floating[Any]]) -> Callable[[NDArray[np.bool_]], Tuple[int, int]]:
    def randomLocationHeuristic(wave: NDArray[np.bool_]) -> Tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(unresolved_cell_mask, preferences, numpy.inf)
        row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return randomLocationHeuristic


def makeEntropyLocationHeuristic(preferences: NDArray[np.floating[Any]]) -> Callable[[NDArray[np.bool_]], Tuple[int, int]]:
    def entropyLocationHeuristic(wave: NDArray[np.bool_]) -> Tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(
            unresolved_cell_mask,
            preferences + numpy.count_nonzero(wave, axis=0),
            numpy.inf,
        )
        row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return entropyLocationHeuristic


def makeAntiEntropyLocationHeuristic(
    preferences: NDArray[np.floating[Any]]
) -> Callable[[NDArray[np.bool_]], Tuple[int, int]]:
    def antiEntropyLocationHeuristic(wave: NDArray[np.bool_]) -> Tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(
            unresolved_cell_mask,
            preferences + numpy.count_nonzero(wave, axis=0),
            -numpy.inf,
        )
        row, col = numpy.unravel_index(numpy.argmax(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return antiEntropyLocationHeuristic


def spiral_transforms() -> Iterator[Tuple[int, int]]:
    for N in itertools.count(start=1):
        if N % 2 == 0:
            yield (0, 1)  # right
            for _ in range(N):
                yield (1, 0)  # down
            for _ in range(N):
                yield (0, -1)  # left
        else:
            yield (0, -1)  # left
            for _ in range(N):
                yield (-1, 0)  # up
            for _ in range(N):
                yield (0, 1)  # right


def spiral_coords(x: int, y: int) -> Iterator[Tuple[int, int]]:
    yield x, y
    for transform in spiral_transforms():
        x += transform[0]
        y += transform[1]
        yield x, y

def fill_with_curve(arr: NDArray[np.floating[T]], curve_gen: Iterable[Tuple[int, int]]) -> NDArray[np.floating[T]]:
    arr_len = numpy.prod(arr.shape)
    fill = 0
    for _, coord in enumerate(curve_gen):
        # print(fill, idx, coord)
        if fill < arr_len:
            try:
                arr[coord[0], coord[1]] = fill / arr_len
                fill += 1
            except IndexError:
                pass
        else:
            break
    # print(arr)
    return arr


def makeSpiralLocationHeuristic(preferences: NDArray[np.floating[Any]]) -> Callable[[NDArray[np.bool_]], Tuple[int, int]]:
    # https://stackoverflow.com/a/23707273/5562922

    spiral_gen = (
        sc for sc in spiral_coords(preferences.shape[0] // 2, preferences.shape[1] // 2)
    )

    cell_order = fill_with_curve(preferences, spiral_gen)

    def spiralLocationHeuristic(wave: NDArray[np.bool_]) -> Tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(unresolved_cell_mask, cell_order, numpy.inf)
        row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return spiralLocationHeuristic


def makeHilbertLocationHeuristic(preferences: NDArray[np.floating[Any]]) -> Callable[[NDArray[np.bool_]], Tuple[int, int]]:
    curve_size = math.ceil(math.sqrt(max(preferences.shape[0], preferences.shape[1])))
    print(curve_size)
    curve_size = 4
    h_curve = HilbertCurve(curve_size, 2)

    def h_coords() -> Iterator[Tuple[int, int]]:
        for i in range(100000):
            # print(i)
            try:
                coords = h_curve.coordinates_from_distance(i)
            except ValueError:
                coords = (0, 0)
            # print(coords)
            yield coords

    cell_order = fill_with_curve(preferences, h_coords())
    # print(cell_order)

    def hilbertLocationHeuristic(wave: NDArray[np.bool_]) -> Tuple[int, int]:
        unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
        cell_weights = numpy.where(unresolved_cell_mask, cell_order, numpy.inf)
        row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
        return row.item(), col.item()

    return hilbertLocationHeuristic


def simpleLocationHeuristic(wave: NDArray[np.bool_]) -> Tuple[int, int]:
    unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
    cell_weights = numpy.where(
        unresolved_cell_mask, numpy.count_nonzero(wave, axis=0), numpy.inf
    )
    row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
    return row.item(), col.item()


def lexicalLocationHeuristic(wave: NDArray[np.bool_]) -> Tuple[int, int]:
    unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
    cell_weights = numpy.where(unresolved_cell_mask, 1.0, numpy.inf)
    row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
    return row.item(), col.item()


#####################################
# Pattern Heuristics


def lexicalPatternHeuristic(weights: NDArray[np.bool_], wave: NDArray[np.bool_]) -> int:
    return numpy.nonzero(weights)[0][0].item()


def makeWeightedPatternHeuristic(weights: NDArray[np.floating[Any]]):
    num_of_patterns = len(weights)

    def weightedPatternHeuristic(wave: NDArray[np.bool_], _: NDArray[np.bool_]) -> int:
        # TODO: there's maybe a faster, more controlled way to do this sampling...
        weighted_wave: NDArray[np.floating[Any]] = weights * wave
        weighted_wave /= weighted_wave.sum()
        result = numpy.random.choice(num_of_patterns, p=weighted_wave)
        return result

    return weightedPatternHeuristic


def makeRarestPatternHeuristic(weights: NDArray[np.floating[Any]]) -> Callable[[NDArray[np.bool_], NDArray[np.bool_]], int]:
    """Return a function that chooses the rarest (currently least-used) pattern."""
    def weightedPatternHeuristic(wave: NDArray[np.bool_], total_wave: NDArray[np.bool_]) -> int:
        print(total_wave.shape)
        # [print(e) for e in wave]
        wave_sums = numpy.sum(total_wave, (1, 2))
        # print(wave_sums)
        selected_pattern = numpy.random.choice(
            numpy.where(wave_sums == wave_sums.max())[0]
        )
        return selected_pattern

    return weightedPatternHeuristic


def makeMostCommonPatternHeuristic(
    weights: NDArray[np.floating[Any]]
) -> Callable[[NDArray[np.bool_], NDArray[np.bool_]], int]:
    """Return a function that chooses the most common (currently most-used) pattern."""
    def weightedPatternHeuristic(wave: NDArray[np.bool_], total_wave: NDArray[np.bool_]) -> int:
        print(total_wave.shape)
        # [print(e) for e in wave]
        wave_sums = numpy.sum(total_wave, (1, 2))
        selected_pattern = numpy.random.choice(
            numpy.where(wave_sums == wave_sums.min())[0]
        )
        return selected_pattern

    return weightedPatternHeuristic


def makeRandomPatternHeuristic(weights: NDArray[np.floating[Any]]) -> Callable[[NDArray[np.bool_], NDArray[np.bool_]], int]:
    num_of_patterns = len(weights)

    def randomPatternHeuristic(wave: NDArray[np.bool_], _: NDArray[np.bool_]) -> int:
        # TODO: there's maybe a faster, more controlled way to do this sampling...
        weighted_wave = 1.0 * wave
        weighted_wave /= weighted_wave.sum()
        result = numpy.random.choice(num_of_patterns, p=weighted_wave)
        return result

    return randomPatternHeuristic


######################################
# Global Constraints


def make_global_use_all_patterns() -> Callable[[NDArray[np.bool_]], bool]:
    def global_use_all_patterns(wave: NDArray[np.bool_]) -> bool:
        """Returns true if at least one instance of each pattern is still possible."""
        return numpy.all(numpy.any(wave, axis=(1, 2))).item()

    return global_use_all_patterns


#####################################
# Solver


def propagate(
    wave: NDArray[np.bool_],
    adj: Mapping[Tuple[int, int], NDArray[numpy.bool_]],
    periodic: bool = False,
    onPropagate: Optional[Callable[[NDArray[numpy.bool_]], None]] = None,
) -> None:
    last_count = wave.sum()

    while True:
        supports = {}
        if periodic:
            padded = numpy.pad(wave, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        else:
            padded = numpy.pad(
                wave, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=True
            )

        # adj is the list of adjacencies. For each direction d in adjacency, 
        # check which patterns are still valid... 
        for d in adj:
            dx, dy = d
            # padded[] is a version of the adjacency matrix with the values wrapped around
            # shifted[] is the padded version with the values shifted over in one direction
            # because my code stores the directions as relative (x,y) coordinates, we can find
            # the adjacent cell for each direction by simply shifting the matrix in that direction,
            # which allows for arbitrary adjacency directions. This is somewhat excessive, but elegant.

            shifted = padded[
                :, 1 + dx : 1 + wave.shape[1] + dx, 1 + dy : 1 + wave.shape[2] + dy
            ]
            # print(f"shifted: {shifted.shape} | adj[d]: {adj[d].shape} | d: {d}")
            # raise StopEarly
            # supports[d] = numpy.einsum('pwh,pq->qwh', shifted, adj[d]) > 0

            # The adjacency matrix is a boolean matrix, indexed by the direction and the two patterns.
            # If the value for (direction, pattern1, pattern2) is True, then this is a valid adjacency.
            # This gives us a rapid way to compare: True is 1, False is 0, so multiplying the matrices
            # gives us the adjacency compatibility.
            supports[d] = (adj[d] @ shifted.reshape(shifted.shape[0], -1)).reshape(
                shifted.shape
            ) > 0
            # supports[d] = ( <- for each cell in the matrix
            # adj[d]  <- the adjacency matrix [sliced by the direction d]
            # @       <- Matrix multiplication
            # shifted.reshape(shifted.shape[0], -1)) <- change the shape of the shifted matrix to 2-dimensions, to make the matrix multiplication easier
            # .reshape(           <- reshape our matrix-multiplied result...
            #   shifted.shape)   <- ...to match the original shape of the shifted matrix
            # > 0    <- is not false

        # multiply the wave matrix by the support matrix to find which patterns are still in the domain
        for d in adj:
            wave *= supports[d]

        if wave.sum() == last_count:
            break
        else:
            last_count = wave.sum()

    if onPropagate:
        onPropagate(wave)

    if (wave.sum(axis=0) == 0).any():
        raise Contradiction


def observe(
    wave: NDArray[np.bool_],
    locationHeuristic: Callable[[NDArray[np.bool_]], Tuple[int, int]],
    patternHeuristic: Callable[[NDArray[np.bool_], NDArray[np.bool_]], int],
) -> Tuple[int, int, int]:
    i, j = locationHeuristic(wave)
    pattern = patternHeuristic(wave[:, i, j], wave)
    return pattern, i, j


# def run_loop(wave, adj, locationHeuristic, patternHeuristic, periodic=False, backtracking=False, onBacktrack=None, onChoice=None, checkFeasible=None):
#   stack = []
#   while True:
#     if checkFeasible:
#       if not checkFeasible(wave):
#         raise Contradiction
#     stack.append(wave.copy())
#     propagate(wave, adj, periodic=periodic)
#     try:
#       pattern, i, j = observe(wave, locationHeuristic, patternHeuristic)
#       if onChoice:
#         onChoice(pattern, i, j)
#       wave[:, i, j] = False
#       wave[pattern, i, j] = True
#       propagate(wave, adj, periodic=periodic)
#       if wave.sum() > wave.shape[1] * wave.shape[2]:
#         pass
#       else:
#         return numpy.argmax(wave, 0)
#     except Contradiction:
#       if backtracking:
#         if onBacktrack:
#           onBacktrack()
#         wave = stack.pop()
#         wave[pattern, i, j] = False
#       else:
#         raise


def run(
    wave: NDArray[np.bool_],
    adj: Mapping[Tuple[int, int], NDArray[numpy.bool_]],
    locationHeuristic: Callable[[NDArray[numpy.bool_]], Tuple[int, int]],
    patternHeuristic: Callable[[NDArray[np.bool_], NDArray[np.bool_]], int],
    periodic: bool = False,
    backtracking: bool = False,
    onBacktrack: Optional[Callable[[], None]] = None,
    onChoice: Optional[Callable[[int, int, int], None]] = None,
    onObserve: Optional[Callable[[NDArray[numpy.bool_]], None]] = None,
    onPropagate: Optional[Callable[[NDArray[numpy.bool_]], None]] = None,
    checkFeasible: Optional[Callable[[NDArray[numpy.bool_]], bool]] = None,
    onFinal: Optional[Callable[[NDArray[numpy.bool_]], None]] = None,
    depth: int = 0,
    depth_limit: Optional[int] = None,
) -> NDArray[numpy.int64]:
    # print("run.")
    if checkFeasible:
        if not checkFeasible(wave):
            raise Contradiction
        if depth_limit:
            if depth > depth_limit:
                raise TimedOut
    if depth % 50 == 0:
        print(depth)
    original = wave.copy()
    propagate(wave, adj, periodic=periodic, onPropagate=onPropagate)
    try:
        pattern, i, j = observe(wave, locationHeuristic, patternHeuristic)
        if onChoice:
            onChoice(pattern, i, j)
        wave[:, i, j] = False
        wave[pattern, i, j] = True
        if onObserve:
            onObserve(wave)
        propagate(wave, adj, periodic=periodic, onPropagate=onPropagate)
        if wave.sum() > wave.shape[1] * wave.shape[2]:
            # return run(wave, adj, locationHeuristic, patternHeuristic, periodic, backtracking, onBacktrack)
            return run(
                wave,
                adj,
                locationHeuristic,
                patternHeuristic,
                periodic=periodic,
                backtracking=backtracking,
                onBacktrack=onBacktrack,
                onChoice=onChoice,
                onObserve=onObserve,
                onPropagate=onPropagate,
                checkFeasible=checkFeasible,
                depth=depth + 1,
                depth_limit=depth_limit,
            )
        else:
            if onFinal:
                onFinal(wave)
            return numpy.argmax(wave, 0)
    except Contradiction:
        if backtracking:
            if onBacktrack:
                onBacktrack()
            wave = original
            wave[pattern, i, j] = False
            return run(
                wave,
                adj,
                locationHeuristic,
                patternHeuristic,
                periodic=periodic,
                backtracking=backtracking,
                onBacktrack=onBacktrack,
                onChoice=onChoice,
                onObserve=onObserve,
                onPropagate=onPropagate,
                checkFeasible=checkFeasible,
                depth=depth + 1,
                depth_limit=depth_limit,
            )
        else:
            if onFinal:
                onFinal(wave)
            raise
