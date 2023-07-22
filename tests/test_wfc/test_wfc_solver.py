from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from minigrid.envs.wfc.wfclogic import solver as wfc_solver


def test_makeWave() -> None:
    wave = wfc_solver.makeWave(3, 10, 20, ground=[-1])
    assert wave.sum() == (2 * 10 * 19) + (1 * 10 * 1)
    assert wave[2, 5, 19]
    assert not wave[1, 5, 19]


def test_entropyLocationHeuristic() -> None:
    wave = np.ones((5, 3, 4), dtype=bool)  # everything is possible
    wave[1:, 0, 0] = False  # first cell is fully observed
    wave[4, :, 2] = False
    preferences: NDArray[np.float_] = np.ones((3, 4), dtype=np.float_) * 0.5
    preferences[1, 2] = 0.3
    preferences[1, 1] = 0.1
    heu = wfc_solver.makeEntropyLocationHeuristic(preferences)
    result = heu(wave)
    assert (1, 2) == result


def test_observe() -> None:
    my_wave = np.ones((5, 3, 4), dtype=np.bool_)
    my_wave[0, 1, 2] = False

    def locHeu(wave: NDArray[np.bool_]) -> tuple[int, int]:
        assert np.array_equal(wave, my_wave)
        return 1, 2

    def patHeu(weights: NDArray[np.bool_], wave: NDArray[np.bool_]) -> int:
        assert np.array_equal(weights, my_wave[:, 1, 2])
        return 3

    assert wfc_solver.observe(
        my_wave, locationHeuristic=locHeu, patternHeuristic=patHeu
    ) == (
        3,
        1,
        2,
    )


def test_propagate() -> None:
    wave = np.ones((3, 3, 4), dtype=bool)
    adjLists = {}
    # checkerboard #0/#1 or solid fill #2
    adjLists[(+1, 0)] = adjLists[(-1, 0)] = adjLists[(0, +1)] = adjLists[(0, -1)] = [
        [1],
        [0],
        [2],
    ]
    wave[:, 0, 0] = False
    wave[0, 0, 0] = True
    adj = wfc_solver.makeAdj(adjLists)
    wfc_solver.propagate(wave, adj, periodic=False)
    expected_result = np.array(
        [
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, False, True, False],
            ],
            [
                [False, True, False, True],
                [True, False, True, False],
                [False, True, False, True],
            ],
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
        ]
    )
    assert np.array_equal(wave, expected_result)


def test_run() -> None:
    wave = wfc_solver.makeWave(3, 3, 4)
    adjLists = {}
    adjLists[(+1, 0)] = adjLists[(-1, 0)] = adjLists[(0, +1)] = adjLists[(0, -1)] = [
        [1],
        [0],
        [2],
    ]
    adj = wfc_solver.makeAdj(adjLists)

    first_result = wfc_solver.run(
        wave.copy(),
        adj,
        locationHeuristic=wfc_solver.lexicalLocationHeuristic,
        patternHeuristic=wfc_solver.lexicalPatternHeuristic,
        periodic=False,
    )

    expected_first_result = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    assert np.array_equal(first_result, expected_first_result)

    event_log: list = []

    def onChoice(pattern: int, i: int, j: int) -> None:
        event_log.append((pattern, i, j))

    def onBacktrack() -> None:
        event_log.append("backtrack")

    second_result = wfc_solver.run(
        wave.copy(),
        adj,
        locationHeuristic=wfc_solver.lexicalLocationHeuristic,
        patternHeuristic=wfc_solver.lexicalPatternHeuristic,
        periodic=True,
        backtracking=True,
        onChoice=onChoice,
        onBacktrack=onBacktrack,
    )

    expected_second_result = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])

    assert np.array_equal(second_result, expected_second_result)
    assert event_log == [(0, 0, 0), "backtrack", (2, 0, 0)]

    class Infeasible(Exception):
        pass

    def explode(wave: NDArray[np.bool_]) -> bool:
        if wave.sum() < 20:
            raise Infeasible
        return False

    with pytest.raises(wfc_solver.Contradiction):
        wfc_solver.run(
            wave.copy(),
            adj,
            locationHeuristic=wfc_solver.lexicalLocationHeuristic,
            patternHeuristic=wfc_solver.lexicalPatternHeuristic,
            periodic=True,
            backtracking=True,
            checkFeasible=explode,
        )
