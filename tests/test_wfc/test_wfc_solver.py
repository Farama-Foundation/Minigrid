import imageio
import numpy
from wfc import wfc_solver
from wfc import wfc_tiles
from wfc import wfc_patterns
from wfc import wfc_adjacency


def test_makeWave():
    wave = wfc_solver.makeWave(3, 10, 20, ground=[-1])
    # print(wave)
    # print(wave.sum())
    # print((2*10*19) + (1*10*1))
    assert wave.sum() == (2 * 10 * 19) + (1 * 10 * 1)
    assert wave[2, 5, 19] == True
    assert wave[1, 5, 19] == False


def test_entropyLocationHeuristic():
    wave = numpy.ones((5, 3, 4), dtype=bool)  # everthing is possible
    wave[1:, 0, 0] = False  # first cell is fully observed
    wave[4, :, 2] = False
    preferences = numpy.ones((3, 4), dtype=float) * 0.5
    preferences[1, 2] = 0.3
    preferences[1, 1] = 0.1
    heu = wfc_solver.makeEntropyLocationHeuristic(preferences)
    result = heu(wave)
    assert [1, 2] == result


def test_observe():

    my_wave = numpy.ones((5, 3, 4), dtype=bool)
    my_wave[0, 1, 2] = False

    def locHeu(wave):
        assert numpy.array_equal(wave, my_wave)
        return 1, 2

    def patHeu(weights, wave):
        assert numpy.array_equal(weights, my_wave[:, 1, 2])
        return 3

    assert wfc_solver.observe(my_wave, locationHeuristic=locHeu, patternHeuristic=patHeu) == (
        3,
        1,
        2,
    )


def test_propagate():
    wave = numpy.ones((3, 3, 4), dtype=bool)
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
    expected_result = numpy.array(
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
    assert numpy.array_equal(wave, expected_result)


def test_run():
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

    expected_first_result = numpy.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    assert numpy.array_equal(first_result, expected_first_result)

    event_log = []

    def onChoice(pattern, i, j):
        event_log.append((pattern, i, j))

    def onBacktrack():
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

    expected_second_result = numpy.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])

    assert numpy.array_equal(second_result, expected_second_result)
    print(event_log)
    assert event_log == [(0, 0, 0), "backtrack", (2, 0, 0)]

    class Infeasible(Exception):
        pass

    def explode(wave):
        if wave.sum() < 20:
            raise Infeasible

    try:
        result = wfc_solver.run(
            wave.copy(),
            adj,
            locationHeuristic=wfc_solver.lexicalLocationHeuristic,
            patternHeuristic=wfc_solver.lexicalPatternHeuristic,
            periodic=True,
            backtracking=True,
            checkFeasible=explode,
        )
        print(result)
        happy = False
    except wfc_solver.Contradiction:
        happy = True

    assert happy


def _test_recurse_vs_loop(resources):
    # FIXME: run_recurse or run_loop do not exist anymore
    filename = resources.get_image("samples/Red Maze.png")
    img = imageio.imread(filename)
    tile_size = 1
    pattern_width = 2
    rotations = 0
    output_size = [84, 84]
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))
    _tile_catalog, tile_grid, _code_list, _unique_tiles = wfc_tiles.make_tile_catalog(img, tile_size)
    pattern_catalog, pattern_weights, pattern_list, pattern_grid = wfc_patterns.make_pattern_catalog(
        tile_grid, pattern_width, rotations
    )
    adjacency_relations = wfc_adjacency.adjacency_extraction(
        pattern_grid, pattern_catalog, direction_offsets
    )
    number_of_patterns = len(pattern_weights)
    decode_patterns = {x: i for i, x in enumerate(pattern_list)}
    adjacency_list = {}
    for i, d in direction_offsets:
        adjacency_list[d] = [set() for i in pattern_weights]
    for i in adjacency_relations:
        adjacency_list[i[0]][decode_patterns[i[1]]].add(decode_patterns[i[2]])
    wave = wfc_solver.makeWave(number_of_patterns, output_size[0], output_size[1])
    adjacency_matrix = wfc_solver.makeAdj(adjacency_list)
    solution_loop = wfc_solver.run(
        wave.copy(),
        adjacency_matrix,
        locationHeuristic=wfc_solver.lexicalLocationHeuristic,
        patternHeuristic=wfc_solver.lexicalPatternHeuristic,
        periodic=True,
        backtracking=False,
        onChoice=None,
        onBacktrack=None,
    )
    solution_recurse = wfc_solver.run_recurse(
        wave.copy(),
        adjacency_matrix,
        locationHeuristic=wfc_solver.lexicalLocationHeuristic,
        patternHeuristic=wfc_solver.lexicalPatternHeuristic,
        periodic=True,
        backtracking=False,
        onChoice=None,
        onBacktrack=None,
    )
    assert numpy.array_equiv(solution_loop, solution_recurse)
