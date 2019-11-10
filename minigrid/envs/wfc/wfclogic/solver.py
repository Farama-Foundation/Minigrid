import numpy
import sys

sys.setrecursionlimit(5500)


class Contradiction(Exception):
  """Solving could not proceed without backtracking/restarting."""
  pass

def makeWave(n, w, h, ground=None):
  wave = numpy.ones((n, w, h),dtype=bool)
  if ground is not None:
    wave[:,:,h-1] = 0
    wave[ground,:,h-1] = 1
  #print(wave)
  return wave

def makeAdj(adjLists):
  adjMatrices = {}
  #print(adjLists)
  num_patterns = len(list(adjLists.values())[0])
  for d in adjLists:
    m = numpy.zeros((num_patterns,num_patterns),dtype=bool)
    for i, js in enumerate(adjLists[d]):
      #print(js)
      for j in js:
        m[i,j] = 1
    adjMatrices[d] = m
  return adjMatrices


######################################
# Location Heuristics

def makeEntropyLocationHeuristic(preferences):
  def entropyLocationHeuristic(wave):
    unresolved_cell_mask = (numpy.count_nonzero(wave, axis=0) > 1)
    cell_weights = numpy.where(unresolved_cell_mask, preferences + numpy.count_nonzero(wave, axis=0), numpy.inf)
    row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
    return [row, col]
  return entropyLocationHeuristic


def lexicalLocationHeuristic(wave):
  unresolved_cell_mask = (numpy.count_nonzero(wave, axis=0) > 1)
  cell_weights = numpy.where(unresolved_cell_mask,  numpy.count_nonzero(wave, axis=0), numpy.inf)
  row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
  return [row, col]

#####################################
# Pattern Heuristics

def lexicalPatternHeuristic(weights):
  return numpy.nonzero(weights)[0][0]

def makeWeightedPatternHeuristic(weights):
  num_of_patterns = len(weights)
  def weightedPatternHeuristic(wave):
    # TODO: there's maybe a faster, more controlled way to do this sampling...
    weighted_wave = (weights * wave)
    weighted_wave /= weighted_wave.sum()
    result = numpy.random.choice(num_of_patterns, p=weighted_wave)
    return result
  return weightedPatternHeuristic


#####################################
# Solver

def propagate(wave, adj, periodic=False):
  last_count = wave.sum()

  while True:
    supports = {}
    if periodic:
      padded = numpy.pad(wave,((0,0),(1,1),(1,1)), mode='wrap')
    else:
      padded = numpy.pad(wave,((0,0),(1,1),(1,1)), mode='constant',constant_values=True)

    for d in adj:
      dx,dy = d
      shifted = padded[:,1+dx:1+wave.shape[1]+dx,1+dy:1+wave.shape[2]+dy]
      supports[d] = numpy.einsum('pwh,pq->qwh', shifted, adj[d]) > 0

    for d in adj:
      wave *= supports[d]

    if wave.sum() == last_count:
      break
    else:
      last_count = wave.sum()

  if wave.sum() == 0:
    raise Contradiction


def observe(wave, locationHeuristic, patternHeuristic):
  i,j = locationHeuristic(wave)
  pattern = patternHeuristic(wave[:,i,j])
  return pattern, i, j


def run_loop(wave, adj, locationHeuristic, patternHeuristic, periodic=False, backtracking=False, onBacktrack=None, onChoice=None, checkFeasible=None):
  stack = []
  while True:
    if checkFeasible:
      if not checkFeasible(wave):
        raise Contradiction
    stack.append(wave.copy())
    propagate(wave, adj, periodic=periodic)
    try:
      pattern, i, j = observe(wave, locationHeuristic, patternHeuristic) 
      if onChoice:
        onChoice(pattern, i, j)
      wave[:, i, j] = False
      wave[pattern, i, j] = True
      propagate(wave, adj, periodic=periodic)
      if wave.sum() > wave.shape[1] * wave.shape[2]:
        pass
      else:
        return numpy.argmax(wave, 0)
    except Contradiction:
      if backtracking:
        if onBacktrack:
          onBacktrack()
        wave = stack.pop()
        wave[pattern, i, j] = False
      else:
        raise

  



def run(wave, adj, locationHeuristic, patternHeuristic, periodic=False, backtracking=False, onBacktrack=None, onChoice=None, checkFeasible=None):
  print(".")
  if checkFeasible:
    if not checkFeasible(wave):
      raise Contradiction
  original = wave.copy()
  #print("propagate")
  propagate(wave, adj, periodic=periodic)
  try:
    pattern, i, j = observe(wave, locationHeuristic, patternHeuristic) 
    if onChoice:
      onChoice(pattern, i, j)
    wave[:, i, j] = False
    wave[pattern, i, j] = True
    #print("propagate")
    propagate(wave, adj, periodic=periodic)
    if wave.sum() > wave.shape[1] * wave.shape[2]:
      return run(wave, adj, locationHeuristic, patternHeuristic, periodic, backtracking, onBacktrack)
    else:
      return numpy.argmax(wave, 0)
  except Contradiction:
    if backtracking:
      if onBacktrack:
        onBacktrack()
      wave = original
      wave[pattern, i, j] = False
      return run(wave, adj, locationHeuristic, patternHeuristic, periodic, backtracking, onBacktrack, checkFeasible)
    else:
      raise

#############################
# Tests

def test_makeWave():
  wave = makeWave(3, 10, 20, ground=-1)
  assert wave.sum() == (10*20*3 - 2*10)
  assert wave[2,5,19] == True
  assert wave[1,5,19] == False

def test_entropyLocationHeuristic():
    wave = numpy.ones((5, 3, 4), dtype=bool) # everthing is possible
    wave[1:,0, 0] = False # first cell is fully observed
    wave[4, :, 2] = False
    preferences = numpy.ones((3, 4), dtype=float) * 0.5
    preferences[1, 2] = 0.3
    preferences[1, 1] = 0.1
    heu = makeEntropyLocationHeuristic(preferences)
    result = heu(wave)
    assert [1, 2] == result

def test_observe():

  my_wave = numpy.ones((5, 3, 4), dtype=bool)
  my_wave[0,1,2] = False

  def locHeu(wave):
    assert numpy.array_equal(wave, my_wave)
    return 1,2
  def patHeu(weights):
    assert numpy.array_equal(weights, my_wave[:,1,2])
    return 3

  assert observe(my_wave,
                 locationHeuristic=locHeu,
                 patternHeuristic=patHeu) == (3,1,2)

def test_propagate():
  wave = numpy.ones((3,3,4),dtype=bool)
  adjLists = {}
  # checkerboard #0/#1 or solid fill #2
  adjLists[(+1,0)] = adjLists[(-1,0)] = adjLists[(0,+1)] = adjLists[(0,-1)] = [[1],[0],[2]]
  wave[:,0,0] = False
  wave[0,0,0] = True
  adj = makeAdj(adjLists)
  propagate(wave, adj, periodic=False)
  expected_result = numpy.array([[[ True, False,  True, False],
          [False,  True, False,  True],
          [ True, False,  True, False]],
        [[False,  True, False,  True],
          [ True, False,  True, False],
          [False,  True, False,  True]],
        [[False, False, False, False],
          [False, False, False, False],
          [False, False, False, False]]])
  assert numpy.array_equal(wave, expected_result)


def test_run():
  wave = makeWave(3,3,4)
  adjLists = {}
  adjLists[(+1,0)] = adjLists[(-1,0)] = adjLists[(0,+1)] = adjLists[(0,-1)] = [[1],[0],[2]]
  adj = makeAdj(adjLists)

  first_result = run(wave.copy(),
      adj,
      locationHeuristic=lexicalLocationHeuristic,
      patternHeuristic=lexicalPatternHeuristic,
      periodic=False)

  expected_first_result = numpy.array([[0, 1, 0, 1],[1, 0, 1, 0],[0, 1, 0, 1]])

  assert numpy.array_equal(first_result, expected_first_result)

  event_log = []
  def onChoice(pattern, i, j):
    event_log.append((pattern,i,j))
  def onBacktrack():
    event_log.append('backtrack')

  second_result = run(wave.copy(),
      adj,
      locationHeuristic=lexicalLocationHeuristic,
      patternHeuristic=lexicalPatternHeuristic,
      periodic=True,
      backtracking=True,
      onChoice=onChoice,
      onBacktrack=onBacktrack)

  expected_second_result = numpy.array([[2, 2, 2, 2],[2, 2, 2, 2],[2, 2, 2, 2]])

  assert numpy.array_equal(second_result, expected_second_result)
  print(event_log)
  assert event_log == [(0, 0, 0), 'backtrack']

  def explode(wave):
    if wave.sum() < 20:
      raise Infeasible

  try:
    result = run(wave.copy(),
        adj,
        locationHeuristic=lexicalLocationHeuristic,
        patternHeuristic=lexicalPatternHeuristic,
        periodic=True,
        backtracking=True,
        checkFeasible=explode)
    print(result)
    happy = False
  except Contradiction:
    happy = True

  assert happy

def test_recurse_vs_loop():
  from wfc_tiles import make_tile_catalog
  from wfc_patterns import make_pattern_catalog, pattern_grid_to_tiles
  from wfc_adjacency import adjacency_extraction
  from wfc_solver import run, makeWave, makeAdj, lexicalLocationHeuristic, lexicalPatternHeuristic
  from wfc_visualize import figure_list_of_tiles, figure_false_color_tile_grid, figure_pattern_catalog, render_tiles_to_output, figure_adjacencies

  import imageio
  img = imageio.imread("images/samples/Red Maze.png")
  tile_size = 1
  pattern_width = 2
  rotations = 0
  output_size = [84, 84]
  ground = None
  direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))
  tile_catalog, tile_grid, code_list, unique_tiles = make_tile_catalog(img, tile_size)
  pattern_catalog, pattern_weights, pattern_list, pattern_grid = make_pattern_catalog(tile_grid, pattern_width, rotations)
  adjacency_relations = adjacency_extraction(pattern_grid, pattern_catalog, direction_offsets)
  number_of_patterns = len(pattern_weights)
  encode_patterns = dict(enumerate(pattern_list))
  decode_patterns = {x: i for i, x in enumerate(pattern_list)}
  decode_directions = {j:i for i,j in direction_offsets}
  adjacency_list = {}
  for i,d in direction_offsets:
    adjacency_list[d] = [set() for i in pattern_weights]
  for i in adjacency_relations:
    adjacency_list[i[0]][decode_patterns[i[1]]].add(decode_patterns[i[2]])
  wave = makeWave(number_of_patterns, output_size[0], output_size[1])
  adjacency_matrix = makeAdj(adjacency_list)
  solution_loop = run(wave.copy(),
               adjacency_matrix,
               locationHeuristic=lexicalLocationHeuristic,
               patternHeuristic=lexicalPatternHeuristic,
               periodic=True,
               backtracking=False,
               onChoice=None,
               onBacktrack=None)
  solution_recurse = run_recurse(wave.copy(),
               adjacency_matrix,
               locationHeuristic=lexicalLocationHeuristic,
               patternHeuristic=lexicalPatternHeuristic,
               periodic=True,
               backtracking=False,
               onChoice=None,
               onBacktrack=None)
  assert (numpy.array_equiv(solution_loop, solution_recurse))




  
  
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

if __name__ == "__main__":
  with PyCallGraph(output=GraphvizOutput()):
    test_makeWave()
    test_entropyLocationHeuristic()
    test_observe()
    test_propagate()
    #test_run()
    test_recurse_vs_loop()

