import wfc.wfc_solver
from wfc.wfc_solver import run
        
    
    

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

