"""Convert input data to adjacency information"""

#import itertools
#import math
#import matplotlib
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure, subplot, subplots, title, matshow
#from wfc.wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center
#from wfc.wfc_tiles import tiles_to_images

import imageio

filename = "images/samples/Red Maze.png"
img = imageio.imread(filename)
print(img)
