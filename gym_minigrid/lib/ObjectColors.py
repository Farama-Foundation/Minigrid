import numpy as np

class ObjectColors:
    # Map of color names to RGB values
    COLORS = {
        'red'   : np.array([255, 0, 0]),
        'green' : np.array([0, 255, 0]),
        'blue'  : np.array([0, 0, 255]),
        'purple': np.array([112, 39, 195]),
        'yellow': np.array([255, 255, 0]),
        'grey'  : np.array([100, 100, 100]),
        'white' : np.array([255, 255, 255]),
        'dark green' : np.array([21, 71, 52])
    }

    COLOR_NAMES = sorted(list(COLORS.keys()))

    # Used to map colors to integers
    COLOR_TO_IDX = {
        'red'   : 0,
        'green' : 1,
        'blue'  : 2,
        'purple': 3,
        'yellow': 4,
        'grey'  : 5,
        'white' : 6,
        'dark green' : 7
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    # Map of object type to integers
    OBJECT_TO_IDX = {
        'Vehicle'       : 4,
        'Lane'          : 7,
        'Sidewalk'      : 5,
        'Crosswalk'     : 6,
        'Building'      : 3,
        # 'key'           : 5,
        # 'ball'          : 6,
        # 'box'           : 7,
        # 'goal'          : 8,
        # 'lava'          : 9,
        # 'agent'         : 10,
    }

    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))