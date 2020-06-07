"""Utility data and functions for WFC"""

import collections
import numpy as np


CoordXY = collections.namedtuple("coords_xy", ["x", "y"])
CoordRC = collections.namedtuple("coords_rc", ["row", "column"])


def hash_downto(a, rank, seed=0):
    state = np.random.RandomState(seed)
    assert rank < len(a.shape)
    # print((np.prod(a.shape[:rank]),-1))
    # print(np.array([np.prod(a.shape[:rank]),-1], dtype=np.int64).dtype)
    u = a.reshape(
        np.array([np.prod(a.shape[:rank]), -1], dtype=np.int64)
    )  # change because lists are by default float64?
    # u = a.reshape((np.prod(a.shape[:rank]),-1))
    v = state.randint(1 - (1 << 63), 1 << 63, np.prod(a.shape[rank:]), dtype="int64")
    return np.inner(u, v).reshape(a.shape[:rank]).astype("int64")


try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False


def load_visualizer(wfc_ns):
    if IN_COLAB:
        from google.colab import files

        uploaded = files.upload()
        for fn in uploaded.keys():
            print(
                'User uploaded file "{name}" with length {length} bytes'.format(
                    name=fn, length=len(uploaded[fn])
                )
            )
    else:
        import matplotlib
        import matplotlib.pylab
        from matplotlib.pyplot import figure
        from matplotlib.pyplot import subplot
        from matplotlib.pyplot import title
        from matplotlib.pyplot import matshow

    wfc_ns.img_filename = f"images/{wfc_ns.img_filename}"
    return wfc_ns


def find_pattern_center(wfc_ns):
    # wfc_ns.pattern_center = (math.floor((wfc_ns.pattern_width - 1) / 2), math.floor((wfc_ns.pattern_width - 1) / 2))
    wfc_ns.pattern_center = (0, 0)
    return wfc_ns
