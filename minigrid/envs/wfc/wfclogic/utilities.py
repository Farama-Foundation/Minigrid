"""Utility data and functions for WFC"""
from __future__ import annotations

import collections
import logging
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

CoordXY = collections.namedtuple("CoordXY", ["x", "y"])
CoordRC = collections.namedtuple("CoordRC", ["row", "column"])


def hash_downto(a: NDArray[np.integer], rank: int, seed: Any=0) -> NDArray[np.int64]:
    state = np.random.RandomState(seed)
    assert rank < len(a.shape)
    # logger.debug((np.prod(a.shape[:rank]),-1))
    # logger.debug(np.array([np.prod(a.shape[:rank]),-1], dtype=np.int64).dtype)
    u: NDArray[np.integer] = a.reshape((np.prod(a.shape[:rank], dtype=np.int64), -1))
    # u = a.reshape((np.prod(a.shape[:rank]),-1))
    v = state.randint(1 - (1 << 63), 1 << 63, np.prod(a.shape[rank:]), dtype=np.int64)
    return np.asarray(np.inner(u, v).reshape(a.shape[:rank]), dtype=np.int64)


try:
    import google.colab  # type: ignore

    IN_COLAB = True
except:
    IN_COLAB = False


def load_visualizer(wfc_ns):
    if IN_COLAB:
        from google.colab import files  # type: ignore

        uploaded = files.upload()
        for fn in uploaded.keys():
            logger.debug(
                'User uploaded file "{name}" with length {length} bytes'.format(
                    name=fn, length=len(uploaded[fn])
                )
            )
    else:
        import matplotlib  # type: ignore
        import matplotlib.pylab  # type: ignore
        from matplotlib.pyplot import figure, subplot, title, matshow  # type: ignore

    wfc_ns.img_filename = f"images/{wfc_ns.img_filename}"
    return wfc_ns


def find_pattern_center(wfc_ns):
    # wfc_ns.pattern_center = (math.floor((wfc_ns.pattern_width - 1) / 2), math.floor((wfc_ns.pattern_width - 1) / 2))
    wfc_ns.pattern_center = (0, 0)
    return wfc_ns
