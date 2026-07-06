from __future__ import annotations

import pytest
from numpy import array, uint8

from minigrid.envs.wfc.config import PATTERN_PATH


class Resources:
    def get_pattern(self, image: str) -> str:
        return PATTERN_PATH / image


@pytest.fixture(scope="session")
def resources() -> Resources:
    return Resources()


@pytest.fixture(scope="session")
def img_redmaze(resources: Resources) -> array:
    try:
        import imageio  # type: ignore

        pattern = resources.get_pattern("RedMaze.png")
        img = imageio.v2.imread(pattern)
    except ImportError:
        b = [0, 0, 0]
        w = [255, 255, 255]
        r = [255, 0, 0]
        img = array(
            [
                [w, w, w, w],
                [w, b, b, b],
                [w, b, r, b],
                [w, b, b, b],
            ],
            dtype=uint8,
        )

    return img
