"""Breaks an image into consituant tiles."""

from __future__ import annotations

from minigrid.envs.wfc.wfclogic import tiles as wfc_tiles


def test_image_to_tile(img_redmaze) -> None:
    img = img_redmaze
    tiles = wfc_tiles.image_to_tiles(img, 1)
    assert tiles[2][2][0][0][0] == 255
    assert tiles[2][2][0][0][1] == 0


def test_make_tile_catalog(img_redmaze) -> None:
    img = img_redmaze
    tc, tg, cl, ut = wfc_tiles.make_tile_catalog(img, 1)
    assert ut[1][0] == 7
