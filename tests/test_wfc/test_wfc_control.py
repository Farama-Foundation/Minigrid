from __future__ import annotations

import numpy as np
import pytest

from minigrid.envs.wfc.config import WFC_PRESETS
from minigrid.envs.wfc.wfclogic import control
from minigrid.envs.wfc.wfclogic.solver import Contradiction, StopEarly, TimedOut


@pytest.mark.parametrize("failure", [Contradiction, TimedOut])
def test_execute_wfc_retries_failed_attempt(mocker, failure):
    """A recoverable failure must not discard the remaining attempt budget."""
    # Fail the first solve and succeed on the second, without relying on chance.
    run = mocker.patch.object(
        control,
        "run",
        side_effect=[failure(), np.zeros((2, 2), dtype=np.int64)],
    )
    image, stats = control.execute_wfc(
        image=np.zeros((3, 3, 3), dtype=np.uint8),
        output_size=(2, 2),
        attempt_limit=3,
        np_random=np.random.default_rng(0),
    )
    assert image is not None
    assert stats["attempts"] == 2
    assert stats["outcome"] == "success"
    # A successful retry should return immediately, leaving attempt three unused.
    assert run.call_count == 2


@pytest.mark.parametrize("failure", [Contradiction, TimedOut])
@pytest.mark.parametrize("attempt_limit", [1, 3])
def test_execute_wfc_exhausts_attempt_limit(mocker, failure, attempt_limit):
    # Every solve fails: use the whole budget, but never attempt an extra solve.
    run = mocker.patch.object(control, "run", side_effect=failure())
    image, stats = control.execute_wfc(
        image=np.zeros((3, 3, 3), dtype=np.uint8),
        output_size=(2, 2),
        attempt_limit=attempt_limit,
        np_random=np.random.default_rng(0),
    )
    assert image is None
    assert stats["attempts"] == attempt_limit
    assert run.call_count == attempt_limit


def test_execute_wfc_stop_early_does_not_retry(mocker):
    # StopEarly is an explicit abort, so it must propagate without retrying.
    run = mocker.patch.object(control, "run", side_effect=StopEarly())
    with pytest.raises(StopEarly):
        control.execute_wfc(
            image=np.zeros((3, 3, 3), dtype=np.uint8),
            output_size=(2, 2),
            attempt_limit=3,
        )
    assert run.call_count == 1


def test_rooms_fabric_recovers_from_seeded_contradiction():
    """Seed 1375 fails its first solve and needs the remaining attempt budget."""
    # Image loading is optional; keep the mocked retry tests usable without it.
    pytest.importorskip("imageio")
    kwargs = WFC_PRESETS["RoomsFabric"].wfc_kwargs
    images = []
    for _ in range(2):
        # This is the default 25x25 environment's interior, excluding its border.
        image, stats = control.execute_wfc(
            **kwargs,
            output_size=(23, 23),
            attempt_limit=1000,
            # Restart the random stream to check reproducibility across retries.
            np_random=np.random.default_rng(1375),
        )
        assert image is not None
        assert image.shape == (23, 23, 3)
        assert stats["outcome"] == "success"
        images.append(image)
    # Retrying must still produce the same output when the seed is repeated.
    assert np.array_equal(*images)
