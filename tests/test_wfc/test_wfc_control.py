from __future__ import annotations

import numpy as np
import pytest

from minigrid.envs.wfc.wfclogic import control
from minigrid.envs.wfc.wfclogic.solver import Contradiction, StopEarly, TimedOut


@pytest.mark.parametrize("failure", [Contradiction, TimedOut])
def test_execute_wfc_retries_failed_attempt(mocker, failure):
    """A recoverable failure must not discard the remaining attempt budget."""
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
    assert run.call_count == 2


@pytest.mark.parametrize("failure", [Contradiction, TimedOut])
@pytest.mark.parametrize("attempt_limit", [1, 3])
def test_execute_wfc_exhausts_attempt_limit(mocker, failure, attempt_limit):
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
    run = mocker.patch.object(control, "run", side_effect=StopEarly())
    with pytest.raises(StopEarly):
        control.execute_wfc(
            image=np.zeros((3, 3, 3), dtype=np.uint8),
            output_size=(2, 2),
            attempt_limit=3,
        )
    assert run.call_count == 1
