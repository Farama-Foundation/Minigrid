from __future__ import annotations

from minigrid.envs.wfc.config import (
    WFC_PRESETS,
    WFC_PRESETS_INCONSISTENT,
    WFC_PRESETS_SLOW,
    WFCConfig,
)

# This is wrapped in a try-except block so the presets can be accessed for registration
# Otherwise, importing here will fail when networkx is not installed
try:
    from minigrid.envs.wfc.wfcenv import WFCEnv
except ImportError:

    class WFCEnv:
        """Dummy class to give a helpful error message when dependencies are missing"""

        def __init__(self, *args, **kwargs):
            from gymnasium.error import DependencyNotInstalled

            raise DependencyNotInstalled(
                'WFC dependencies are missing, please run `pip install "minigrid[wfc]"`'
            )
