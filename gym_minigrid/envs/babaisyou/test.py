from gym_minigrid.babaisyou import BabaIsYouEnv, BabaIsYouGrid
from gym_minigrid.minigrid import Grid
from .core.flexible_world_object import RuleProperty, RuleIs, RuleObject


class TestEnv(BabaIsYouEnv):
    def __init__(self, size=5, **kwargs):
        self.size = size
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = BabaIsYouGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(RuleIs(), *(2, 2))

        self.place_agent()
