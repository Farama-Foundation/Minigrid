import numpy as np

from .core.flexible_world_object import FBall, FWall, RuleProperty, RuleIs, RuleObject
from .core.utils import grid_random_position
from gym_minigrid.minigrid import MiniGridEnv, MissionSpace, Grid
from ...babaisyou import BabaIsYouGrid

RuleObjPos = tuple[int, int]
RuleIsPos = tuple[int, int]
RulePropPos = tuple[int, int]


class BaseGridEnv(MiniGridEnv):
    def __init__(self, size, **kwargs):
        self.size = size
        mission_space = MissionSpace(
            mission_func=lambda: ""
        )
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def put_rule(self, obj: str, property: str, positions: list[RuleObjPos, RuleIsPos, RulePropPos], can_push=True):
        self.put_obj(RuleObject(obj, can_push=can_push), *positions[0])
        self.put_obj(RuleIs(can_push=can_push), *positions[1])
        self.put_obj(RuleProperty(property, can_push=can_push), *positions[2])


def random_rule_pos(size, margin):
    rule_pos = grid_random_position(size, n_samples=1, margin=margin)[0]
    rule_pos = [(rule_pos[0]-1, rule_pos[1]), rule_pos, (rule_pos[0]+1, rule_pos[1])]
    return rule_pos


class GoToObjEnv(BaseGridEnv):
    def __init__(self, size=8, agent_start_dir=0, random_rule_position=False, push_rule_block=False, **kwargs):
        self.size = size
        self.agent_start_dir = agent_start_dir
        self.random_rule_position = random_rule_position
        self.push_rule_block = push_rule_block
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        # rule blocks position
        if self.random_rule_position:
            # self.rule_pos = random_position(self.size, n_samples=3, margin=3)
            self.rule_pos = random_rule_pos(self.size, margin=2)
        else:
            self.rule_pos = [(2, 2), (3, 2), (4, 2)]

        # agent and ball positions
        agent_start_pos, self.ball_pos = grid_random_position(self.size, n_samples=2, margin=1)
        while agent_start_pos in self.rule_pos or self.ball_pos in self.rule_pos:
            agent_start_pos, self.ball_pos = grid_random_position(self.size, n_samples=2, margin=1)

        self.agent_start_pos = agent_start_pos

        # Create an empty grid
        self.grid = BabaIsYouGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.put_obj(FBall(), width - 2, height - 2)
        self.put_obj(FBall(), *self.ball_pos)

        # rule_y = 2
        # self.put_obj(RuleObject('fball', can_push=self.push_rule_block), *self.rule_pos[0])
        # self.put_obj(RuleIs(can_push=self.push_rule_block), *self.rule_pos[1])
        # self.put_obj(RuleProperty('is_goal', can_push=self.push_rule_block), *self.rule_pos[2])
        # self.put_obj(RuleProperty('is_defeat', can_push=self.push_rule_block), *self.rule_pos[2])
        self.put_rule(obj='fball', property='is_goal', positions=self.rule_pos, can_push=self.push_rule_block)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


class GoToWinObjEnv(BaseGridEnv):
    def __init__(self, size=8, **kwargs):
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # randomly sample which object is win
        is_ball_win = np.random.choice([0, 1])

        self.rule1_pos = [(1, 1), (2, 1), (3, 1)]
        self.rule2_pos = [(1, 2), (2, 2), (3, 2)]

        ball_property = 'is_goal' if is_ball_win else 'is_defeat'
        wall_property = 'is_defeat' if is_ball_win else 'is_goal'
        self.put_rule('fball', ball_property, self.rule1_pos)
        self.put_rule('fwall', wall_property, self.rule2_pos)

        wall_pos, ball_pos = grid_random_position(self.size, n_samples=2, margin=1,
                                                  exclude_pos=[*self.rule1_pos, *self.rule2_pos])

        self.put_obj(FWall(), *wall_pos)
        self.put_obj(FBall(), *ball_pos)

        self.place_agent(rand_dir=True)
