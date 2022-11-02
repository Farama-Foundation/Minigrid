from gym_minigrid.babaisyou import BabaIsYouGrid
from gym_minigrid.envs.babaisyou.core.flexible_world_object import FBall, Baba
from gym_minigrid.envs.babaisyou.goto import BaseGridEnv


class TestWinLoseEnv(BaseGridEnv):
    def __init__(self, size=8, is_lose=False, **kwargs):
        self.size = size
        self.is_lose = is_lose
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        self.rule_pos = [(2, 2), (3, 2), (4, 2)]
        self.agent_start_pos = (4, 4)
        ball_pos = (5, 4)

        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_rule(obj='baba', property='is_agent', positions=[(1, 1), (2, 1), (3, 1)])
        if not self.is_lose:
            self.put_rule(obj='fball', property='is_goal', positions=self.rule_pos)
        else:
            self.put_rule(obj='fball', property='is_defeat', positions=self.rule_pos)

        self.put_obj(FBall(), *ball_pos)
        self.put_obj(Baba(), *self.agent_start_pos)
        self.place_agent()


def test_win():
    env = TestWinLoseEnv()
    env.reset()
    obs, reward, done, info = env.step(env.actions.right)
    assert done
    assert reward > 0


def test_lose():
    env = TestWinLoseEnv(is_lose=True)
    env.reset()
    obs, reward, done, info = env.step(env.actions.right)
    assert done
    assert reward < 0
