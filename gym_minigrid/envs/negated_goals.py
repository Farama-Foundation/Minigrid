from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        num_distractors=6,
        types = ('key', 'ball', 'box'),
        colors = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_distractors = num_distractors

        self.types = ('key', 'ball', 'box')
        self.colors = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')

        self.type_dict = {
            'key': Key,
            'ball': Ball,
            'box': Box
        }

        self.base_templates = [
            "Go to the object that is <not><desc>",
            "Navigate to the object that is <not><desc>",
            "Find the object that is <not><desc>",
            "Pick up the object that is <not><desc>",
            "The goal is object that is <not><desc>",
            "The object that is <not><desc> is the goal",
        ]

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def get_oracle_goal(self):
        return self.target_cell

    def direct_mission(self):
        # 1. Choose a target type and color
        # 2. Create N distraction objects that cannot share the same tuple
        type_idx = self._rand_int(0, len(self.types))
        color_idx = self._rand_int(0, len(self.colors))
        self.target_type = self.types[type_idx]
        self.target_color = self.colors[color_idx]
        target_obj = self.type_dict[self.target_type](self.target_color)
        self.target_cell = self.place_obj(target_obj)

        distractor_type_opts = self.types[:type_idx] + self.types[type_idx + 1:]
        distractor_color_opts = self.colors[:color_idx] + self.colors[color_idx + 1:]

        for d in range(self.num_distractors):
            obj_type = self._rand_elem(distractor_type_opts)
            obj_color = self._rand_elem(distractor_color_opts)

            obj = self.type_dict[obj_type](obj_color)

            self.place_obj(obj)

        template = self._rand_elem(self.base_templates)
        mission = template.replace("<not>", "").replace("<desc>", self.target_color)
        return mission

    def negated_mission(self):
        # 1. Choose a target type and color
        # 2. Choose a negated description
        # 3. Create N distraction objects that share the negated description
        type_idx = self._rand_int(0, len(self.types))
        color_idx = self._rand_int(0, len(self.colors))
        self.target_type = self.types[type_idx]
        self.target_color = self.colors[color_idx]
        target_obj = self.type_dict[self.target_type](self.target_color)
        self.target_cell = self.place_obj(target_obj)

        distractor_type_opts = self.types[:type_idx] + self.types[type_idx + 1:]
        distractor_color_opts = self.colors[:color_idx] + self.colors[color_idx + 1:]

        dist_color = self._rand_elem(distractor_color_opts)

        for d in range(self.num_distractors):
            obj_type = self._rand_elem(distractor_type_opts)
            obj = self.type_dict[obj_type](dist_color)
            self.place_obj(obj)

        template = self._rand_elem(self.base_templates)
        mission = template.replace("<not>", "not ").replace("<desc>", dist_color)
        return mission


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Randomize the player start position and orientation
        self.place_agent()

        if self._rand_bool():
            self.mission = self.direct_mission()
        else:
            self.mission = self.negated_mission()

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.color == self.target_color and \
                    self.carrying.type == self.target_type:
                reward = self._reward()
                done = True
            else:
                reward = -1
                done = True

        return obs, reward, done, info

class NegatedSimple(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, **kwargs)

register(
    id='MiniGrid-Negated-Simple-v0',
    entry_point='gym_minigrid.envs:NegatedSimple'
)
