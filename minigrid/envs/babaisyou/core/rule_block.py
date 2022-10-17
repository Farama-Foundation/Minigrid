import numpy as np

from minigrid.minigrid_env import WorldObj, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.utils.rendering import fill_coords, point_in_rect
from .utils import add_img_text


properties = [
    # 'can_overlap',
    'is_block',
    'can_push',
    'is_goal',
    'is_defeat'
]

name_mapping = {
    'fwall': 'wall',
    'fball': 'ball',
    'can_push': 'push',
    'is_block': 'stop',
    'is_goal': 'win',
    'is_defeat': 'lose'
}


def add_color_types(color_types):
    last_idx = len(COLOR_TO_IDX)-1
    COLOR_TO_IDX.update({
        t: last_idx+1+i
    for i, t in enumerate(color_types)})


add_color_types(['wall', 'is', 'stop', 'win', 'ball', 'lose'])


class RuleBlock(WorldObj):
    """
    By default, rule blocks can be pushed by the agent.
    """
    def __init__(self, name, type, color, can_push=True):
        super().__init__(type, color)
        self._can_push = can_push
        self.name = name = name_mapping.get(name, name)
        self.margin = 10
        img = np.zeros((96-2*self.margin, 96-2*self.margin, 3), np.uint8)
        add_img_text(img, name)
        self.img = img

    def can_overlap(self):
        return False

    def can_push(self):
        return self._can_push

    def render(self, img):
        fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), [235, 235, 235])
        img[self.margin:-self.margin, self.margin:-self.margin] = self.img

    # TODO: different encodings of the rule blocks for the agent observation
    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # RuleBlock characterized by their name instead of color
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.name], 0)


class RuleObject(RuleBlock):
    def __init__(self, object, can_push=True):
        super().__init__(object, 'rule_object', 'purple', can_push=can_push)
        self.object = object


class RuleProperty(RuleBlock):
    def __init__(self, property, can_push=True):
        super().__init__(property, 'rule_property', 'purple', can_push=can_push)
        assert property in properties, property
        self.property = property


class RuleIs(RuleBlock):
    def __init__(self, can_push=True):
        super().__init__('is', 'rule_is', 'purple', can_push=can_push)
