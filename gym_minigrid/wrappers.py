import math
import operator
from functools import reduce

import gym
import numpy as np
from gym import spaces
from gym.core import ObservationWrapper, Wrapper

from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX, Goal


class ReseedWrapper(Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class ActionBonus(Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class StateBonus(Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"]


class OneHotPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space["image"].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        new_image_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {**obs, "image": out}


class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(mode="rgb_array", highlight=True, tile_size=self.tile_size)

        return {**obs, "image": rgb_img}


class RGBImgPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env.get_obs_render(obs["image"], tile_size=self.tile_size)

        return {**obs, "image": rgb_img_partial}


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**obs, "image": full_grid}


class DictObservationSpaceWrapper(ObservationWrapper):
    """
    Transforms the observation space (that has a textual component) to a fully numerical observation space,
    where the textual instructions are replaced by arrays representing the indices of each word in a fixed vocabulary.
    """

    def __init__(self, env, max_words_in_mission=50, word_dict=None):
        """
        max_words_in_mission is the length of the array to represent a mission, value 0 for missing words
        word_dict is a dictionary of words to use (keys=words, values=indices from 1 to < max_words_in_mission),
                  if None, use the Minigrid language
        """
        super().__init__(env)

        if word_dict is None:
            word_dict = self.get_minigrid_words()

        self.max_words_in_mission = max_words_in_mission
        self.word_dict = word_dict

        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": spaces.MultiDiscrete(
                    [len(self.word_dict.keys())] * max_words_in_mission
                ),
            }
        )

    @staticmethod
    def get_minigrid_words():
        colors = ["red", "green", "blue", "yellow", "purple", "grey"]
        objects = [
            "unseen",
            "empty",
            "wall",
            "floor",
            "box",
            "key",
            "ball",
            "door",
            "goal",
            "agent",
            "lava",
        ]

        verbs = [
            "pick",
            "avoid",
            "get",
            "find",
            "put",
            "use",
            "open",
            "go",
            "fetch",
            "reach",
            "unlock",
            "traverse",
        ]

        extra_words = [
            "up",
            "the",
            "a",
            "at",
            ",",
            "square",
            "and",
            "then",
            "to",
            "of",
            "rooms",
            "near",
            "opening",
            "must",
            "you",
            "matching",
            "end",
            "hallway",
            "object",
            "from",
            "room",
        ]

        all_words = colors + objects + verbs + extra_words
        assert len(all_words) == len(set(all_words))
        return {word: i for i, word in enumerate(all_words)}

    def string_to_indices(self, string, offset=1):
        """
        Convert a string to a list of indices.
        """
        indices = []
        # adding space before and after commas
        string = string.replace(",", " , ")
        for word in string.split():
            if word in self.word_dict.keys():
                indices.append(self.word_dict[word] + offset)
            else:
                raise ValueError(f"Unknown word: {word}")
        return indices

    def observation(self, obs):
        obs["mission"] = self.string_to_indices(obs["mission"])
        assert len(obs["mission"]) < self.max_words_in_mission
        obs["mission"] += [0] * (self.max_words_in_mission - len(obs["mission"]))

        return obs


class FlatObsWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            dtype="uint8",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert (
                len(mission) <= self.maxStrLen
            ), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            strArray = np.zeros(
                shape=(self.maxStrLen, self.numCharCodes), dtype="float32"
            )

            for idx, ch in enumerate(mission):
                if ch >= "a" and ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(
                        f"Character {ch} is not available in mission string."
                    )
                assert chNo < self.numCharCodes, "%s : %d" % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs


class ViewSizeWrapper(Wrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "image": image}


class DirectionObsWrapper(ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}
    """

    def __init__(self, env, type="slope"):
        super().__init__(env)
        self.goal_position: tuple = None
        self.type = type

    def reset(self):
        obs = self.env.reset()
        if not self.goal_position:
            self.goal_position = [
                x for x, y in enumerate(self.grid.grid) if isinstance(y, Goal)
            ]
            # in case there are multiple goals , needs to be handled for other env types
            if len(self.goal_position) >= 1:
                self.goal_position = (
                    int(self.goal_position[0] / self.height),
                    self.goal_position[0] % self.width,
                )
        return obs

    def observation(self, obs):
        slope = np.divide(
            self.goal_position[1] - self.agent_pos[1],
            self.goal_position[0] - self.agent_pos[0],
        )
        obs["goal_direction"] = np.arctan(slope) if self.type == "angle" else slope
        return obs


class SymbolicObsWrapper(ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    The symbol is a triple of (X, Y, IDX), where X and Y are
    the coordinates on the grid, and IDX is the id of the object.
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        )
        w, h = self.width, self.height
        grid = np.mgrid[:w, :h]
        grid = np.concatenate([grid, objects.reshape(1, w, h)])
        grid = np.transpose(grid, (1, 2, 0))
        obs["image"] = grid
        return obs
