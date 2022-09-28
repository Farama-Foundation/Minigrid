from typing import Any, Callable, Optional, Union

from gymnasium import spaces
from gymnasium.utils import seeding


def check_if_no_duplicate(duplicate_list: list) -> bool:
    """Check if given list contains any duplicates"""
    return len(set(duplicate_list)) == len(duplicate_list)


class MissionSpace(spaces.Space[str]):
    r"""A space representing a mission for the Gym-Minigrid environments.
    The space allows generating random mission strings constructed with an input placeholder list.
    Example Usage::
        >>> observation_space = MissionSpace(mission_func=lambda color: f"Get the {color} ball.",
                                                ordered_placeholders=[["green", "blue"]])
        >>> observation_space.sample()
            "Get the green ball."
        >>> observation_space = MissionSpace(mission_func=lambda : "Get the ball.".,
                                                ordered_placeholders=None)
        >>> observation_space.sample()
            "Get the ball."
    """

    def __init__(
        self,
        mission_func: Callable[..., str],
        ordered_placeholders: Optional["list[list[str]]"] = None,
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ):
        r"""Constructor of :class:`MissionSpace` space.

        Args:
            mission_func (lambda _placeholders(str): _mission(str)): Function that generates a mission string from random placeholders.
            ordered_placeholders (Optional["list[list[str]]"]): List of lists of placeholders ordered in placing order in the mission function mission_func.
            seed: seed: The seed for sampling from the space.
        """
        # Check that the ordered placeholders and mission function are well defined.
        if ordered_placeholders is not None:
            assert (
                len(ordered_placeholders) == mission_func.__code__.co_argcount
            ), f"The number of placeholders {len(ordered_placeholders)} is different from the number of parameters in the mission function {mission_func.__code__.co_argcount}."
            for placeholder_list in ordered_placeholders:
                assert check_if_no_duplicate(
                    placeholder_list
                ), "Make sure that the placeholders don't have any duplicate values."
        else:
            assert (
                mission_func.__code__.co_argcount == 0
            ), f"If the ordered placeholders are {ordered_placeholders}, the mission function shouldn't have any parameters."

        self.ordered_placeholders = ordered_placeholders
        self.mission_func = mission_func

        super().__init__(dtype=str, seed=seed)

        # Check that mission_func returns a string
        sampled_mission = self.sample()
        assert isinstance(
            sampled_mission, str
        ), f"mission_func must return type str not {type(sampled_mission)}"

    def sample(self) -> str:
        """Sample a random mission string."""
        if self.ordered_placeholders is not None:
            placeholders = []
            for rand_var_list in self.ordered_placeholders:
                idx = self.np_random.integers(0, len(rand_var_list))

                placeholders.append(rand_var_list[idx])

            return self.mission_func(*placeholders)
        else:
            return self.mission_func()

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        # Store a list of all the placeholders from self.ordered_placeholders that appear in x
        if self.ordered_placeholders is not None:
            check_placeholder_list = []
            for placeholder_list in self.ordered_placeholders:
                for placeholder in placeholder_list:
                    if placeholder in x:
                        check_placeholder_list.append(placeholder)

            # Remove duplicates from the list
            check_placeholder_list = list(set(check_placeholder_list))

            start_id_placeholder = []
            end_id_placeholder = []
            # Get the starting and ending id of the identified placeholders with possible duplicates
            new_check_placeholder_list = []
            for placeholder in check_placeholder_list:
                new_start_id_placeholder = [
                    i for i in range(len(x)) if x.startswith(placeholder, i)
                ]
                new_check_placeholder_list += [placeholder] * len(
                    new_start_id_placeholder
                )
                end_id_placeholder += [
                    start_id + len(placeholder) - 1
                    for start_id in new_start_id_placeholder
                ]
                start_id_placeholder += new_start_id_placeholder

            # Order by starting id the placeholders
            ordered_placeholder_list = sorted(
                zip(
                    start_id_placeholder, end_id_placeholder, new_check_placeholder_list
                )
            )

            # Check for repeated placeholders contained in each other
            remove_placeholder_id = []
            for i, placeholder_1 in enumerate(ordered_placeholder_list):
                starting_id = i + 1
                for j, placeholder_2 in enumerate(
                    ordered_placeholder_list[starting_id:]
                ):
                    # Check if place holder ids overlap and keep the longest
                    if max(placeholder_1[0], placeholder_2[0]) < min(
                        placeholder_1[1], placeholder_2[1]
                    ):
                        remove_placeholder = min(
                            placeholder_1[2], placeholder_2[2], key=len
                        )
                        if remove_placeholder == placeholder_1[2]:
                            remove_placeholder_id.append(i)
                        else:
                            remove_placeholder_id.append(i + j + 1)
            for id in remove_placeholder_id:
                del ordered_placeholder_list[id]

            final_placeholders = [
                placeholder[2] for placeholder in ordered_placeholder_list
            ]

            # Check that the identified final placeholders are in the same order as the original placeholders.
            for orered_placeholder, final_placeholder in zip(
                self.ordered_placeholders, final_placeholders
            ):
                if final_placeholder in orered_placeholder:
                    continue
                else:
                    return False
            try:
                mission_string_with_placeholders = self.mission_func(
                    *final_placeholders
                )
            except Exception as e:
                print(
                    f"{x} is not contained in MissionSpace due to the following exception: {e}"
                )
                return False

            return bool(mission_string_with_placeholders == x)

        else:
            return bool(self.mission_func() == x)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"MissionSpace({self.mission_func}, {self.ordered_placeholders})"

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        if isinstance(other, MissionSpace):

            # Check that place holder lists are the same
            if self.ordered_placeholders is not None:
                # Check length
                if (len(self.order_placeholder) == len(other.order_placeholder)) and (
                    all(
                        set(i) == set(j)
                        for i, j in zip(self.order_placeholder, other.order_placeholder)
                    )
                ):
                    # Check mission string is the same with dummy space placeholders
                    test_placeholders = [""] * len(self.order_placeholder)
                    mission = self.mission_func(*test_placeholders)
                    other_mission = other.mission_func(*test_placeholders)
                    return mission == other_mission
            else:

                # Check that other is also None
                if other.ordered_placeholders is None:

                    # Check mission string is the same
                    mission = self.mission_func()
                    other_mission = other.mission_func()
                    return mission == other_mission

        # If none of the statements above return then False
        return False
