from gym_minigrid.minigrid import Ball
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register

class Unlock(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)

        self.place_agent(0, 0)

        self.door = door
        self.mission = "open the door"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                done = True

        return obs, reward, done, info

register(
    id='MiniGrid-Unlock-v0',
    entry_point='gym_minigrid.envs:Unlock'
)
