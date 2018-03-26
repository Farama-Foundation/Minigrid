from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Room:
    def __init__(
        self,
        top,
        size
    ):
        # Top-left corner and size (tuples)
        self.top = top
        self.size = size

        # List of door objects and door positions
        # Order of the doors is right, down, left, up
        self.doors = [None] * 4
        self.door_pos = [None] * 4

        # List of rooms this is connected to
        # Order of the neighbors is right, down, left, up
        self.neighbors = [None] * 4

        # Indicates if this room is locked
        self.locked = False

        # List of objects contained
        self.objs = []

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._randPos(
            topX + 1, topX + sizeX - 1,
            topY + 1, topY + sizeY - 1
        )

class RoomGrid(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This is meant to serve as a base class for other environments.
    """

    def __init__(
        self,
        room_size=6,
        num_cols=4,
        lockedRooms=False
    ):
        assert room_size > 0
        assert room_size >= 4
        assert num_cols > 0
        self.room_size = room_size
        self.num_cols = num_cols
        self.num_rows = num_cols
        self.lockedRooms = False

        grid_size = (room_size - 1) * num_cols + 1
        super().__init__(gridSize=grid_size, maxSteps=6*grid_size)

        self.reward_range = (0, 1)

    def room_from_pos(self, x, y):
        """Get the room a given position maps to"""

        assert x >= 0
        assert y >= 0

        i = x // self.room_size
        j = y // self.room_size

        assert i < self.num_cols
        assert j < self.num_rows

        return self.room_grid[j][i]

    def get_room(self, i, j):
        assert i < self.num_cols
        assert j < self.num_rows
        return self.room_grid[j][i]

    def _genGrid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        self.room_grid = []

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = Room(
                    (i * (self.room_size-1), j * (self.room_size-1)),
                    (self.room_size, self.room_size)
                )

                row.append(room)

                # Generate the walls for this room
                self.grid.wallRect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = room.top
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._randInt(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._randInt(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.startPos = (
            (self.num_cols // 2) * (self.room_size-1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size-1) + (self.room_size // 2)
        )
        self.startDir = 0

        # By default, this environment has no mission
        self.mission = ''

    def add_object(self, i, j, kind, color):
        """
        Add a new object to room (i, j)
        """

        # TODO: we probably want to add an Object.make helper function
        assert kind in ['key', 'ball', 'box']
        if kind == 'key':
            obj = Key(color)
        elif kind == 'ball':
            obj = Ball(color)
        elif kind == 'box':
            obj = Box(color)

        room = self.get_room(i, j)

        self.placeObj(obj, room.top, room.size)

        room.objs.append(obj)

        return obj

    def add_door(self, i, j, k, color, locked=False):
        """
        Add a door to a room, connecting it to a neighbor
        """

        room = self.get_room(i, j)
        assert room.doors[k] is None, "door already exists"

        if locked:
            door = LockedDoor(color)
            room.locked = True
        else:
            door = Door(color)

        self.grid.set(*room.door_pos[k], door)

        neighbor = room.neighbors[k]
        room.doors[k] = door
        neighbor.doors[(k+2) % 4] = door

    def connect_all(self):
        """
        Make sure that all rooms are reachable by the agent from its
        starting position
        """

        start_room = self.room_from_pos(*self.startPos)

        def find_reach():
            reach = set()
            stack = [start_room]
            while len(stack) > 0:
                room = stack.pop()
                if room in reach:
                    continue
                reach.add(room)
                for i in range(0, 4):
                    if room.doors[i]:
                        stack.append(room.neighbors[i])
            return reach

        while True:
            # If all rooms are reachable, stop
            reach = find_reach()
            if len(reach) == self.num_rows * self.num_cols:
                break

            # Pick a random room and door position
            i = self._randInt(0, self.num_cols)
            j = self._randInt(0, self.num_rows)
            k = self._randInt(0, 4)
            room = self.get_room(i, j)

            # If there is already a door there, skip
            if not room.door_pos[k] or room.doors[k]:
                continue

            if room.locked or room.neighbors[k].locked:
                continue

            color = self._randElem(COLOR_NAMES)
            self.add_door(i, j, k, color)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

register(
    id='MiniGrid-RoomGrid-v0',
    entry_point='gym_minigrid.envs:RoomGrid'
)
