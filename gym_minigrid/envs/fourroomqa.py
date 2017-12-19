from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Room:
    def __init__(
        self,
        top,
        size,
        color,
        objects
    ):
        self.top = top
        self.size = size

        # Color of the room
        self.color = color

        # List of objects contained
        self.objects = objects

class FourRoomQAEnv(MiniGridEnv):
    """
    Environment to experiment with embodied question answering
    https://arxiv.org/abs/1711.11543
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2
        toggle = 3
        say = 4

    def __init__(self, size=16):
        assert size >= 10
        super(FourRoomQAEnv, self).__init__(gridSize=size, maxSteps=8*size)

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # TODO: dictionary action_space, to include answer sentence?
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # TODO: dictionary observation_space, to include question?

    def _randPos(self, room, border=1):
        return (
            self._randInt(
                room.top[0] + border,
                room.top[0] + room.size[0] - border
            ),
            self._randInt(
                room.top[1] + border,
                room.top[1] + room.size[1] - border
            ),
        )

    def _genGrid(self, width, height):
        grid = Grid(width, height)

        # Horizontal and vertical split indices
        vSplitIdx = self._randInt(5, width-4)
        hSplitIdx = self._randInt(5, height-4)

        # Create the four rooms
        self.rooms = []
        self.rooms.append(Room(
            (0, 0),
            (vSplitIdx, hSplitIdx),
            'red',
            []
        ))
        self.rooms.append(Room(
            (vSplitIdx, 0),
            (width - vSplitIdx, hSplitIdx),
            'purple',
            []
        ))
        self.rooms.append(Room(
            (0, hSplitIdx),
            (vSplitIdx, height - hSplitIdx),
            'blue',
            []
        ))
        self.rooms.append(Room(
            (vSplitIdx, hSplitIdx),
            (width - vSplitIdx, height - hSplitIdx),
            'yellow',
            []
        ))

        # Place the room walls
        for room in self.rooms:
            x, y = room.top
            w, h = room.size

            # Horizontal walls
            for i in range(w):
                grid.set(x + i, y, Wall(room.color))
                grid.set(x + i, y + h - 1, Wall(room.color))

            # Vertical walls
            for j in range(h):
                grid.set(x, y + j, Wall(room.color))
                grid.set(x + w - 1, y + j, Wall(room.color))

        # Place wall openings connecting the rooms
        hIdx = self._randInt(1, hSplitIdx-1)
        grid.set(vSplitIdx, hIdx, None)
        grid.set(vSplitIdx-1, hIdx, None)
        hIdx = self._randInt(hSplitIdx+1, height-1)
        grid.set(vSplitIdx, hIdx, None)
        grid.set(vSplitIdx-1, hIdx, None)

        vIdx = self._randInt(1, vSplitIdx-1)
        grid.set(vIdx, hSplitIdx, None)
        grid.set(vIdx, hSplitIdx-1, None)
        vIdx = self._randInt(vSplitIdx+1, width-1)
        grid.set(vIdx, hSplitIdx, None)
        grid.set(vIdx, hSplitIdx-1, None)

        # Select a random position for the agent to start at
        self.startDir = self._randInt(0, 4)
        room = self._randElem(self.rooms)
        self.startPos = self._randPos(room)

        # Possible object types and colors
        types = ['key', 'ball']
        colors = list(COLORS.keys())

        # Place a number of random objects
        numObjs = self._randInt(1, 10)
        for i in range(0, numObjs):
            # Generate a random object
            objType = self._randElem(types)
            objColor = self._randElem(colors)
            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)

            # Pick a random position that doesn't overlap with anything
            while True:
                room = self._randElem(self.rooms)
                pos = self._randPos(room, border=2)
                if pos == self.startPos:
                    continue
                if grid.get(*pos) != None:
                    continue
                grid.set(*pos, obj)
                break

            room.objects.append(obj)

        # Question examples:
        # - What color is the X?
        # - What color is the X in the ROOM?
        # - What room is the X located in?
        # - What color is the X in the blue room?
        # - How many rooms contain chairs?
        # - How many keys are there in the yellow room?
        # - How many <OBJs> in the <ROOM>?

        # Pick a random room to be the subject of the question
        room = self._randElem(self.rooms)

        # Pick a random object type
        objType = self._randElem(types)

        # Count the number of objects of this type in the room
        count = len(list(filter(lambda o: o.type == objType, room.objects)))

        # TODO: identify unique objects

        self.question = "Are there any %ss in the %s room?" % (objType, room.color)
        self.answer = "yes" if count > 0 else "no"

        # TODO: how many X in the Y room question type







        print(self.question)
        print(self.answer)

        return grid

    def _reset(self):
        obs = MiniGridEnv._reset(self)

        obs = {
            'image': obs,
            'question': self.question
        }

        return obs

    def _step(self, action):
        if isinstance(action, dict):
            answer = action['answer']
            action = action['action']
        else:
            answer = ''

        obs, reward, done, info = MiniGridEnv._step(self, action)

        if answer == self.answer:
            reward = 1000 - self.stepCount
            done = True

        obs = {
            'image': obs,
            'question': self.question
        }

        return obs, reward, done, info

register(
    id='MiniGrid-FourRoomQA-v0',
    entry_point='gym_minigrid.envs:FourRoomQAEnv'
)
