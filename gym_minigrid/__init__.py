from gym.envs.registration import register

from gym_minigrid.minigrid import Wall


def register_minigrid_envs():
    # BlockedUnlockPickup
    # ----------------------------------------

    register(
        id="MiniGrid-BlockedUnlockPickup-v0",
        entry_point="gym_minigrid.envs:BlockedUnlockPickupEnv",
    )

    # LavaCrossing
    # ----------------------------------------
    register(
        id="MiniGrid-LavaCrossingS9N1-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 1},
    )

    register(
        id="MiniGrid-LavaCrossingS9N2-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 2},
    )

    register(
        id="MiniGrid-LavaCrossingS9N3-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 3},
    )

    register(
        id="MiniGrid-LavaCrossingS11N5-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 11, "num_crossings": 5},
    )

    # SimpleCrossing
    # ----------------------------------------

    register(
        id="MiniGrid-SimpleCrossingS9N1-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 1, "obstacle_type": Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS9N2-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 2, "obstacle_type": Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS9N3-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 3, "obstacle_type": Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS11N5-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 11, "num_crossings": 5, "obstacle_type": Wall},
    )

    # DistShift
    # ----------------------------------------

    register(
        id="MiniGrid-DistShift1-v0",
        entry_point="gym_minigrid.envs:DistShiftEnv",
        kwargs={"strip2_row": 2},
    )

    register(
        id="MiniGrid-DistShift2-v0",
        entry_point="gym_minigrid.envs:DistShiftEnv",
        kwargs={"strip2_row": 5},
    )

    # DoorKey
    # ----------------------------------------

    register(
        id="MiniGrid-DoorKey-5x5-v0",
        entry_point="gym_minigrid.envs:DoorKeyEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-DoorKey-6x6-v0",
        entry_point="gym_minigrid.envs:DoorKeyEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-DoorKey-8x8-v0",
        entry_point="gym_minigrid.envs:DoorKeyEnv",
        kwargs={"size": 8},
    )

    register(
        id="MiniGrid-DoorKey-16x16-v0",
        entry_point="gym_minigrid.envs:DoorKeyEnv",
        kwargs={"size": 16},
    )

    # Dynamic-Obstacles
    # ----------------------------------------

    register(
        id="MiniGrid-Dynamic-Obstacles-5x5-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 5, "n_obstacles": 2},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 5, "agent_start_pos": None, "n_obstacles": 2},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-6x6-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 6, "n_obstacles": 3},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 6, "agent_start_pos": None, "n_obstacles": 3},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-16x16-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 16, "n_obstacles": 8},
    )

    # Empty
    # ----------------------------------------

    register(
        id="MiniGrid-Empty-5x5-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-Empty-Random-5x5-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 5, "agent_start_pos": None},
    )

    register(
        id="MiniGrid-Empty-6x6-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-Empty-Random-6x6-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 6, "agent_start_pos": None},
    )

    register(
        id="MiniGrid-Empty-8x8-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
    )

    register(
        id="MiniGrid-Empty-16x16-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 16},
    )

    # Fetch
    # ----------------------------------------

    register(
        id="MiniGrid-Fetch-5x5-N2-v0",
        entry_point="gym_minigrid.envs:FetchEnv",
        kwargs={"size": 5, "numObjs": 2},
    )

    register(
        id="MiniGrid-Fetch-6x6-N2-v0",
        entry_point="gym_minigrid.envs:FetchEnv",
        kwargs={"size": 6, "numObjs": 2},
    )

    register(id="MiniGrid-Fetch-8x8-N3-v0", entry_point="gym_minigrid.envs:FetchEnv")

    # FourRooms
    # ----------------------------------------

    register(
        id="MiniGrid-FourRooms-v0",
        entry_point="gym_minigrid.envs:FourRoomsEnv",
    )

    # GoToDoor
    # ----------------------------------------

    register(
        id="MiniGrid-GoToDoor-5x5-v0",
        entry_point="gym_minigrid.envs:GoToDoorEnv",
    )

    register(
        id="MiniGrid-GoToDoor-6x6-v0",
        entry_point="gym_minigrid.envs:GoToDoorEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-GoToDoor-8x8-v0",
        entry_point="gym_minigrid.envs:GoToDoorEnv",
        kwargs={"size": 8},
    )

    # GoToObject
    # ----------------------------------------

    register(
        id="MiniGrid-GoToObject-6x6-N2-v0",
        entry_point="gym_minigrid.envs:GoToObjectEnv",
    )

    register(
        id="MiniGrid-GoToObject-8x8-N2-v0",
        entry_point="gym_minigrid.envs:GoToObjectEnv",
        kwargs={"size": 8, "numObjs": 2},
    )

    # KeyCorridor
    # ----------------------------------------

    register(
        id="MiniGrid-KeyCorridorS3R1-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 1},
    )

    register(
        id="MiniGrid-KeyCorridorS3R2-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 2},
    )

    register(
        id="MiniGrid-KeyCorridorS3R3-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS4R3-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 4, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS5R3-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 5, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS6R3-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 6, "num_rows": 3},
    )

    # LavaGap
    # ----------------------------------------

    register(
        id="MiniGrid-LavaGapS5-v0",
        entry_point="gym_minigrid.envs:LavaGapEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-LavaGapS6-v0",
        entry_point="gym_minigrid.envs:LavaGapEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-LavaGapS7-v0",
        entry_point="gym_minigrid.envs:LavaGapEnv",
        kwargs={"size": 7},
    )

    # LockedRoom
    # ----------------------------------------

    register(
        id="MiniGrid-LockedRoom-v0",
        entry_point="gym_minigrid.envs:LockedRoomEnv",
    )

    # Memory
    # ----------------------------------------

    register(
        id="MiniGrid-MemoryS17Random-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 17, "random_length": True},
    )

    register(
        id="MiniGrid-MemoryS13Random-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 13, "random_length": True},
    )

    register(
        id="MiniGrid-MemoryS13-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 13},
    )

    register(
        id="MiniGrid-MemoryS11-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 11},
    )

    register(
        id="MiniGrid-MemoryS9-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 9},
    )

    register(
        id="MiniGrid-MemoryS7-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 7},
    )

    # MultiRoom
    # ----------------------------------------

    register(
        id="MiniGrid-MultiRoom-N2-S4-v0",
        entry_point="gym_minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 2, "maxNumRooms": 2, "maxRoomSize": 4},
    )

    register(
        id="MiniGrid-MultiRoom-N4-S5-v0",
        entry_point="gym_minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 6, "maxNumRooms": 6, "maxRoomSize": 5},
    )

    register(
        id="MiniGrid-MultiRoom-N6-v0",
        entry_point="gym_minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 6, "maxNumRooms": 6},
    )

    # ObstructedMaze
    # ----------------------------------------

    register(
        id="MiniGrid-ObstructedMaze-1Dl-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_1Dlhb",
        kwargs={"key_in_box": False, "blocked": False},
    )

    register(
        id="MiniGrid-ObstructedMaze-1Dlh-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_1Dlhb",
        kwargs={"key_in_box": True, "blocked": False},
    )

    register(
        id="MiniGrid-ObstructedMaze-1Dlhb-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_1Dlhb",
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dl-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": False,
            "blocked": False,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dlh-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": False,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dlhb-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-1Q-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (1, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 1,
            "num_rooms_visited": 5,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Q-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 2,
            "num_rooms_visited": 11,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-Full-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
    )

    # Playground
    # ----------------------------------------

    register(
        id="MiniGrid-Playground-v0",
        entry_point="gym_minigrid.envs:PlaygroundEnv",
    )

    # PutNear
    # ----------------------------------------

    register(
        id="MiniGrid-PutNear-6x6-N2-v0",
        entry_point="gym_minigrid.envs:PutNearEnv",
    )

    register(
        id="MiniGrid-PutNear-8x8-N3-v0",
        entry_point="gym_minigrid.envs:PutNearEnv",
        kwargs={"size": 8, "numObjs": 3},
    )

    # RedBlueDoors
    # ----------------------------------------

    register(
        id="MiniGrid-RedBlueDoors-6x6-v0",
        entry_point="gym_minigrid.envs:RedBlueDoorEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-RedBlueDoors-8x8-v0",
        entry_point="gym_minigrid.envs:RedBlueDoorEnv",
    )

    # Unlock
    # ----------------------------------------

    register(id="MiniGrid-Unlock-v0", entry_point="gym_minigrid.envs:UnlockEnv")

    # UnlockPickup
    # ----------------------------------------

    register(
        id="MiniGrid-UnlockPickup-v0",
        entry_point="gym_minigrid.envs:UnlockPickupEnv",
    )
