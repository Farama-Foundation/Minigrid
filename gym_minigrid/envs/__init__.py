from gym.envs.registration import register as gym_register

from gym_minigrid.envs.empty import *
from gym_minigrid.envs.doorkey import *
from gym_minigrid.envs.multiroom import *
from gym_minigrid.envs.fetch import *
from gym_minigrid.envs.gotoobject import *
from gym_minigrid.envs.gotodoor import *
from gym_minigrid.envs.putnear import *
from gym_minigrid.envs.lockedroom import *
from gym_minigrid.envs.keycorridor import *
from gym_minigrid.envs.unlock import *
from gym_minigrid.envs.unlockpickup import *
from gym_minigrid.envs.blockedunlockpickup import *
from gym_minigrid.envs.playground_v0 import *
from gym_minigrid.envs.redbluedoors import *
from gym_minigrid.envs.obstructedmaze import *
from gym_minigrid.envs.memory import *
from gym_minigrid.envs.fourrooms import *
from gym_minigrid.envs.crossing import *

REWARD_THRESHOLD = 900

# Fetch
# --------------------------------------------------

# NOTE: see end of file for final type
ENV_LIST = []


def register(**kwargs):
    ENV_LIST.append(kwargs['id'])
    return gym_register(**kwargs)


register(
    id='MiniGridFetch5x5N2-v0',
    entry_point='gym_minigrid.envs:FetchEnv5x5N2',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridFetch6x6N2-v0',
    entry_point='gym_minigrid.envs:FetchEnv6x6N2',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridFetch8x8N3-v0',
    entry_point='gym_minigrid.envs:FetchEnv',
    reward_threshold=REWARD_THRESHOLD,
)

# FourRooms
# --------------------------------------------------

register(
    id='MiniGridFourRooms-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv',
    reward_threshold=REWARD_THRESHOLD,
)

# BlockedUnlockPickup
# --------------------------------------------------

register(
    id='MiniGridBlockedUnlockPickup-v0',
    entry_point='gym_minigrid.envs:BlockedUnlockPickup',
    reward_threshold=REWARD_THRESHOLD,
)

# LavaCrossing
# --------------------------------------------------

register(
    id='MiniGridLavaCrossingS9N1-v0',
    entry_point='gym_minigrid.envs:LavaCrossingEnv',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridLavaCrossingS9N2-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS9N2Env',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridLavaCrossingS9N3-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS9N3Env',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridLavaCrossingS11N5-v0',
    entry_point='gym_minigrid.envs:LavaCrossingS11N5Env',
    reward_threshold=REWARD_THRESHOLD,
)

# SimpleCrossing
# --------------------------------------------------

register(
    id='MiniGridSimpleCrossingS9N1-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingEnv',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridSimpleCrossingS9N2-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS9N2Env',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridSimpleCrossingS9N3-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS9N3Env',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridSimpleCrossingS11N5-v0',
    entry_point='gym_minigrid.envs:SimpleCrossingS11N5Env',
    reward_threshold=REWARD_THRESHOLD,
)

# DoorKey
# --------------------------------------------------

register(
    id='MiniGridDoorKey5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv5x5',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridDoorKey6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv6x6',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridDoorKey8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridDoorKey16x16-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv16x16',
    reward_threshold=REWARD_THRESHOLD,
)

# Empty
# --------------------------------------------------

register(
    id='MiniGridEmpty6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridEmpty8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridEmpty16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16',
    reward_threshold=REWARD_THRESHOLD,
)

# GoToDoor
# --------------------------------------------------

register(
    id='MiniGridGoToDoor5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorEnv',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridGoToDoor6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoor6x6Env',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridGoToDoor8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoor8x8Env',
    reward_threshold=REWARD_THRESHOLD,
)

# GoToObject
# --------------------------------------------------

register(
    id='MiniGridGoToObject6x6N2-v0',
    entry_point='gym_minigrid.envs:GoToObjectEnv',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridGoToObject8x8N2-v0',
    entry_point='gym_minigrid.envs:GotoEnv8x8N2',
    reward_threshold=REWARD_THRESHOLD,
)

# KeyCorridor
# --------------------------------------------------

register(
    id='MiniGridKeyCorridorS3R1-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS3R1',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridKeyCorridorS3R2-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS3R2',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridKeyCorridorS3R3-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS3R3',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridKeyCorridorS4R3-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS4R3',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridKeyCorridorS5R3-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS5R3',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridKeyCorridorS6R3-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS6R3',
    reward_threshold=REWARD_THRESHOLD,
)

# LockedRoom
# --------------------------------------------------

register(
    id='MiniGridLockedRoom-v0',
    entry_point='gym_minigrid.envs:LockedRoom',
    reward_threshold=REWARD_THRESHOLD,
)

# Memory
# --------------------------------------------------

register(
    id='MiniGridMemoryS7-v0',
    entry_point='gym_minigrid.envs:MemoryS7',
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id='MiniGridMemoryS9-v0',
    entry_point='gym_minigrid.envs:MemoryS9',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridMemoryS11-v0',
    entry_point='gym_minigrid.envs:MemoryS11',
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id='MiniGridMemoryS13-v0',
    entry_point='gym_minigrid.envs:MemoryS13',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridMemoryS13Random-v0',
    entry_point='gym_minigrid.envs:MemoryS13Random',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridMemoryS17Random-v0',
    entry_point='gym_minigrid.envs:MemoryS17Random',
    reward_threshold=REWARD_THRESHOLD,
)

# MultiRoom
# --------------------------------------------------

register(
    id='MiniGridMultiRoomN2S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN2S4',
    reward_threshold=1000.0,
)

register(
    id='MiniGridMultiRoomN6-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN6',
    reward_threshold=1000.0,
)

# ObstructedMaze
# --------------------------------------------------

register(
    id="MiniGridObstructedMaze1Dl-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_1Dl",
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id="MiniGridObstructedMaze1Dlh-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_1Dlh",
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id="MiniGridObstructedMaze1Dlhb-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_1Dlhb",
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id="MiniGridObstructedMaze2Dl-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_2Dl",
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id="MiniGridObstructedMaze2Dlh-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_2Dlh",
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id="MiniGridObstructedMaze2Dlhb-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_2Dlhb",
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id="MiniGridObstructedMaze1Q-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_1Q",
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id="MiniGridObstructedMaze2Q-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_2Q",
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id="MiniGridObstructedMazeFull-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_Full",
    reward_threshold=REWARD_THRESHOLD,
)

# PlayGround
# --------------------------------------------------

register(
    id='MiniGridPlayground-v0',
    entry_point='gym_minigrid.envs:PlaygroundV0',
    reward_threshold=REWARD_THRESHOLD,
)

# PutNear
# --------------------------------------------------

register(
    id='MiniGridPutNear6x6N2-v0',
    entry_point='gym_minigrid.envs:PutNearEnv',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridPutNear8x8N3-v0',
    entry_point='gym_minigrid.envs:PutNear8x8N3',
    reward_threshold=REWARD_THRESHOLD,
)

# RedBlueDoor
# --------------------------------------------------

register(
    id='MiniGridRedBlueDoors6x6-v0',
    entry_point='gym_minigrid.envs:RedBlueDoorEnv6x6',
    reward_threshold=REWARD_THRESHOLD,
)

register(
    id='MiniGridRedBlueDoors8x8-v0',
    entry_point='gym_minigrid.envs:RedBlueDoorEnv',
    reward_threshold=REWARD_THRESHOLD,
)

# UnLock
# --------------------------------------------------

register(
    id='MiniGridUnlock-v0',
    entry_point='gym_minigrid.envs:Unlock',
    reward_threshold=REWARD_THRESHOLD,
)

# UnlockPickup
# --------------------------------------------------

register(
    id='MiniGridUnlockPickup-v0',
    entry_point='gym_minigrid.envs:UnlockPickup',
    reward_threshold=REWARD_THRESHOLD,
)

# END

ENV_LIST = tuple(ENV_LIST)
