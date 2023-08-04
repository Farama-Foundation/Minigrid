from enum import IntEnum, auto

class EnvEvent(IntEnum):
    stepBefore = auto()
    stepAfter = auto()
    stepParallel1 = auto()
    stepParallel2 = auto()


