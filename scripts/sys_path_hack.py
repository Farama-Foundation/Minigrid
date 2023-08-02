if __name__ == "__main__" and __package__ is None:
    import sys
    from sys import path
    import os
    from os.path import dirname as dir

    path.insert(0, dir(path[0]))
    os.chdir(dir(path[0]))
    __package__ = "scripts"