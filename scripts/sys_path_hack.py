
if __name__ == "__main__" and __package__ is None:
    from sys import path
    import os
    from os.path import dirname as dir

    path.append(dir(path[0]))
    os.chdir(dir(path[0]))
    __package__ = "scripts"