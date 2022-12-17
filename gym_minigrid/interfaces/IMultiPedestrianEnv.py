from abc import abstractmethod

class IMultiPedestrianEnv:

    @abstractmethod
    def step(self, action):
        pass