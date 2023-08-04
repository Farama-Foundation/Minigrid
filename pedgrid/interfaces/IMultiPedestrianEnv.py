from abc import abstractmethod
from pedgrid.envs.pedestrian.EnvEvent import EnvEvent

class IMultiPedestrianEnv:


    @abstractmethod
    def getAgents(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def unsubscribe(self, envEvent: EnvEvent, handler):
        pass

    @abstractmethod
    def subscribe(self, envEvent, handler):
        pass