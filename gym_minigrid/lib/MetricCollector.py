from gym_minigrid.interfaces.IMultiPedestrianEnv import IMultiPedestrianEnv
from gym_minigrid.envs.pedestrian.EnvEvent import EnvEvent
import logging
from collections import defaultdict

class MetricCollector:

    def __init__(self, 
            env: IMultiPedestrianEnv,
            stepsToIgnoreAtTheBeginning = 100,
            stepsToRecord = 10000
        ):

        self.stepsToIgnoreAtTheBeginning = stepsToIgnoreAtTheBeginning
        self.stepsToRecord = stepsToRecord

        self.previousState = defaultdict(lambda : defaultdict(lambda : None)) # TODO assuming previous state does not have values past the last step.

        # subscribe to the stepAfter event so that it can read the updated world state
        logging.info("Attaching metric collector the the stepAfter event")
        env.subscribe(EnvEvent.stepAfter, self.handleStepAfter)

        self.stepStats = defaultdict(lambda: []) # average stuff in every step
    

    def handleStepAfter(self, env: IMultiPedestrianEnv):
        if env.step_count < self.stepsToIgnoreAtTheBeginning:
            logging.debug(f"MetricCollector: ignoring step {env.step_count}")
            return None

        # collect you metrics here
        logging.debug("MetricCollector: Collecting metrics")

        # collect speed
        self.collectSpeed(env)




    
    def getStatistics(self):
        return self.stepStats
        pass

    
    def collectSpeed(self, env):
        totalXSpeed = 0
        totalYSpeed = 0
        for agent in env.getAgents():
            if self.previousState[agent]["position"] is None:
                self.previousState[agent]["position"] = agent.position
            else:
                # now we have a previous position and a new position, 
                xSpeed = abs(agent.position[0] - self.previousState[agent]["position"][0])
                ySpeed = abs(agent.position[1] - self.previousState[agent]["position"][1])

                totalXSpeed += xSpeed
                totalYSpeed += ySpeed

                #reset
                self.previousState[agent]["position"] = agent.position
                self.previousState[agent]["xSpeed"] = xSpeed
                self.previousState[agent]["ySpeed"] = ySpeed
        
        self.stepStats["xSpeed"].append(totalXSpeed / len(env.getAgents()))
        self.stepStats["ySpeed"].append(totalYSpeed / len(env.getAgents()))
        

