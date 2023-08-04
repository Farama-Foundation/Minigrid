from pedgrid.interfaces.IMultiPedestrianEnv import IMultiPedestrianEnv
from pedgrid.envs.pedestrian.EnvEvent import EnvEvent
import logging
from collections import defaultdict

class MetricCollector:

    def __init__(self, 
            env: IMultiPedestrianEnv,
            stepsToIgnoreAtTheBeginning = 0,
            stepsToRecord = 1100
        ):

        self.stepsToIgnoreAtTheBeginning = stepsToIgnoreAtTheBeginning
        self.stepsToRecord = stepsToRecord

        self.previousState = defaultdict(lambda : defaultdict(lambda : None))
        # TODO assuming previous state does not have values past the last step.
        # ^ resolved by using handleStepBefore initially

        # subscribe to the stepAfter event so that it can read the updated world state
        logging.info("Attaching metric collector the the stepAfter event")
        env.subscribe(EnvEvent.stepBefore, self.handleStepBefore)
        env.subscribe(EnvEvent.stepAfter, self.handleStepAfter)

        self.stepStats = defaultdict(lambda: []) # average stuff in every step
        self.volumeStats = []
        self.pedestrianPositions = defaultdict(lambda : [])
        self.vehiclePositions = defaultdict(lambda : defaultdict(lambda : []))

    def handleStepBefore(self, env: IMultiPedestrianEnv):
        # store initial state values
        for agent in env.getPedAgents():
            #reset
            self.previousState[agent]["position"] = agent.position
            self.previousState[agent]["direction"] = agent.direction
        if hasattr(env, "getVehicleAgents"):
            for agent in env.getVehicleAgents():
                #reset
                self.previousState[agent]["topLeft"] = agent.topLeft
                self.previousState[agent]["bottomRight"] = agent.bottomRight

        # if env.step_count == 0:
        env.unsubscribe(EnvEvent.stepBefore, self.handleStepBefore)

    def handleStepAfter(self, env: IMultiPedestrianEnv):
        if env.step_count < self.stepsToIgnoreAtTheBeginning or env.step_count > self.stepsToRecord + self.stepsToIgnoreAtTheBeginning:
            logging.debug(f"MetricCollector: ignoring step {env.step_count}")
            return None

        # collect you metrics here
        logging.debug("MetricCollector: Collecting metrics")

        # collect speed
        self.collectSpeed(env)
        # collect volume
        self.collectVolume(env)
        # collect pedestrian positions
        self.collectPedestrianPositions(env)
        #collect vehicle positions
        if hasattr(env, "getVehicleAgents"):
            self.collectVehiclePositions(env)

        for agent in env.getPedAgents():
            #reset
            self.previousState[agent]["position"] = agent.position
            self.previousState[agent]["direction"] = agent.direction
        if hasattr(env, "getVehicleAgents"):
            for agent in env.getVehicleAgents():
                #reset
                self.previousState[agent]["topLeft"] = agent.topLeft
                self.previousState[agent]["bottomRight"] = agent.bottomRight
    
    def getStatistics(self):
        return [self.stepStats, self.volumeStats]
        pass

    def getPositions(self):
        return [self.pedestrianPositions, self.vehiclePositions]

    def collectVolume(self, env):
        revolutions = 0
        for agent in env.getPedAgents():
            if self.previousState[agent]["direction"] is None:
                self.previousState[agent]["direction"] = agent.direction
            # elif list(agent.position) != list(self.previousState[agent]["position"]) and agent.direction != self.previousState[agent]["direction"]:
            #     revolutions += 1
            # elif agent.direction == 0 and self.previousState[agent]["position"][0] > agent.position[0]:
            #     revolutions += 1
            # elif agent.direction == 2 and agent.position[0] > self.previousState[agent]["position"][0]:
            #     revolutions += 1
            elif agent.direction != self.previousState[agent]["direction"]:
                revolutions += 1
            self.previousState[agent]["direction"] = agent.direction
        
        
        self.volumeStats.append(revolutions*self.stepStats["xSpeed"][-1]/(env.height - 2))
        # self.volumeStats.append(revolutions/(env.height - 2))
    
    def collectSpeed(self, env):
        totalXSpeed = 0
        totalYSpeed = 0
        for agent in env.getPedAgents():
            if self.previousState[agent]["position"] is None:
                self.previousState[agent]["position"] = agent.position
            else:
                # now we have a previous position and a new position, 
                xSpeed = abs(agent.position[0] - self.previousState[agent]["position"][0])
                ySpeed = abs(agent.position[1] - self.previousState[agent]["position"][1])

                if agent.direction == 0 and self.previousState[agent]["position"][0] > agent.position[0]:
                    xSpeed = abs((env.width - 2) - self.previousState[agent]["position"][0])
                elif agent.direction == 2 and agent.position[0] > self.previousState[agent]["position"][0]:
                    xSpeed = abs(self.previousState[agent]["position"][0] - 1)

                # xSpeed = agent.speed
                # ^ try this if data still doesn't match up

                totalXSpeed += xSpeed
                totalYSpeed += ySpeed

                self.previousState[agent]["xSpeed"] = xSpeed
                self.previousState[agent]["ySpeed"] = ySpeed
        
        self.stepStats["xSpeed"].append(totalXSpeed / len(env.getPedAgents()))
        self.stepStats["ySpeed"].append(totalYSpeed / len(env.getPedAgents()))

    def collectPedestrianPositions(self, env):
        if env.step_count == 1:
            for pedestrian in env.getPedAgents():
                self.pedestrianPositions[pedestrian].append(self.previousState[pedestrian]["position"])
        
        for pedestrian in env.getPedAgents():
            self.pedestrianPositions[pedestrian].append(pedestrian.position)

    def collectVehiclePositions(self, env):
        if env.step_count == 1:
            for vehicle in env.getVehicleAgents():
                self.vehiclePositions[vehicle]["topLeft"].append(self.previousState[vehicle]["topLeft"])
                self.vehiclePositions[vehicle]["bottomRight"].append(self.previousState[vehicle]["bottomRight"])
        
        for vehicle in env.getVehicleAgents():
                self.vehiclePositions[vehicle]["topLeft"].append(vehicle.topLeft)
                self.vehiclePositions[vehicle]["bottomRight"].append(vehicle.bottomRight)