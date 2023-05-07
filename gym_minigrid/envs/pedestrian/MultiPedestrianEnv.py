from typing import List
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import Agent, PedActions, PedAgent
from gym_minigrid.envs.pedestrian.PedGrid import PedGrid
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.ForwardAction import ForwardAction
from gym_minigrid.lib.Direction import Direction
from .EnvEvent import EnvEvent
import logging
import random
# from pyee.base import EventEmitter

class MultiPedestrianEnv(MiniGridEnv):
    def __init__(
        self,
        agents: List[Agent]=None,
        width=8,
        height=8,
        stepsIgnore = 100
    ):

        if agents is None:
            self.agents = []
        else:
            self.agents = agents
        self.stepsIgnore = stepsIgnore
        super().__init__(
            width=width,
            height=height,
            max_steps=100000, #4*width*height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

        self.stepsTaken = 0

        self.stepBefore = []
        self.stepParallel1 = []
        self.stepParallel2 = []
        self.stepParallel3 = []
        self.stepParallel4 = []
        self.stepAfter = []

        self._actionHandlers = {
            LaneAction: self.executeLaneAction,
            ForwardAction: self.executeForwardAction,
        }

    pass

    #region agent management

    def getAgents(self):
        return self.agents

    def addAgents(self, agents: List[Agent]):
        for agent in agents:
            self.addAgent(agent)
            
    def addAgent(self, agent: Agent):
        self.agents.append(agent)

        ## attach event handlers
        # TODO: this subscription must not be done in side the evironment as only the research knows about its decision and action phases.
        self.subscribe(EnvEvent.stepParallel1, agent.parallel1) # TODO event name should be enums
        self.subscribe(EnvEvent.stepParallel2, agent.parallel2) # TODO event name should be enums
        
    def getNumAgents(self):
        return len(self.agents)

    def resetAgents(self):
        for agent in self.agents:
            agent.reset()

    # moved to MetricCollector and can be removed
    # def getDensity(self):
    #     cells = (self.width - 1) * (self.height - 1)
    #     agents = len(self.agents)
    #     return agents/cells

    # def getAverageSpeed(self):

    #     return self.stepsTaken / len(self.agents) / (self.step_count - self.stepsIgnore)

    def removeAgent(self, agent):
        if agent in self.agents:
            self.unsubscribe(EnvEvent.stepParallel1, agent.parallel1) # TODO event name should be enums
            self.unsubscribe(EnvEvent.stepParallel2, agent.parallel2) # TODO event name should be enums
            self.agents.remove(agent)
        else:
            logging.warn("Agent not in list")

    def forwardPedestrian(self, agent: Agent):
        # TODO DONE
        if self.step_count >= self.stepsIgnore:
            self.stepsTaken += agent.speed
        # Get the position in front of the agent
        assert agent.direction >= 0 and agent.direction < 4
        fwd_pos = agent.position + agent.speed * DIR_TO_VEC[agent.direction]
        if agent.position[1] < 0 or agent.position[1] >= self.height:
            logging.info(f"id: {agent.id} pos: {agent.position} canRight: {agent.canShiftRight} canleft: {agent.canShiftLeft}")
        # print("Id ", agent.id, "speed ", agent.speed)
        # if fwd_pos[0] <= 0 o]r fwd_pos[0] >= self.width - 1: # = sign is to include gray squares on left & right
        #     if fwd_pos[0] <= 0:
        #         agent.position = (1, agent.position[1])
        #     else:
        #         agent.position = (self.width - 2, agent.position[1])
        #     agent.direction = (agent.direction + 2) % 4
        #     return
        if fwd_pos[0] < 1:
            # random may introduce conflict
            # agent.position = (self.width - 2, random.randint(1, self.height - 2))
            # agent.position = (self.width - 2, agent.position[1])
            agent.position = (1, agent.position[1])
            agent.direction = Direction.LR
            return
        elif fwd_pos[0] > self.width - 2:
            # agent.position = (1, random.randint(1, self.height - 2))
            # agent.position = (1, agent.position[1])
            agent.position = (self.width - 2, agent.position[1])
            agent.direction = Direction.RL
            return

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward if no overlap
        if fwd_cell == None or fwd_cell.can_overlap():
            agent.position = (fwd_pos[0], fwd_pos[1])
        # Terry - Once we get validateAgentPositions working, we won't need to check
        pass

    # Terry - move left and right functions are below
    def shiftLeft(self, agent: Agent):
        assert agent.direction >= 0 and agent.direction < 4
        #Terry - uses the direction to left of agent to find vector to move left
        # left_dir = agent.direction - 1
        # if left_dir < 0:
        #     left_dir += 4
        # left_pos = agent.position + DIR_TO_VEC[left_dir]

        # agent.position[0] = left_pos
        agent.position = (agent.position[0], agent.position[1] - 1)

    def shiftRight(self, agent: Agent):
        # assert agent.direction >= 0 and agent.direction < 4
        # #Terry - uses the direction to left of agent to find vector to move left
        # right_dir = (agent.direction + 1) % 4
        # right_pos = agent.position + DIR_TO_VEC[right_dir]
        
        # agent.position = right_pos
        agent.position = (agent.position[0], agent.position[1] + 1)
        
    #endregion

    #region sidewalk

    def genSidewalks(self):
        for i in range(1, self.height-1):
            self.put_obj(Goal(), 1, i)
            self.put_obj(Goal(), self.width - 2, i)
        pass

    #endregion

    #region gym env overrides

    def validateAgentPositions(self):
        # TODO iterate over our agents and make sure that they can be placed there
        
        # # Check that the agent doesn't overlap with an object
        # start_cell = self.grid.get(*self.agent_pos)
        # assert start_cell is None or start_cell.can_overlap()
        pass


    def reset(self):
        
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        self.resetAgents()
        self.validateAgentPositions()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs



    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = PedGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.genSidewalks()
        self.mission = "switch sidewalks"


    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        img = self.grid.render(
            tile_size,
            self.agents,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=None
            # highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def eliminateConflict(self):
        for agent in self.agents:
            if agent.position[1] <= 1:
                agent.canShiftLeft = False
            if agent.position[1] >= self.height - 2:
                agent.canShiftRight = False
        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent1 == agent2 or agent1.position[0] != agent2.position[0]:
                    continue

                if (agent1.position[1] - agent2.position[1]) == 1:
                    # they are adjacent
                    agent1.canShiftLeft = False
                    agent2.canShiftRight = False
                elif (agent1.position[1] - agent2.position[1]) == 2 and agent1.canShiftLeft == True and agent2.canShiftRight == True:  
                    # they have one cell between them
                    if np.random.random() > 0.5:
                        agent1.canShiftLeft = False 
                    else: 
                        agent2.canShiftRight = False


    def unsubscribe(self, envEvent: EnvEvent, handler):

        if envEvent == envEvent.stepBefore: 
            self.stepBefore.remove(handler)

        if envEvent == envEvent.stepAfter: 
            self.stepAfter.remove(handler)
            
        if envEvent == envEvent.stepParallel1: 
            self.stepParallel1.remove(handler)

        if envEvent == envEvent.stepParallel2: 
            self.stepParallel2.remove(handler)
    
    def subscribe(self, envEvent, handler):

        if envEvent == envEvent.stepBefore: 
            self.stepBefore.append(handler)

        if envEvent == envEvent.stepAfter: 
            self.stepAfter.append(handler)

        if envEvent == envEvent.stepParallel1: 
            self.stepParallel1.append(handler)

        if envEvent == envEvent.stepParallel2: 
            self.stepParallel2.append(handler)
    
    def emitEventAndGetResponse(self, envEvent) -> List[Action]:

        logging.debug(f"executing {envEvent}")
        if envEvent == EnvEvent.stepBefore: 
            return [handler(self) for handler in self.stepBefore]

        if envEvent == EnvEvent.stepAfter: 
            return [handler(self) for handler in self.stepAfter]

        # logging.debug(self.stepParallel1)
        # logging.debug(self.stepParallel2)
        if envEvent == EnvEvent.stepParallel1: 
            return [handler(self) for handler in self.stepParallel1] # TODO fix for multiple actions by a single handler.

        if envEvent == EnvEvent.stepParallel2: 
            return [handler(self) for handler in self.stepParallel2]


    

    def step(self, action=None):
        """This step is tightly coupled with the research, how can we decouple it?

        Args:
            action (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.emitEventAndGetResponse(EnvEvent.stepBefore)

        self.step_count += 1

        reward = 0
        done = False

        self.eliminateConflict()

        actions = self.emitEventAndGetResponse(EnvEvent.stepParallel1)
        
        self.executeActions(actions)
        actions = self.emitEventAndGetResponse(EnvEvent.stepParallel2)
        self.executeActions(actions)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        
        self.emitEventAndGetResponse(EnvEvent.stepAfter)
        
        return obs, reward, done, {}

    def executeActions(self, actions: List[Action]):
        if len(actions) == 0:
            return
        # TODO

        for action in actions:
            if action is not None:
                self._actionHandlers[action.action.__class__](action)
            
        pass

    def executeLaneAction(self, action: Action):
        if action is None:
            return 
        if action.action == LaneAction.LEFT:
            self.shiftLeft(action.agent)
        elif action.action == LaneAction.RIGHT:
            self.shiftRight(action.agent)
        pass

    def executeForwardAction(self, action: Action):
        if action is None:
            return 


            
        agent = action.agent

        logging.debug(f"forwarding agent {agent.id}")

        self.forwardAgent(agent)
        agent.canShiftLeft = True
        agent.canShiftRight = True
        pass

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        # TODO
        # 1. There is no observation
        obs = {
        }

        return obs

    #endregion

    
# TODO extract the following to a registration file

class MultiPedestrianEnv20x80(MultiPedestrianEnv):
    def __init__(self):
        width = 80
        height = 20 # actual height: 10 + 2 gray square on top and bottom
        super().__init__(
            width=width,
            height=height,
            agents=None
        )

class MultiPedestrianEnv5x20(MultiPedestrianEnv):
    def __init__(self):
        width = 30
        height = 6 # actual height: 10 + 2 gray square on top and bottom
        super().__init__(
            width=width,
            height=height,
            agents=None,
            stepsIgnore=0
        )

class MultiPedestrianEnv1x20(MultiPedestrianEnv):
    def __init__(self):
        width = 20
        height = 3 # actual height: 10 + 2 gray square on top and bottom
        super().__init__(
            width=width,
            height=height,
            agents=None
        )

register(
    id='MultiPedestrian-Empty-20x80-v0',
    entry_point='gym_minigrid.envs.pedestrian.MultiPedestrianEnv:MultiPedestrianEnv20x80'
)
register(
    id='MultiPedestrian-Empty-5x20-v0',
    entry_point='gym_minigrid.envs.pedestrian.MultiPedestrianEnv:MultiPedestrianEnv5x20'
)
register(
    id='MultiPedestrian-Empty-1x20-v0',
    entry_point='gym_minigrid.envs.pedestrian.MultiPedestrianEnv:MultiPedestrianEnv1x20'
)
