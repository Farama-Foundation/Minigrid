from typing import List
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import Agent, PedActions
from gym_minigrid.envs.pedestrian.PedGrid import PedGrid
import numpy as np

class MultiPedestrianEnv(MiniGridEnv):
    def __init__(
        self,
        agents: List[Agent]=None,
        width=8,
        height=8,
    ):
        if agents is None:
            self.agents = []
        else:
            self.agents = agents
        
        # Terry - added an agents grid to easily check for overlap of agents
        # and to be able to calculate the actions each agent should take
        # some agents will need to share lanes, so this will allow us to consider that
        self.agentsGrid = np.zeros((width, height))
        for agent in self.agents:
            self.agentsGrid[agent.position[0]][agent.position[1]] = 1

        super().__init__(
            width=width,
            height=height,
            max_steps=4*width*height,
            # Set this to True for maximum speed
            see_through_walls=True
        )
    pass

    #region agent management
    def addAgents(self, agents: List[Agent]):
        for agent in agents:
            if self.canPlaceAgent(agent):
                self.agents.append(agent)
            else:
                print("Can't place agent with position " + str(agent.position))

    def addAgent(self, agent: Agent):
        if self.canPlaceAgent(agent):
            self.agents.append(agent)
        else:
            print("Can't place agent with position " + str(agent.position))

    # Terry - We need to use this in the addAgent methods to make sure the given
    # positions of the new agents won't overlap with existing ones
    def canPlaceAgent(self, agent: Agent):
        if self.agentsGrid[agent.position[0]][agent.position[1]] == 1:
            return False
            #Terry - ^ if an agent already exists there, we can't place a new agent there
        else:
            return True
        
    def getNumAgents(self):
        return len(self.agents)

    def resetAgents(self):
        for agent in self.agents:
            agent.reset()
    
    def getDensity(self):
        cells = (self.width - 1) * (self.height - 1)
        agents = len(self.agents)
        return agents/cells

    def getDensity(self):
        cells = (self.width - 1) * (self.height - 1)
        agents = len(self.agents)
        return agents/cells

    def forwardAgent(self, agent: Agent):
        # TODO DONE
        
        # Get the position in front of the agent
        assert agent.direction >= 0 and agent.direction < 4
        fwd_pos = agent.position + agent.speed * DIR_TO_VEC[agent.direction]
        # Terry - implemented speed ^ by multiplying speed with direction unit vector

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward if no overlap
        if fwd_cell == None or fwd_cell.can_overlap():
                agent.position = fwd_pos
        # Terry - Once we get validateAgentPositions working, we won't need to check
        pass

    # Terry - move left and right functions are below
    def shiftLeft(self, agent: Agent):
        assert agent.direction >= 0 and agent.direction < 4
        #Terry - uses the direction to left of agent to find vector to move left
        left_dir = agent.direction - 1
        if left_dir < 0:
            left_dir += 4
        left_pos = agent.position + DIR_TO_VEC[left_dir]

        agent.position = left_pos

    def shiftRight(self, agent: Agent):
        assert agent.direction >= 0 and agent.direction < 4
        #Terry - uses the direction to left of agent to find vector to move left
        right_dir = (agent.direction + 1) % 4
        right_pos = agent.position + DIR_TO_VEC[right_dir]
        
        agent.position = right_pos
    #endregion

    #region sidewalk

    def genSidewalks(self):

        # TODO turn this into 2 side walks. DONE

        # Terry - added goals to the left side
        # not sure if we are making the sidewalks go horizontally or vertically
        for i in range(1, self.height-1):
            self.put_obj(Goal(), 1, i)
            self.put_obj(Goal(), self.width - 2, i)
        pass

    #endregion

    #region gym env overrides

    def validateAgentPositions(self):
        # TODO iterate over our agents and make sure that they can be placed there

        # Terry - Is this to validate after getting a list of actions for parallel update?
        # If so, I have started part of that in the step function.

        # Will change whether they can move left or right
        # Agent A has tile to the right, agent B has same tile to left
        # One will have moveRight = true, other will have moveLeft = false

        # Check that the agent doesn't overlap with an object
        for agent in self.agents:
            start_cell = self.grid.get(*agent.position)
            assert start_cell is None or start_cell.can_overlap()
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

    def parallel1():
        #TODO Simulate lane change
        pass

    def parallel2():
        #TODO What lane allows agent to move using max speed
        pass

    # One step after parallel1 and parallel2
    # Save plans from parallel1 and parallel2 before actually executing it

    def step(self, action=None):
        self.step_count += 1

        reward = 0
        done = False

        # TODO:
        # 1. get the agent
        # 2. decide what type of action (move forward, change lane (moveLeft or moveRight), do nothing)
        # 3. call the method that takes the action

        actions = []
        for agent in self.agents:
            action = agent.getAction()
            actions.append(action)

        newAgentsGrid = np.zeros((self.width, self.height))
        # Terry - create the new agentsGrid here before actually testing the actions
        # We simulate the new positions here to check for an overlap of agents
        # Change value at agent positions to 1 as we iterate over all the agents
        # Check if value equals 1 before setting the value to check if there
        # is already an agent there
        # If yes, then we can't have the agent take that action because there would
        # be 2 agents in the same position
        # After simulating the new agents grid and there isn't any problems
        # with all the actions, then set self.agentsGrid = newAgentsGrid
        # to update the existing grid and proceed to taking the actions below
        
        index = 0
        for agent in self.agents:

            if actions[index] == PedActions.forward:
                self.forwardAgent(agent)
            elif actions[index] == PedActions.shiftLeft:
                self.shiftLeft(agent)
            elif actions[index] == PedActions.shiftRight:
                self.shiftRight(agent)
            else:
                assert False, f"unknown action {action}"

            index += 1

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

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

class MultiPedestrianEnv20x80(MultiPedestrianEnv):
    def __init__(self):
        width = 10
        height = 20
        super().__init__(
            width=width,
            height=height,
            agents=None
        )

register(
    id='MultiPedestrian-Empty-20x80-v0',
    entry_point='gym_minigrid.envs.pedestrian.MultiPedestrianEnv:MultiPedestrianEnv20x80'
)
