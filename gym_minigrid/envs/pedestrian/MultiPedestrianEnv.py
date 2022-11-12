from typing import List
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import Agent, PedActions, PedAgent
from gym_minigrid.envs.pedestrian.PedGrid import PedGrid

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

        super().__init__(
            width=width,
            height=height,
            max_steps=1000, #4*width*height,
            # Set this to True for maximum speed
            see_through_walls=True
        )
    pass

    #region agent management
    def addAgents(self, agents: List[Agent]):
        self.agents.extend(agents)
            
    def addAgent(self, agent: Agent):
        self.agents.append(agent)
        
    def getNumAgents(self):
        return len(self.agents)

    def resetAgents(self):
        for agent in self.agents:
            agent.reset()

    def forwardAgent(self, agent: Agent):
        # TODO DONE
        
        # Get the position in front of the agent
        assert agent.direction >= 0 and agent.direction < 4
        fwd_pos = agent.position + agent.speed * DIR_TO_VEC[agent.direction]
        if fwd_pos[1] < 0 or fwd_pos[1] >= self.height:
            return
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
        # left_dir = agent.direction - 1
        # if left_dir < 0:
        #     left_dir += 4
        # left_pos = agent.position + DIR_TO_VEC[left_dir]

        # agent.position[0] = left_pos
        agent.position = (agent.position[0] - 1, agent.position[1])

    def shiftRight(self, agent: Agent):
        # assert agent.direction >= 0 and agent.direction < 4
        # #Terry - uses the direction to left of agent to find vector to move left
        # right_dir = (agent.direction + 1) % 4
        # right_pos = agent.position + DIR_TO_VEC[right_dir]
        
        # agent.position = right_pos
        agent.position = (agent.position[0] + 1, agent.position[1])
        
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
            if agent.position[0] == 1:
                agent.canShiftLeft = False
            if agent.position[0] == self.width - 2:
                agent.canShiftRight = False
        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent1 == agent2 or agent1.position[1] != agent2.position[1]:
                    continue

                if (agent1.position[0] - agent2.position[0]) == 1:
                    # they are adjacent
                    agent1.canShiftLeft = False
                    agent2.canShiftRight = False
                elif (agent1.position[0] - agent2.position[0]) == 2 and agent1.canShiftLeft == True and agent2.canShiftRight == True:  
                    # they have one cell between them
                    if np.random.random() > 0.5:
                        agent1.canShiftLeft = False 
                    else: 
                        agent2.canShiftRight = False

        for agent in self.agents:
            agent.position

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
        self.eliminateConflict()
        LaneActions = []
        for agent in self.agents:
            lane = agent.parallel1(self.agents)
            print(lane, agent.speed, agent.gapSame, agent.gapOpp)
            LaneActions.append(lane)
        
        for i, laneAction in enumerate(LaneActions):
            if laneAction == 1:
                self.shiftLeft(self.agents[i])
            elif laneAction == 2:
                self.shiftRight(self.agents[i])
                
        print('done')

        for agent in self.agents:
            self.forwardAgent(agent)
            agent.canShiftLeft = True
            agent.canShiftRight = True
        # actions = []
        # for agent in self.agents:
        #     action = agent.getAction()
        #     actions.append(action)

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
        
        
        # index = 0
        # for agent in self.agents:

        #     if actions[index] == PedActions.forward:
        #         self.forwardAgent(agent)
        #     elif actions[index] == PedActions.shiftLeft:
        #         self.shiftLeft(agent)
        #     elif actions[index] == PedActions.shiftRight:
        #         self.shiftRight(agent)
        #     else:
        #         assert False, f"unknown action {action}"

        #     index += 1

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
        width = 20
        height = 60
        super().__init__(
            width=width,
            height=height,
            agents=None
        )

register(
    id='MultiPedestrian-Empty-20x80-v0',
    entry_point='gym_minigrid.envs.pedestrian.MultiPedestrianEnv:MultiPedestrianEnv20x80'
)
