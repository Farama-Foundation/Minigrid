from typing import List
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import Agent, PedActions
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
            max_steps=4*width*height,
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
        # TODO 
        pass
    #endregion

    #region sidewalk

    def genSidewalks(self):
        
        # TODO turn this into 2 side walks.
        # Place a goal square in the bottom-right corner
        for i in range(1, self.height-1):
            self.put_obj(Goal(), self.width - 2, i)
        
        for i in range(1, self.height-1):
            self.put_obj(Goal(), 1, i)
        



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


    def step(self, action=None):
        self.step_count += 1

        reward = 0
        done = False

        for agent in self.agents:

            # TODO:
            # 1. get the agent

            # 2. decide what type of action (move forward, change lane (moveLeft or moveRight), do nothing)
            action = agent.getAction()
            # 3. call the method that takes the action
            if action == PedActions.forward:
                self.forwardAgent(agent)
            # Rotate left # replace with rotation
            elif action == PedActions.moveLeft:
                agent.direction -= 1
                if agent.direction < 0:
                    agent.direction += 4
            # Rotate right
            elif action == PedActions.moveRight:
                agent.direction = (agent.direction + 1) % 4
            else:
                assert False, f"unknown action {action}"
        


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
