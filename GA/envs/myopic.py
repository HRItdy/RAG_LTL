from hashlib import new
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

# rough hack
import sys
sys.path.insert(0, './envs')
from minigrid_extensions import *
# sys.path.insert(0, '../')
from resolver import progress

from random import randint


class AdversarialMyopicEnv(MiniGridEnv):
    """
    An environment where a myopic agent will fail. The two possible goals are "Reach blue then green" or "Reach blue then red".
    """

    def __init__(
        self,
        size=8,                 # size of the grid world
        agent_start_pos=(1,1),  # starting agent position
        agent_start_dir=0,      # starting agent orientation
        timeout=100             # max steps that the agent can do
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.event_objs = []

        self.timeout = timeout
        self.time = 0
        self.complete_task = None

        self.super_init(
            grid_size=size,
            max_steps=4*size*size,
            see_through_walls=True # set this to True for maximum speed
        )

    def super_init(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7
    ):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(180,), # Mission needs to be padded
            dtype='uint8'
        )
        # self.observation_space = spaces.Dict({
        #     'image': self.observation_space
        # })

        # Range of possible rewards
        self.reward_range = (-1, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    
    def reset(self):
        obs = super().reset()
        self.time = 0
        return obs


    def draw_task(self):
        ''' Helper function to randomly draw a new LTL task from the task distribution. '''

        tasks = [
            [['E', 'b'], ['E', 'r']],   # blue first, then red
            [['E', 'b'], ['E', 'g']]    # blue first, then green
        ]
        return tasks[randint(0, len(tasks) - 1)]



    def _gen_grid(self, width, height):
        ''' Helper function to generate a new random world. Called at every env reset. '''

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate inner walls
        self.grid.vert_wall(4, 0)
        self.grid.horz_wall(4, 4)

        self.door_1 = Door(COLOR_NAMES[0], is_open=True)
        self.door_2 = Door(COLOR_NAMES[0], is_open=True)

        self.door_1_loc = (4,2)
        self.door_2_loc = (4,6)
        self.grid.set(*self.door_1_loc, self.door_1)
        self.grid.set(*self.door_2_loc, self.door_2)

        # Place a goal square in the bottom-right corner
        self.blue_goal_1_pos = (5, 7)
        self.blue_goal_2_pos = (5, 1)
        self.blue_goal_1 = CGoal('blue')
        self.blue_goal_2 = CGoal('blue')

        self.green_goal_pos = (7, 7)
        self.red_goal_pos = (7, 1)
        self.green_goal = CGoal('green')
        self.red_goal = CGoal('red')

        # Randomize which room contains the green and red goals
        if randint(0,1) == 0:
            self.green_goal_pos, self.red_goal_pos = self.red_goal_pos, self.green_goal_pos

        self.put_obj(self.green_goal, *self.green_goal_pos)
        self.put_obj(self.red_goal, *self.red_goal_pos)
        self.put_obj(self.blue_goal_1, *self.blue_goal_1_pos)
        self.put_obj(self.blue_goal_2, *self.blue_goal_2_pos)

        self.event_objs = []
        self.event_objs.append((self.blue_goal_1_pos, 'b'))
        self.event_objs.append((self.blue_goal_2_pos, 'b'))
        self.event_objs.append((self.green_goal_pos, 'g'))
        self.event_objs.append((self.red_goal_pos, 'r'))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent(top=(1,1), size=(3,7))

        # Task
        self.complete_task = self.draw_task()
        self.mission = str(self.complete_task[0])


    def reward(self):
        '''
            Helper function to establish the reward and the done signals.
            Returns the (reward, done) tuple.
        '''

        if self.mission == "True":      return (1, True)
        elif self.mission == "False":   return (-1, True)
        else:                           return (0, False)


    def encode_mission(self, mission):
        syms = "AONGUXE[]rgb"
        V = {k: v+1 for v, k in enumerate(syms)}
        return [V[e] for e in mission if e not in ["\'", ",", " "]]        

    
    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        obs = super().gen_obs()
        
        # obs = {
        #     'image': image, (7,7,3)
        #     'direction': self.agent_dir,
        #     'mission': self.mission
        # }
        
        img = np.array(obs["image"]).reshape(-1) #147
        direction = np.array([obs["direction"]]) # 1
        mission = np.array(self.encode_mission(obs["mission"]))

        new_obs = np.concatenate((img, direction, mission))

        obs = np.zeros(180)
        obs[:new_obs.shape[0]] = new_obs
        
        return obs


    def step(self, action):

        # Lock the door automatically behind you
        if action == self.actions.forward and self.agent_dir == 0:
            if tuple(self.agent_pos) == self.door_1_loc:
                self.door_1.is_open = False
                self.door_1.is_locked = True
            elif tuple(self.agent_pos) == self.door_2_loc:
                self.door_2.is_open = False
                self.door_2.is_locked = True

        obs, _, _, _ = super().step(action)

        # prog function call
        if progress(self.complete_task[0], self.get_events()) == "True":
            self.complete_task.pop(0)
        self.mission = str(self.complete_task[0]) if self.complete_task else "True"

        reward, done = self.reward()

        # max steps elapsed
        self.time += 1
        if self.time >= self.timeout:
            reward, done = -1, True

        return obs, reward, done, {}


    def get_events(self):
        ''' Event detector. '''

        events = []
        for obj in self.event_objs:
            if tuple(self.agent_pos) == obj[0]:
                events.append(obj[1])
        return events


class AdversarialMyopicEnv9x9(AdversarialMyopicEnv):
    def __init__(self, agent_start_pos=None):
        super().__init__(size=9, agent_start_pos=agent_start_pos)

