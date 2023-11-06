
import random, math, os
import numpy as np

import gym
from gym import spaces
# from gym_minigrid.register import register

# rough hack
import sys
# sys.path.insert(0, '../')
from resolver import progress, is_accomplished
from . import ltl_sampler as ls
from . import ltl2tree as lt

from random import randint
from pathlib import Path

class LTLBootcamp(gym.Env):
    """
    An environment to pre-train the LTL embedding.
    """

    def __init__(
        self,
        fixed_task=None,    # set an LTL instruction to be kept at every env reset
        timeout=25,         # max steps that the agent can do
        samplenum = 50,     # number of ltl tasks should be sampled
        alphabet=['a', 'b', 'c', 'd', 'e', 'f'],  #alphabet for ltl sampler
        rel = [" ", "s", "p", "l", "r", "p_", "l_", "r_"], #relationships for LTL tasks
        regen = True,      # control whether to load task from txt
    ):

        self.timeout = timeout
        self.time = 0
        self.fixed_task = fixed_task
        self.task = None
        self.alphabet = alphabet
        self.rel = rel

        # Sample LTL tasks and convert to format like ['A', ['G', ['N', 'b']], ['E', 'r']]
        # Begin to generate LTL tasks...
        task_file = Path("tasks.txt")
        normal_task_file = Path("normal_task.txt")
        if not regen and task_file.is_file():
            #TODO: reload from txt
            pass
        else:
            ltls = ls.ltl_sampler(self.alphabet, n_samples=samplenum)
            tasks = set()
            self.tasks = []
            self.normal_tasks = []
            for ltl in ltls:
                for formula in ltl:
                    if formula is not None:
                        ltl_tree = lt.ltl2tree(formula, self.alphabet)
                        ltl_str = lt.ltl_tree_str(ltl_tree)
                        if ltl_str not in tasks:
                            ltl_list = lt.unroll_tree(ltl_tree)
                            tasks.add(ltl_str)
                            self.tasks.append(ltl_list)
                            self.normal_tasks.append(formula)

            self.tasks.sort(key=lambda x: (len(x)))
            # record to txt file
            with open(task_file, 'w') as f,\
                 open(normal_task_file, 'w') as n_f:
                for task in tasks:
                    f.write(task)
                    f.write('\n')
                for n_task in self.normal_tasks:
                    n_f.write(n_task)
                    n_f.write('\n')
            print("Write tasks completed")
            f.close(), n_f.close()

        # Actions are discrete integer values
        # self.action_space = spaces.Discrete(len(' rgb'))
        self.action_space = spaces.Discrete(len(self.alphabet)+1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(110,), # Mission needs to be padded
            dtype='uint8'
        )

        # Initialize the state
        self.reset()

    def length(self, formula):
        ''' Aux method to recursively determine the length of formula. '''
        if type(formula) == str:
            return 1
        elif len(formula) == 2:
            return self.length(formula[1]) + 1
        elif len(formula) == 3:
            return self.length(formula[1]) + self.length(formula[2]) + 1
        else:
            return 10+1 #max_len + 1 to give up this task

    def draw_task(self, max_len=10):
        ''' Helper function to randomly draw a new LTL task from the task distribution. '''
        if self.fixed_task is not None:
            return self.fixed_task
        index = randint(0, len(self.tasks) - 1)
        while self.length(self.tasks[index]) >= max_len:
            index = randint(0, len(self.tasks) - 1)
        return self.tasks[index]



    def reset(self):
        ''' Env reset, must be called every time 'done' becomes True. '''
        
        self.task = self.draw_task()
        self.mission = str(self.task)
        print('task ended, pick a new task:{}'.format(self.mission))
        return self.gen_obs()


    def reward(self):
        '''
            Helper function to establish the reward and the done signals.
            Returns the (reward, done) tuple.
        '''

        if self.task == "True" or is_accomplished(self.task):   return (1, True)
        elif self.task == "False":  return (-1, True)
        return (0, False)
    
    

    def build_relations(self, formula, max_len=10):
        ''' Function to construct the skew-symmetric relation matrix. '''
        # define a max_len x max_len matrix with as many 1s on the diagonal as the length of formula
        rel_len = self.length(formula)   # length of formula
        pad_len = max_len - rel_len # length of padding
        mat = np.diag(np.concatenate([np.ones(rel_len), np.zeros(pad_len)]))

        # define a proper relation vocabulary
        V = {k: v for v, k in enumerate([" ", "s", "p", "l", "r", "p_", "l_", "r_"])}

        def tagger(formula, mat, prev_idx=-1, idx=0, rel=None):
            ''' Aux method to recursively fill the relation matrix based on formula. '''
            def aux(mat, i, j, rel):
                if prev_idx != -1 and rel is not None:
                    mat[prev_idx, idx] = V[rel]
                    mat[idx, prev_idx] = V[rel + '_']
            if type(formula) == str:
                aux(mat, prev_idx, idx, rel)
                return idx
            if len(formula) == 2:
                aux(mat, prev_idx, idx, rel)
                return tagger(formula[1], mat, idx, idx+1, 'p')
            if len(formula) == 3:
                aux(mat, prev_idx, idx, rel)
                offset = tagger(formula[1], mat, idx, idx+1, 'l')
                return tagger(formula[2], mat, idx, offset+1, 'r')

        tagger(formula, mat)

        # return the properly filled relation matrix of formula
        return mat


    
    def gen_obs(self):

        def encode_mission(mission):
            assert mission == str(self.task), "Task and mission are not equivalent!"
            syms = "AONGUXE"+"".join(self.alphabet)
            V = {k: v+1 for v, k in enumerate(syms)}
            enc = np.array([V[e] for e in mission if e not in "\', []"])
            # TODO: parameterize
            rels = self.build_relations(self.task, 10)
            return np.concatenate([enc, np.zeros(10 - len(enc))]), np.array(rels)

        obs = np.zeros(110) # max mission length
        if self.mission == 'True' or self.mission == 'False':
            return obs
        mission, rels = encode_mission(self.mission)
        return np.concatenate((mission, rels.reshape(-1))) # 10 + 10x10 = 110


    def step(self, action):

        # event detector
        if action == 0:
            event_objs = []
            print("No action.")
        else:
            event_objs = self.alphabet[action - 1]
            print("Execute action: {}".format(event_objs))

        # prog function call
        self.task = progress(self.task, event_objs)
        self.mission = str(self.task)
        print("task after execution: {}".format(self.mission))

        #TODO: parameterize
        if self.length(self.task) >= 10:
            print("give up task, too long!")
            reward, done = -0.5, True
        else:
            reward, done = self.reward()

        # max steps elapsed
        self.time += 1
        if self.time > self.timeout:
            reward, done = -1, True
            self.time = 0

        return self.gen_obs(), reward, done, {}

