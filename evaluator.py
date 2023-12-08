import torch as th

from GA.custom_policy import CustomActorCriticPolicy
from GA.envs.ltl_bootcamp import LTLBootcamp

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from ltl_utils import LTL
from GA.ltl_progression import *
import numpy as np
import random
from utils import *
import networkx as nx
import openai

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
openai.api_key = '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'

import json
with open('./Retrieve/retrieve_msg.json') as retrieve_data:
    retrieve_dict = json.load(retrieve_data)

task_spec = random.choice(list(retrieve_dict.keys()))
GEN_PROMPT = generate_prompt(task_spec)
## Get the result
response = get_response(GEN_PROMPT)
#response = 'F(a & F(! b))'
ltl = LTL(response)
dfa = ltl.to_networkx()
## Get all the guards
guards = ltl.get_guards(dfa)
## For each guard, generate the corresponding event combination
truth_assignment = {}
for guard in guards.values():
    truth_assignment[guard] = []
    truth_assignment[guard].extend(ltl.get_events(guard))
## Initialize multiple LLMs to evaluate the 
records = {}
for guard, truth in truth_assignment.items():
    ## Progress the natural language task specification
    prompt = process_prompt(task_spec, truth)
    ## Progress the linear temporal logic task
    events = []
    for ap, value in truth.items():
        if value: 
            events.append(ap)
    processed_task = progress(response, events)
# Evaluate the generated task 
times = 0
