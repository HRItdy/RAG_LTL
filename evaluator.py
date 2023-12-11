import torch as th

from GA.custom_policy import CustomActorCriticPolicy
from GA.envs.ltl_bootcamp import LTLBootcamp

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from ltl_utils import LTL
from GA.envs.resolver import progress
from GA.envs import ltl2tree as lt
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
#response = get_response(GEN_PROMPT)
response = 'F(a & F(! b))'
ltl = LTL(response)
response = ltl.re_replace(response)
dfa = ltl.to_networkx()
## Get all the guards
guards = ltl.get_guards(dfa)
## For each guard, generate the corresponding event combination
truth_assignment = {}
for guard in guards.values():
    truth_assignment[guard] = []
    truth_assignment[guard].extend(ltl.get_events(guard))
## Initialize multiple LLMs to evaluate
records = {} # {ltl_task:{guard:{events: {progressed_ltl:   progressed_spec:  evaluate: true or false}}}}
for guard, truth in truth_assignment.items():
    for assignment in truth:
        events = [atomic for atomic, value in assignment.items() if value is True]
        events.sort()
        event_str = "".join(events) if isinstance(events,list) else ""
        #progress the ltl task and store them into the records
        ltl_tree = lt.ltl2tree(response, ltl.get_alphabet(response))
        ltl_str = lt.ltl_tree_str(ltl_tree)
        ltl_list = lt.unroll_tree(ltl_tree)
        processed_task = progress(ltl_list, events)
        processed_ltl_str = lt.ltl_tree_str(processed_task)
    ## Progress the natural language task specification
    prompt = process_prompt(task_spec, truth)
    processed_spec = get_response(prompt)
    ## Progress the linear temporal logic task
    events = []
    for ap, value in truth.items():
        if value: 
            events.append(ap)
    # Evaluate the generated task 
    ltl = LTL(processed_task)
    dot = ltl.get_dot()
    judge_prompt = generate_evaluation(processed_spec, dot)
    result = get_response(judge_prompt)
    records[processed_task][guard] = result
times = 0

# how to randomly select tasks, exam whether they are included in records, if not, progress them and exam.
