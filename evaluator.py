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
openai.api_key = 'TO-BE-SET'

import json
with open('./Retrieve/retrieve_msg.json') as retrieve_data:
    retrieve_dict = json.load(retrieve_data)

task_spec = random.choice(list(retrieve_dict.keys()))
GEN_PROMPT = generate_prompt(task_spec)
## Get the result
#response = get_response(GEN_PROMPT)
response = 'F(a & F(! b))'
response = regular(response)
# get the eval result of the root task
ltl = LTL(response)
response = ltl.re_replace(response)
dfa = ltl.to_networkx()
dot, num = ltl.get_dot_and_number()
## Get all the guards
guards = ltl.get_guards(dfa)
eval_pro = evaluate_prompt(task_spec, dot)
eval = get_response(eval_pro)
eval = int(eval)
## For each guard, generate the corresponding event combination
records = {response:{'spec':[task_spec], 'dot':dot, 'eval': eval, 'num': num}} # {ltl_task:{specification:[], DOT:..., evaluate:0-100}}}}
truth_assignment = {}
for guard in guards.values():
    truth_assignment[guard] = []
    truth_assignment[guard].extend(ltl.get_events(guard))
walks = ltl.random_walk(walk_num=15, walk_length=10)

def get_progress(exam_task, guard, truth_assignment):
    truths = truth_assignment[guard]
    for truth in truths:
        events = [atomic for atomic, value in truth.items() if value is True]
        events.sort()
        event_str = ",".join(events) if isinstance(events,list) else "no action"
        #progress the ltl task and store them into the records
        ltl_tree = lt.ltl2tree(exam_task, ltl.get_alphabet(exam_task))
        # ltl_str = lt.ltl_tree_str(ltl_tree)
        ltl_list = lt.unroll_tree(ltl_tree)
        progress_task = progress(ltl_list, events)
        progress_str = lt.reconstruct(progress_task)
        progress_str = regular(progress_str)
    return event_str, progress_str

def get_dot_num(task):
    ltl = LTL(task)
    task = ltl.re_replace(task)
    dot, num = ltl.get_dot_and_number()
    return dot, num

def get_eval_score(records):
    max_num = 0
    eval_score = 0
    counter = 0
    for _, info in records:
        max_num = max(max_num, info['num'])
        counter += 1
    for _, info in records:
        eval_score += (max_num - info['num'])/max_num * info['eval']
    eval_score /= counter
    
for walk in walks:
    exam_task = regular(response)
    tem_spec = task_spec
    for guard in walk:
        # progress ltl task and natural language task specification
        event_str, progress_str = get_progress(exam_task, guard, truth_assignment)
        proc = process_prompt(tem_spec, event_str)
        tem_spec = get_response(proc)
        # tem_spec = tem_spec + 'a'
        exam_task = regular(progress_str)
        if exam_task == 'True':
            break
        # if the progressed task has been in records, add the new natural language task specification to augment
        if exam_task in records.keys():
            records[exam_task]['spec'].append(tem_spec)
            continue 
        # if the progressed task not in records, add them
        else: 
            dot, num = get_dot_num(exam_task)
            eval_pro = evaluate_prompt(tem_spec, dot)
            eval = get_response(eval_pro)
            eval = int(eval)
            records[exam_task] = {'spec': [tem_spec], 'dot': dot, 'eval': eval, 'num': num}

eval_score = get_eval_score(records)