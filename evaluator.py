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
import wandb

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
openai.api_key = 'sk-TO_BE_SET'

import json
with open('./Retrieve/retrieve_msg.json') as retrieve_data:
    retrieve_dict = json.load(retrieve_data)
wandb.init()
prediction_table = wandb.Table(columns=["task", "natural_task", "dot_format", "eval_score", "state_number"])
task_spec = random.choice(list(retrieve_dict.keys()))
GEN_PROMPT = generate_prompt(task_spec)
## Get the result
response = get_response(GEN_PROMPT)
# response = 'F(a & F(! b))'
response = regular(response)
# get the eval result of the root task
ltl = LTL(response)
response = ltl.re_replace(response)
dfa = ltl.to_networkx()
dot, num = ltl.get_dot_and_number()
# get the alphabet
alpha = ltl.get_alphabet(response)
## Get all the guards
guards = ltl.get_guards(dfa)
eval_pro = evaluate_prompt(task_spec, dot)
eval = get_response(eval_pro)
eval = int(eval)
## For each guard, generate the corresponding event combination
records = {'task': ['( '+response+' )'], 'spec': [eval_pro], 'dot': [dot], 'eval': [eval], 'num': [num]} # {ltl_task: ,specification:, DOT:..., evaluate:0-100}}}}
prediction_table.add_data(response, task_spec, dot, eval, num)

truth_assignment = {}
for guard in guards.values():
    truth_assignment[guard] = []
    truth_assignment[guard].extend(ltl.get_events(guard))
walks = ltl.random_walk(walk_num=15, walk_length=10)

def get_progress(exam_task, guard, truth_assignment, alphaset):
    truths = truth_assignment[guard]  
    event_str = ''
    progress_str = exam_task
    for truth in truths:
        events = [atomic for atomic, value in truth.items() if value is True]
        event_str = get_policy_str(alphaset, events)
        #progress the ltl task and store them into the records
        ltl_tree = lt.ltl2tree(exam_task, ltl.get_alphabet(exam_task))
        # ltl_str = lt.ltl_tree_str(ltl_tree)
        ltl_list = lt.unroll_tree(ltl_tree)
        progress_task = progress(ltl_list, events)
        progress_str = lt.reconstruct(progress_task)
        progress_str = regular(progress_str)
    return event_str, progress_str

def get_policy_str(alphaset, events):
    policy_str = '{'
    for char in alphaset:
        if char in events:
            policy_str = policy_str + char + ' is true,'
        else:
            policy_str = policy_str + char + ' is false,'
    policy_str = policy_str.rstrip(',') + '}'
    return policy_str

def get_dot_num(task):
    ltl = LTL(task)
    task = ltl.re_replace(task)
    dot, num = ltl.get_dot_and_number()
    return dot, num

def get_eval_score(records): #TODO
    assert len(records['task']) == len(records['spec']) == len(records['dot']) == len(records['eval']) == len(records['num']), 'entry numbers are not equal!'
    max_num = max(records['num'])
    eval_score = 0
    for i, value in enumerate(records['eval']):
        eval_score += (max_num - records['num'][i])/max_num * value
    eval_score /= len(records['task'])

def add_record(records, task, eval_pro, dot, eval, num):
    records['task'].append(task)
    records['spec'].append(eval_pro)
    records['dot'].append(dot)
    records['eval'].append(eval)
    records['num'].append(num)

for walk in walks:
    exam_task = regular(response)
    #tem_spec = task_spec
    event = ''
    for guard in walk:
        # progress ltl task and natural language task specification
        event_str, progress_str = get_progress(exam_task, guard, truth_assignment, alpha)
        event = event+','+event_str if event_str is not '' else event+ ','+ 'no action'
        if progress_str == 'True' or progress_str == 'False':
            eval_pro = evaluate_transit(task_spec, event, progress_str)
            eval = get_response(eval_pro)
            eval = int(eval)
            add_record(records, exam_task, eval_pro, None, eval, None)
            prediction_table.add_data(exam_task, eval_pro, None, eval, None)
            break
        # proc = process_prompt(tem_spec, event_str)
        # tem_spec = get_response(proc)
        # tem_spec = tem_spec + 'a'
        # if the progressed task has been in records, add the new natural language task specification to augment
        # if exam_task in records.keys():
        #     # records[exam_task]['spec'].append(tem_spec)
        #     continue 
        # # if the progressed task not in records, add them
        # else: 
        exam_task = regular(progress_str)
        dot, num = get_dot_num(exam_task)
        eval_pro = evaluate_raw_spec(task_spec, event, dot)
        eval = get_response(eval_pro)
        eval = int(eval)
        add_record(records, exam_task, eval_pro, dot, eval, num)
        prediction_table.add_data(exam_task, eval_pro, dot, eval, num)
wandb.log({'predictions': prediction_table})
wandb.finish()
eval_score = get_eval_score(records)