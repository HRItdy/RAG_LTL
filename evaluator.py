import torch as th

from GA.custom_policy import CustomActorCriticPolicy
from GA.envs.ltl_bootcamp import LTLBootcamp

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from ltl_utils import LTL
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

ltl = LTL(response)
dfa = ltl.to_networkx()
# get all the guards
guards = LTL.get_guards(dfa)
# for each guard, generate the corresponding 


# Evaluate the generated task 
times = 0
top_k = 3 # top_k

while violation and times <= 5:
    #as long as there is violation, we need to regenerate the ltl task
    ## Step 2: find the top-k similar tasks
    times += 1  
    emb = get_task_embedding(task)
    # Get the top k most similar vectors
    top_k_indices = find_top_k_similar_tasks(emb, embedding, top_k)
    top_k_tasks = [tasks_r[i] for i in top_k_indices]   
    # Find out the top-k similar error messages within top-k tasks
    tasks, errors, tasks_p = find_top_k_similar_error(error_msg, top_k_tasks, retrieve_dict, top_k) 
    RE_PROMPT = revise_prompt(origin=task, error=error_msg, k=top_k, tasks=tasks, errors=errors, revise=tasks_p)
    ## Step 3: Use these records as prompt to revise the generated task
    task = get_response(RE_PROMPT)[0]
    ltl_model = LTL(task)
    random_walks = ltl_model.random_walk()
    violation, error_msg = check_violation(nl_task, random_walks)