import gym
import torch as th

# from GA.custom_policy import CustomActorCriticPolicy
# from GA.envs.ltl_bootcamp import LTLBootcamp

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import openai

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

## Input the prompt, and the task to be generated
# GEN_PROMPT = """

# Think step by step, and output the task without extra explaination.


# generate the linear temporal logic of: TASK-TO-BE-REPLACED"""


# def rephrase_a_sentence(nl_task):
#     import re
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=GEN_PROMPT.replace("TASK-TO-BE-PLACED", nl_task),
#         temperature=0.7,
#         max_tokens=512,
#         top_p=1,
#         best_of=1,
#         frequency_penalty=0.1,
#         presence_penalty=0
#         )
#     output = response['choices'][0]['text']
#     # LTL task should be included in brackets
#     ltl = re.findall(r'\(.*?\)', output)  
#     return ltl

# nl_task = input("Specify the task:\n")
# ## Get the result
# task = rephrase_a_sentence(nl_task)

## Step 1: Evaluate the generated task 
# ltl2ba, generate the policy sketch
import networkx as nx
import pydot


from flloat.parser.ltlf import LTLfParser

# parse the formula
parser = LTLfParser()
formula = "F (a & !b)"
parsed_formula = parser(formula)

# # evaluate over finite traces
# t1 = [
#     {"a": False, "b": False},
#     {"a": True, "b": False},
#     {"a": True, "b": False},
#     {"a": True, "b": True},
#     {"a": False, "b": False},
# ]
# assert parsed_formula.truth(t1, 0)
# t2 = [
#     {"a": False, "b": False},
#     {"a": True, "b": True},
#     {"a": False, "b": True},
# ]
# assert not parsed_formula.truth(t2, 0)

# from LTLf formula to DFA
dfa = parsed_formula.to_automaton()
# assert dfa.accepts(t1)
# assert not dfa.accepts(t2)


# print the automaton
graph = dfa.to_graphviz()
graph.render("./my-automaton") 



graphs = pydot.graph_from_dot_data(dfa)
graph = graphs[0]
G = nx.Graph(nx.nx_pydot.from_pydot(graph))
nx.draw(G)

policy_sketch = []
# prompt the LLM to find the error message
ERR_PROMPT = """
dsfasdf
"""

error_msg = rephrase_a_sentence(policy_sketch)

## Step 2: find the top-k similar tasks
# env = LTLBootcamp()
env.task = task
model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./tensorboard", device=DEVICE)

# Load pre-trained weights for the LTL module
model.policy.mlp_extractor.ltl_embedder.load_state_dict(th.load("./pre_logs/weights_ltl.pt"))
model.policy.mlp_extractor.ga.load_state_dict(th.load("./pre_logs/weights_ga.pt"))
# Get the embedding of the generated task

# Load the retrieve dataset
#...json

# Find out the top-k similar tasks
for record in records:
    pass

# Find out the top-k similar error messages

## Step 3: Use these records as prompt to revise the generated task
