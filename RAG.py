import gym
import torch as th

from GA.custom_policy import CustomActorCriticPolicy
from GA.envs.ltl_bootcamp import LTLBootcamp

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import openai

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
## Initialize. load the retrieve dataset and get the embedding
env = LTLBootcamp()
model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./tensorboard", device=DEVICE)

# Load pre-trained weights for the LTL module
model.policy.mlp_extractor.ltl_embedder.load_state_dict(th.load("./pre_logs/weights_ltl.pt"))
model.policy.mlp_extractor.ga.load_state_dict(th.load("./pre_logs/weights_ga.pt"))
# Get the embedding of the generated task
def get_emb(task):
    env.task = task
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.policy.mlp_extractor.ga.register_forward_hook(get_activation('ga'))
    output = model(env.gen_obs())  # is it correct? check the pretrain whats the input of the forward function, and which part is ltl
    em_query = activation['ga']
    return em_query
# Load the retrieve dataset
import json
with open('./Retrieve/retrieve_msg.json') as retrieve_data:
    retrieve_dict = json.load(retrieve_data)

embedding = []
tasks_r = []
for task_r in retrieve_dict.keys():
    emb = get_emb(task_r)
    embedding.append(emb)
    tasks_r.append(task_r)
    


## Input the prompt, and the task to be generated
GEN_PROMPT = """

Think step by step, and output the task without extra explaination.


generate the linear temporal logic of: TASK-TO-BE-REPLACED"""


def rephrase_a_sentence(nl_task):
    import re
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=GEN_PROMPT.replace("TASK-TO-BE-PLACED", nl_task),
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        best_of=1,
        frequency_penalty=0.1,
        presence_penalty=0
        )
    output = response['choices'][0]['text']
    # LTL task should be included in brackets
    ltl = re.findall(r'\(.*?\)', output)  
    return ltl

nl_task = input("Specify the task:\n")
## Get the result
task = rephrase_a_sentence(nl_task)

## Step 1: Evaluate the generated task 
# ltl2ba, generate the policy sketch
from ltl_utils import LTL
violation = True
times = 0

def check_violation(task_spec, random_walks):
    satisfy = True
    for walk in random_walks:
        parsed_policy, truth = ltl_model.eval_parse(walk)
        if truth:
            # prompt the LLM to find whether this policy sketch can complete the task, if not, it should generate error message
            ERR_PROMPT = """
                        dsfasdf
                            """
        def get_feedback(task_spec, prompt):
            import re
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt.replace("TASK-TO-BE-PLACED", parsed_policy).replace...task_spec,
                temperature=0.7,
                max_tokens=512,
                top_p=1,
                best_of=1,
                frequency_penalty=0.1,
                presence_penalty=0
                )
            output = response['choices'][0]['text']
            # LTL task should be included in brackets
            satisfy = re.findall(r'\{.*?\}', output)
            message = re.findall(r'\(.*?\)', output)  
            return satisfy, message
        
        satisfy, error_msg = get_feedback(nl_task, ERR_PROMPT)
        if not satisfy:
            # generated task doesn't satisfy task specification, need to be revised, just return
            break
    return satisfy, error_msg

ltl_model = LTL(task)
random_walks = ltl_model.random_walk()
violation, error_msg = check_violation(nl_task, random_walks)

while violation and times <= 5:
    #as long as there is violation, we need to regenerate the ltl task
    ## Step 2: find the top-k similar tasks
    times += 1
    
    # Find out the top-k similar tasks
    def top_k_tasks(k = 3):
        import numpy as np
        def cosine(x, y):
            # Compute the dot product between x and y
            dot_product = np.dot(x, y)
            # Compute the L2 norms (magnitudes) of x and y
            magnitude_x = np.sqrt(np.sum(x**2)) 
            magnitude_y = np.sqrt(np.sum(y**2))
            # Compute the cosine similarity
            cosine_similarity = dot_product / (magnitude_x * magnitude_y)
            return cosine_similarity
        
        top_k = []
        mini_sim = -1
        for re_task in retrieve_dict.keys():

            pass

    # Find out the top-k similar error messages

    ## Step 3: Use these records as prompt to revise the generated task
    ...
    task = response
    ltl_model = LTL(task)
    random_walks = ltl_model.random_walk()
    violation, error_msg = check_violation(nl_task, random_walks)
