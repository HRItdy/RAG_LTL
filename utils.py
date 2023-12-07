import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

def get_text_embedding(text_to_embed):
	# Embed a line of text
	response = openai.Embedding.create(
    	model= "text-embedding-ada-002",
    	input=[text_to_embed]
	)
	# Extract the AI output embedding as a list of floats
	embedding = response["data"][0]["embedding"]
	return embedding

def get_task_embedding(task_to_embed, env, model):
    env.task = task_to_embed
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.policy.mlp_extractor.ga.register_forward_hook(get_activation('ga'))
    output = model(env.gen_obs())  # is it correct? check the pretrain whats the input of the forward function, and which part is ltl
    embedding = activation['ga']
    return embedding

def find_top_k_similar_error(error_msg, top_k_tasks, retrieve_dict, k):
    error_emb = get_text_embedding(error_msg)
    ori_tasks = []
    errors = []
    revised_tasks = []
    error_embs = []
    for task in top_k_tasks:
        for error in retrieve_dict[task].keys():
            ori_tasks.append(task)
            errors.append(error)
            revised_tasks.append(retrieve_dict[task][error])
            error_embs.append(get_text_embedding(error))

    error_emb = get_text_embedding(error_msg)
    top_k_indices = find_top_k_similar_tasks(error_emb, error_embs, k)
    tasks = [ori_tasks[i] for i in top_k_indices]
    error_msgs = [errors[i] for i in top_k_indices]
    revised_task = [revised_tasks[i] for i in top_k_indices]
    return tasks, error_msgs, revised_task
 
def find_top_k_similar_tasks(emb, emb_list, k):
    # Convert vectors to NumPy arrays
    emb = np.array([emb])
    emb_list = np.array(emb_list)
    # Compute cosine similarity
    similarities = cosine_similarity(emb, emb_list)
    # Get indices of the top k most similar vectors
    top_k_indices = np.argsort(similarities[0])[-k:][::-1]
    return top_k_indices 

def check_violation(task_spec, random_walks, ltl_model):
    satisfy = True
    for walk in random_walks:
        parsed_policy, truth = ltl_model.eval_parse(walk)
        if truth:
            # prompt the LLM to find whether this policy sketch can complete the task, if not, it should generate error message
            ERR_PROMPT = evaluate_prompt(task_spec, parsed_policy) 
            results = get_response(ERR_PROMPT)
            satisfy, error_msg = results[0], results[1]
        if not satisfy:
            # generated task doesn't satisfy task specification, need to be revised, just return
            break
    return satisfy, error_msg

def generate_prompt(nl_task):
    prompt = "Generate the linear temporal logic task of: TASK-TO-BE-REPLACED. The output should be the task without extra explanation. You should use '&', '|', '!', 'G', 'U', 'X', 'F' for 'and', 'or', 'not', 'always', 'until', 'next', 'Eventually'.".replace('TASK-TO-BE-REPLACED', nl_task)
    return prompt

def evaluate_prompt(task_spec, policy_sketch):
    #preprocess the policy_sketch first?
    policy_sketch = ",".join(step for step in policy_sketch)
    prompt = """Analyze whether the policy sketch satisfies the task. If not, give out the output. 
                Example:
                task specification: \{c\} should always be true, or \{a\} and \{b\} be true, but subsequently \{c\} should be true.
                policy sketch: 'a & b', 'a & b & c', 'a & b'
                output: satisfy: (No), error message:('a & b' after 'a & b & c'. According to the LTL task, once c becomes true, it should always remain true. However, the policy returns to a state where c's truth is not guaranteed. This violates the requirement that "subsequently c should be true" after its initial truth.)
                Now the task specification is:""" + task_spec + "\n" + """policy sketch :""" + policy_sketch + """. Give the output following the format in the example without extra explanation.
                output:"""
    return prompt

def revise_prompt(org, error, k, **kwargs):
    prompt = "The original task is: " + org + "\n" \
             +"Detected error message: " + error + "\n" \
             +"Given the following examples, please give out the revised task in {} without extra explanation: \n" \
             +"".join(["Original task is: "+kwargs['tasks'][i]+ ", error message is: "+kwargs['errors'][i] +", \
                       the revised task is: "+kwargs['revise'][i]+"." for i in k]) + "\n" \
             +"Output:\n"
    return prompt    

def get_response(prompt):
    #assert prompt_response in ['generate', 'evaluate', 'revise'], "Unknown prompt type!"
    import re
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        best_of=1,
        frequency_penalty=0.1,
        presence_penalty=0
        )
    output = response['choices'][0]['text']
    output = output.lstrip('\n')
    # result = re.findall(r'\{.*?})', output)  
    return output

