## These tasks are used to train the embedding model and should be stored in '../GA/tasks.txt'.
##TODO: load from config
LTL_OPS = ['A', 'O', 'N', 'G', 'U', 'X', 'E']
APs = ['a', 'b', 'c', 'd', 'e', 'f']

file_path = "../GA/normal_task.txt"
with open(file_path) as f:
    tasks = f.read().splitlines()
print(tasks[0])
print(len(tasks))
f.close()

## Preprocess the tasks.
## Remove the AP-repeated tasks.
import difflib
def compare_tasks(task_a, task_b):
    # Return true if two tasks are only different in atomic propositions
    d = difflib.Differ()
    diffs = list(d.compare(task_a, task_b))
    for diff in diffs:
        if diff.startswith('+') or diff.startswith('-'):
            if diff[-1] in LTL_OPS:
                return False
    return True

def onlyAP(task, record_task): 
    # Return true if some task only AP different has been added in list
    for record in record_task:
        if compare_tasks(task, record):
            return True
    return False

from tqdm import tqdm
print('Preprocessing tasks...')
record_task = [tasks[0]]
for i in tqdm(range(len(tasks))):
    if onlyAP(tasks[i], record_task):
        continue
    else:
        record_task.append(tasks[i])     

import json, openai
# You need to set your OPENAI API key here
# https://beta.openai.com/account/api-keys
#openai.api_key = "TO_BE_SET"
openai.api_key = "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"

PROMPT = """Given a linear temporal logic task {ltlt}, assume this task doesn't satisfy the expected behavior, specify 5 similar tasks with corresponding behaviors, and analyse the mistake of {ltlt}. The following is an example, the subsequent output should have the same output format without other explaination.

Task: 
F(a U (b A X c))

Output:

Possible mistake: expected behavior is {a} should be true first. Such specification omits {a}.
Revised task: a U (b A X (c))

Possible mistake: expected behavior is to make {b} and {c} be true simultanously.
Revised task: F(a U (b A c))

Possible mistake: expected behavior is to make {b} and {c} be true sequentially and don't need to be adjacent.
Revised task: F(a U (b A F(c)))

Possible mistake: expected behavior is to make {a}, {b} and {c} be true sequentially and don't need to be adjacent.
Revised task: F(a A F(b A F(c)))

Possible mistake: expected behavior is {a}, {b} and {c} finally be true simultanously.
Revised task: F(a A b A c)

Possible mistake: expected behavior is more than one atomic proposition in {{a}, {b}, {c}} should be true.
Revised task: F(a O b O c)

Possible mistake: expected behavior is {b} shoule be true, or {b} should not be true first, and then {b} or {c} should be true.
Revised task: F(a U (b O X(c)))

Possible mistake: expected behavior is {b} and {c} should be true simultanously, and then {c} should always be true.
Revised task: F(a U (b A G(c)))

Possible mistake: expected behavior is finally {c} should be true, no matter what the values of {a} and {b} are.
Revised task: F(a U (b U c))

Possible mistake: expected behavior is {c} should always be true, or {a} and {b} be true, but subsequently {c} should be true.
Revised task: G(a U (b U c))

Task: 
TASK-TO-BE-PLACED

Outputs:
"""
def rephrase_a_sentence(task):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=PROMPT.replace("TASK-TO-BE-PLACED", task),
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        best_of=1,
        frequency_penalty=0.1,
        presence_penalty=0
        )
    output = response['choices'][0]['text']
    try:
        mistakes, revised = generate_response(output)
    except:
        print("Error in parsing response")
        print(output)
        return output, "ERROR"
    #print(mistakes, revised)
    return generate_response(output)

def generate_response(response):
    lines = response.split('\n')
    print(lines[0])
    response = {}
    mistakes = []
    revised = []
    for idx, line in enumerate(lines):
        if line.startswith('Possible mistake:'):
            mistakes.append(line.replace('Possible mistake:', '').lstrip().rstrip())
        elif line.startswith('Revised task:'):
            revised.append(line.replace('Revised task:', '').lstrip().rstrip())
        else:
            continue
    assert len(mistakes) == len(revised), 'The number of mistakes and revised tasks should be the same!'
    return mistakes, revised

info = {}
def add_dict(task, mistakes, revised):
    info[task] = {}
    if len(mistakes) == len(revised):
        for i, mistake in enumerate(mistakes):
            info[task][mistake] = revised[i]
    else:
        print('Length of mistakes and revised are not equal!')
        r = min(len(mistakes), len(revised))
        for i in range(r):
            info[task][mistakes[i]] = revised[i]

print('Begin to generate responses...')
from tqdm import tqdm
# for i in tqdm(range(59, int(len(record_task)/10))):
for i in tqdm(range(60, int(len(record_task))), leave=None):
    task = record_task[i].strip()
    mistakes, revised = rephrase_a_sentence(task)
    add_dict(task, mistakes, revised)

with open("retrieve_msg.json", "w") as file:
    json.dump(info, file, indent=2)

file.close()

# with open('retrieve_msg.json') as json_file:
#     data = json.load(json_file)

# print('a')