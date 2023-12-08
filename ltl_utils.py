import networkx as nx
from flloat.parser.ltlf import LTLfParser
from ltlf2dfa.parser.ltlf import LTLfParser as LTLf_dot
from itertools import product

import sympy

class LTL():
    def __init__(self, task) -> None:
        parser = LTLfParser()
        dot_parser = LTLf_dot()
        self.task_str = self.replace(task)
        self.task = parser(self.task_str)
        self.dot_task = dot_parser(self.task_str)

    def replace(self, task):
        # Replace other symbols with the standard symbols for flloat  ['A', 'O', 'N', 'G', 'U', 'X', 'E'] -> ['&', '|', '!', 'G', 'U', 'X', 'F']
        task = task.replace('A','&')
        task = task.replace('O','|')
        task = task.replace('N','!')
        task = task.replace('~','!')
        task = task.replace('E','F')
        return task

    def to_dfa(self):
        return self.task.to_automaton()
    
    def to_graphviz(self):
        return self.to_dfa.to_graphviz()
    
    def to_networkx(self):
        dfa = self.to_dfa()
        G = nx.DiGraph()

        # Record the initial state
        self.start_node = dfa.initial_state

        # Add the accepting nodes
        for node in dfa.states:
            if node in dfa.accepting_states:
                G.add_node(node, accept = True)
            else:
                G.add_node(node, accept = False)

        # Add the transitions
        def dict_to_list(d, prefix=[]):
            result = []
            for key, value in d.items():
                current_key = prefix + [key]
                if isinstance(value, dict):
                    result.extend(dict_to_list(value, current_key))
                else:
                    result.append(current_key + [value])
            return result
        edges = dict_to_list(dfa._transition_function)
        for edge in edges:
            G.add_edge(edge[0], edge[1], guard = str(edge[2]))
        return G

    def random_walk(self, walk_num=10, walk_length=25, end_in_accept=True, maximal_attempt = 6):
        # If want to generate random walk with fixed length, end_in_accept should be False
        # maximal_attampt is the maximal number of steps trying to reach the accept node
        import random
        task_graph = self.to_networkx()
        walks = [[] for i in range(walk_num)]
        accepts = nx.get_node_attributes(task_graph, "accept")
        guards = nx.get_edge_attributes(task_graph, "guard")
        for i in range(walk_num):
            current_node = self.start_node
            attempt = 0
            if end_in_accept:
                # Avoid the first visit of the accepting node, and need to generate policy with length of at least minimal_attempt 
                while (not accepts[current_node] and attempt <= maximal_attempt) \
                        or (accepts[current_node] and attempt == 0):  
                    neighbors = list(task_graph.neighbors(current_node))
                    if not neighbors:
                        break  # Stop if there are no neighbors
                    next_node = random.choice(neighbors)
                    walks[i].append(guards[(current_node, next_node)])
                    attempt += 1
                    current_node = next_node
            else:
                while attempt <= walk_length:
                    neighbors = list(task_graph.neighbors(current_node))
                    if not neighbors:
                        break  # Stop if there are no neighbors
                    next_node = random.choice(neighbors)
                    walks[i].append(guards[(current_node, next_node)])
                    attempt += 1
                    current_node = next_node
        return walks

    def eval_parse(self, policy_sketch, timestep=0, mini_length = 3):
        def parse_policy(policy_sketch, mini_length=3):
            import random
            #generate more diverse policy sketches
            #example:
            #    too short policies: duplicate the atomic to distinguish 'always' and 'eventually'
            parsed_policy = []
            if len(policy_sketch)<=mini_length:
                # enhance the policy and permute the list
                for i in range(mini_length):
                    parsed_policy.append(random.choice(policy_sketch))
            return parsed_policy
        parsed_policy = parse_policy(policy_sketch, mini_length)
        return parsed_policy, self.task.truth(parsed_policy, timestep)
    
    def get_dot(self):
        task_graph = self.to_networkx()
        dot = nx.drawing.nx_pydot.to_pydot(task_graph)
        dot = dot.to_string()
        return dot
    
    def get_guards(self, G):
        guards = nx.get_edge_attributes(G, "guard")
        return guards

    def get_alphabet(self, task):
        alphabet = set()
        for char in task:
            if char not in ['A', 'O', 'N', 'G', 'U', 'X', 'E', '&', '|', '!', 'F', ' ', '\\', '~']:
                alphabet.add(char)
        return alphabet
    
    def get_events(self, guard):
        #decompose the guards into composition of atomic propositions
        if guard in ['True', 'False']:
            return []
        guard = self.replace(guard)
        events = []
        alphabet = self.get_alphabet(guard)
        guard_parser = LTLfParser()
        parsed_guard = guard_parser(guard)
        def combo(alphabet):
            # Generate all possible combinations of True/False for the selected letters
            combinations = list(product([True, False], repeat=len(alphabet)))
            # Format the output as a list of dictionaries
            output = []
            for combo in combinations:
                output.append({letter: value for letter, value in zip(alphabet, combo)})
            return output
        combos = combo(alphabet)
        for combo in combos:
            if parsed_guard.truth([combo], 0):
                events.append(combo)
        return events

if __name__ == '__main__':
    a = LTL('X(!(a U b))')
    l = a.get_events('a | !b')
    print('a')

    


