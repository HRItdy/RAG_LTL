import numpy as np
import random

from easydict import EasyDict as edict
from itertools import combinations
from .ltl2tree import *


def add_basic_ltl(alphabets):
    ltls = []
    for a in alphabets:
        ltls.append((a, None))
    for key, val in OP2NARG.items():
        if key == 'N': continue
        if val == 1:
            ltl = key + ' ' + np.random.choice(alphabets)
        else:
            args = np.random.choice(alphabets, 2, replace=False)
            ltl = args[0] + ' ' + key + ' ' + args[1]
        ltls.append((ltl, None))
    return ltls


def permute(alphabets, ltl):
    tokens = ltl.split(' ')
    for i, token in enumerate(tokens):
        flip = np.random.choice([True, False], p=[0.3, 0.7])
        if flip:
            if token in alphabets:
                tokens[i] = np.random.choice(alphabets)
            elif token in OP_1:
                tokens[i] = np.random.choice(OP_1)
            elif token in OP_2:
                tokens[i] = np.random.choice(OP_2)
    # sample to add a `not'
    out_tokens = []
    for i, token in enumerate(tokens):
        if token in alphabets or token == '(':
            add_not = np.random.choice([True, False], p=[0.05, 0.95])
            if add_not:
                out_tokens.append('N')
        out_tokens.append(token)
    return ' '.join(out_tokens)

def ltl_sampler(alphabets, 
                n_samples=100,
                add_basics=True,
                min_symbol_len=1,
                max_symbol_len=10,
                n_steps=15,
                n_accept=10**9):
    filtered_alphabets = [a for a in alphabets if 'C_' not in a]
    cfg = get_ltl_grammar(alphabets)
    ltls = []; considered = set()
    if add_basics:
        ltls = add_basic_ltl(alphabets)
        considered = set([ltl for ltl, _ in ltls])
    n_samples = n_samples - len(ltls)
    for i in range(n_samples):
        print('Generate {}th formula'.format(i))
        while True:
            # generate LTL formula (including its pair)
            ltl = generate_ltl(cfg)
            symbols = [s for s in ltl.split(' ') if s != ')' and s != '(']
            if len(symbols) <= min_symbol_len or len(symbols) > max_symbol_len:
                considered.add(ltl)
                continue
            new_ltl = permute(alphabets, ltl)
            if new_ltl is None or new_ltl in considered:
                print(' reject {}'.format(new_ltl))
                continue
            # add the generated formulas
            print(' add {}'.format(new_ltl))
            considered.add(new_ltl)
            new_ltl = replace_symbols(new_ltl)
            ltls.append((new_ltl, ltl))
            considered.add(ltl)
            ltl = replace_symbols(ltl)
            break
            
    ltls.sort(key=lambda x: (len(x[0])))
    return ltls


if __name__ == '__main__':
    # parse ltl formula to cfg tree and then convert to an expression tree
    alphabets = ['a', 'b', 'c', 'd', 'e', 'f']
    # cfg_tree = parse_ltl(ltl_formula, alphabets)
    # print(cfg_tree)
    # ltl_tree, idx = convert_ltl_tree(cfg_tree)
    # print(ltl_tree_str(ltl_tree))
    ltls = ltl_sampler(alphabets, n_samples=5000)
