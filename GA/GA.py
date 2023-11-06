
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class GA(nn.Module):

    def __init__(self,
                 seq_len,       # sequence length
                 num_rel,       # number of possible relations
                 num_heads,     # number of attention heads
                 num_layers,    # deep of the module
                 device='cpu',  # {'cuda' | 'cpu'}
                ):
        super().__init__()

        self.num_heads = num_heads

        # relation embedding layer
        self.rel_embs = nn.Embedding(num_rel,
                                     num_heads,
                                     padding_idx=0)

        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, "lin" + str(i), nn.Parameter(torch.randn((seq_len, num_heads, 1))))
            setattr(self, "bias" + str(i), nn.Parameter(torch.randn((seq_len, 1))))

        self.to(device)
        self.device = device


    def forward(self, x, relations):
        '''
        params:
            x:          torch.FloatTensor [B,S,D]
            relations:  torch.LongTensor  [B,S,S]
        returns:
            res:        torch.FloatTensor [B,S,D]
        '''

        # Relation-based multi head self-attention

        A = self.rel_embs(relations) # [B,S,S,H]
        A = A.transpose(3,2).transpose(2,1) # [B,H,S,S]
        A = F.softmax(A, dim=-1) # row-softmaxed relation matrix

        for i in range(self.num_layers):
            x = self.forward_(x, A, i)

        return x


    def forward_(self, x, A, idx_layer):
        '''
            x:          [B,S,D]
            A:          [B,H,S,S]
            idx_layer:  number of current layer
        '''
        
        lin = getattr(self, "lin" + str(idx_layer))
        bias = getattr(self, "bias" + str(idx_layer))

        x = x.unsqueeze(dim=1).repeat(1, self.num_heads, 1, 1) # [B,H,S,D]
        Ax = torch.matmul(A, x) # [B,H,S,D]

        # put all the heads together through a linear transformation
        Ax = Ax.transpose(1,2).transpose(2,3) # [B,S,D,H]
        r = torch.matmul(Ax, lin).squeeze(-1) + bias # [B,S,D]
        
        return torch.tanh(r) # non-linear activation function

def build_relations(formula, max_len=10):
    ''' Function to construct the skew-symmetric relation matrix. '''

    def length(formula):
        ''' Aux method to recursively determine the length of formula. '''
        if type(formula) == str:
            return 1
        if len(formula) == 2:
            return length(formula[1]) + 1
        if len(formula) == 3:
            return length(formula[1]) + length(formula[2]) + 1

    # define a max_len x max_len matrix with as many 1s on the diagonal as the length of formula
    rel_len = length(formula)   # length of formula
    pad_len = max_len - rel_len # length of padding
    mat = np.diag(np.concatenate([np.ones(rel_len), np.zeros(pad_len)]))

    # define a proper relation vocabulary
    V = {k: v for v, k in enumerate([" ", "s", "p", "l", "r", "p_", "l_", "r_"])}

    def tagger(formula, mat, prev_idx=-1, idx=0, rel=None):
        ''' Aux method to recursively fill the relation matrix based on formula. '''
        def aux(mat, i, j, rel):
            if prev_idx != -1 and rel is not None:
                mat[prev_idx, idx] = V[rel]
                mat[idx, prev_idx] = V[rel + '_']
        if type(formula) == str:
            aux(mat, prev_idx, idx, rel)
            return idx
        if len(formula) == 2:
            aux(mat, prev_idx, idx, rel)
            return tagger(formula[1], mat, idx, idx+1, 'p')
        if len(formula) == 3:
            aux(mat, prev_idx, idx, rel)
            offset = tagger(formula[1], mat, idx, idx+1, 'l')
            return tagger(formula[2], mat, idx, offset+1, 'r')

    tagger(formula, mat)

    # return the properly filled relation matrix of formula
    return mat