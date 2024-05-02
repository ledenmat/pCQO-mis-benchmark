import time
import numpy as np
import torch
from torch.nn import Module

import networkx as nx
from networkx import Graph


class iSCO:
    def __init__(self, G: Graph, params):
        super().__init__()
        self.learning_rate = params.get("learning_rate", 0.001)
        self.number_of_steps = params.get("number_of_steps", 10000)

        self.graph = G
        self.c = torch.ones(len(G))

        self.A = torch.Tensor(nx.adjacency_matrix(self.graph).todense()).to_dense()

        pass

    def solve(self):
        best = torch.ones(len(self.graph))
        best_eval = self.objective(self.c, best, self.A, 1.0001)

        current = best
        current_eval = best_eval

        temp = 10

        m_iterations = 10000
        n_iterations = 10

        for i in range(m_iterations):
            for j in range (n_iterations):
                candidate = current
                candidate_eval = self.objective(self.c, candidate, self.A, 1.0001)
                if candidate_eval < current_eval:
                    best = candidate
                    best_eval = candidate_eval

                difference = (1 - 2* self.c)

                t = 10

                metropolis = torch.exp(torch.ones(1))

                print(best_eval)

                # if difference < 0 or torch.rand() < metropolis:
                #     current = candidate
                #     current_eval = candidate_eval

            # postprocessing step
            

            temp = temp * 1(-(i+1)/m_iterations)
        pass

    def objective(self, c, x, A, penalty):
        # this is our energy function
        c_and_x = c.T @ x

        penalty_term = penalty * (x.T @ A @ x)/2

        return c_and_x + penalty_term
    
def post_processing(x, adjacency_matrix):
    d = len(x)  # Number of nodes
    changed = True  # Flag to indicate if any changes were made in the current iteration

    while changed:
        changed = False  # Reset changed flag for each iteration
        for i in range(d):
            if x[i] == 1:
                for j in range(d):
                    if i != j and adjacency_matrix[i][j] == 0 and x[j] == 1:
                        x[j] = 0
                        changed = True  # Set changed flag to True if a change was made
    return x


graph = nx.gnm_random_graph(5, 10)
solver = iSCO(graph, {})

solver.solve()


class PathAuxiliarySampler(Module):
    def __init__(self, R, args):
        super().__init__()
        self.R = R
        self.ess_ratio = args.ess_ratio
        self._steps = 0
        self._evals = []
        self._hops = []

    def step(self, x, model):
        # batch size
        bsize = x.shape[0]

        # rank of the x matrix
        x_rank = len(x.shape) - 1

        # radi
        radius = torch.randint(1, self.R * 2, size=(bsize, 1))
        max_r = torch.max(radius).item()
        r_mask = torch.arange(max_r).expand(bsize, max_r) < radius
        r_mask = r_mask.float().to(x.device)

        Zx, Zy = 1., 1.
        b_idx = torch.arange(bsize).to(x.device)
        cur_x = x.clone()
        with torch.no_grad():
            for step in range(max_r):
                score_change_x = model.change(cur_x) / 2.0
                if step == 0:
                    Zx = torch.logsumexp(score_change_x, dim=1)
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                cur_bits = cur_x[b_idx, index]
                new_bits = 1.0 - cur_bits
                cur_r_mask = r_mask[:, step]
                cur_x[b_idx, index] = cur_r_mask * new_bits + (1.0 - cur_r_mask) * cur_bits
            y = cur_x

        score_change_y = model.change(y) / 2.0
        Zy = torch.logsumexp(score_change_y, dim=1)
        
        log_acc = Zx - Zy
        accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
        new_x = y * accepted + (1.0 - accepted) * x
        self._steps += 1
        self._evals.append((0 + radius.sum()).item() / bsize)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x

    @property
    def evals(self):
        return self._evals[-1]

    @property
    def hops(self):
        return self._hops[-1]

    @property
    def avg_evals(self):
        ratio = self.ess_ratio
        return sum(self._evals[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def avg_hops(self):
        ratio = self.ess_ratio
        return sum(self._hops[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def weight(self):
        return torch.ones(2 * self.R - 1) / (2 * self.R - 1)