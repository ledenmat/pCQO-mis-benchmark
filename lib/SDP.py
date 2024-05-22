from mosek.fusion import Model, Domain, ObjectiveSense, Expr
import torch
import numpy as np


def solve_SDP(graph):
    n = len(graph)
    matrix_of_ones = np.ones((n, n))

    with Model("SDP_init") as m:
        X = m.variable("X", [n, n], Domain.inPSDCone())
        objective_expr = Expr.dot(matrix_of_ones, X)
        m.objective("obj", ObjectiveSense.Maximize, objective_expr)
        m.constraint("tr", Expr.sum(X.diag()), Domain.equalsTo(1.0))
        for edge in graph.edges:
            i, j = edge
            m.constraint(f"con_{i}_{j}", X.index(i, j), Domain.equalsTo(0.0))
            m.constraint(f"con_{j}_{i}", X.index(j, i), Domain.equalsTo(0.0))

        m.solve()
        X_solution = X.level()
        X_solution = np.array(X_solution)
        X_solution = X_solution.reshape(n, n)

    X_SDP_solution = X_solution

    ## Coding W_SDP_uv:
    W_SDP_uv = torch.zeros(n, n)

    # create a dictionary of key = edge (u,v) in G' and value is equation to X(u,v)
    G_hat_edges_strength_dict = {}

    for v in range(n):
        for u in range(n):
            if (v, u) not in graph.edges() and u != v:
                G_hat_edges_strength_dict[(v, u)] = X_SDP_solution[v, u]

    # Normalize
    values_array = np.array(list(G_hat_edges_strength_dict.values()))
    normalized_values_MINMAX = (values_array - np.min(values_array)) / (
        np.max(values_array) - np.min(values_array)
    )
    G_hat_edges_strength_dict_normalized_MINMAX = {
        key: normalized_values_MINMAX[i]
        for i, key in enumerate(G_hat_edges_strength_dict)
    }  ######## USE THIS

    # Assign to W_SDP_uv
    for v in range(n):
        for u in range(n):
            if (v, u) not in graph.edges() and v != u:
                W_SDP_uv[v, u] = G_hat_edges_strength_dict_normalized_MINMAX[(v, u)]

    ## Coding s_SDP_v and X_SDP_init:

    sol_diag = np.diag(X_SDP_solution)
    # s_SDP_v = sol_diag / np.max(sol_diag)
    X_SDP_init = (sol_diag - np.min(sol_diag)) / (np.max(sol_diag) - np.min(sol_diag))

    ### Converting to torch tensors: Convert s_SDP_v and W_SDP_uv to torch tensors
    # s_SDP_v_tensor = torch.tensor(s_SDP_v, dtype=torch.float32)
    X_SDP_init_tensor = torch.tensor(X_SDP_init, dtype=torch.float32)
    # W_SDP_uv_tensor = torch.tensor(W_SDP_uv, dtype=torch.float32)

    return X_SDP_init_tensor
