import cvxpy as cp
import numpy as np
import sys


def construct_objective_matrix(P):
    """Construct matrix for quadratic programming objective.
    """
    n_taxa = P[0].shape[1]
    Q = np.zeros((n_taxa, n_taxa))
    for p in P:
        p /= p.sum(axis=1, keepdims=True)
        Q += np.cov(p.T, ddof=0)
    Q = 0.5*(Q + Q.T)
    return Q


def find_stable_subset(P):
    print("Running MLIP solver for denominator...")
    Q = construct_objective_matrix(P)
    n_taxa = P[0].shape[1]

    # z_k = x_i * x_j
    # i = 0...n-1
    # j = 0...i
    # i+1 entries per row
    # => row i starts at 0 + 1 + 2 + ... + i = 0.5*i*(i+1) 
    # k = 0.5*i*(i+1) - 1 + j

    k_to_ij = {}
    ij_to_k = {}
    for i in range(n_taxa):
        for j in range(i+1):
            k = int(0.5*i*(i+1)) + j
            k_to_ij[k] = (i, j)
            ij_to_k[(i,j)] = k

    # total number of variables is all pairs i,j
    # for x_i*x_j plus diagonal x_i*x_i
    nvar = int(0.5*n_taxa*(n_taxa-1) + n_taxa)
    z = cp.Variable(nvar, boolean=True)
    constraints = []

    # get x_i x_i entries
    idxes = [ij_to_k[(i,i)] for i in range(n_taxa)]
    constraints.append(cp.sum(z[idxes]) <= n_taxa-1)

    # the sum must not vanish on any observation
    for p in P:
        p /= p.sum(axis=1,keepdims=True)
        for pt in p:
            constraints.append(cp.sum(pt*z[idxes]) >= 1e-4)

    # x_i x_i >= x_i x_j
    for i in range(n_taxa):
        k1 = ij_to_k[(i,i)]
        for j in range(0,i+1):
            k2 = ij_to_k[(i,j)]
            constraints.append(z[k1] >= z[k2])

    obj_expr = []
    for k in range(nvar):
        i,j = k_to_ij[k]
        if i == j:
            obj_expr.append(0.5*z[k]*Q[i,j])
        else:
            obj_expr.append(z[k]*Q[i,j])
            
    for i in range(n_taxa):
        k = ij_to_k[(i,i)]
        obj_expr.append(z[k])

    obj_fn = cp.sum(obj_expr)

    obj = cp.Minimize(obj_fn)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver="GLPK_MI")

    if prob.status != "optimal":
        print("Error: MILP solver failed", file=sys.stderr)
        exit(1)

    #print("Problem status:", prob.status)
    #print("The optimal value is", prob.value)

    # extract diagonal
    z = z.value
    x = np.zeros(n_taxa)
    for i in range(n_taxa):
        k = ij_to_k[(i,i)]
        x[i] = z[k]
    #print("The optimal x is", x)

    for i in range(n_taxa):
        for j in range(i+1):
            k = ij_to_k[(i,j)]

    print()
    return np.argwhere(x == 1).flatten()


