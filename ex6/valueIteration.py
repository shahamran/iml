import numpy as np
##
# Runs value iteration.
# sizeS - number of states
# sizeA - number of actions
# tau - transition prbabilities of the MDP, size is sizeSxsizeAxsizeS.
#       tau[s,a,s2] is the probability of moving to s2 after taking action a in
#       state s.
# rho - reward function of the MDP, size is sizeSxsizeA.
#       here rho[s,a] is the expected reward of taking actions a in state s.
# gamma - discount factor.
# num_iter - number of iterations to run, by default runs until convergence
# of the value with gap 1e-06.
##


def valueIteration(sizeS, sizeA, tau, rho, gamma, num_iter=-1):
    V = np.zeros(sizeS)
    Q = np.zeros((sizeS, sizeA))
    EPSILON = 1e-06
    delta = 1

    cur_iter = 0
    if num_iter == -1:
        cur_iter = -1

    while delta > EPSILON and (cur_iter < num_iter or cur_iter == -1):
        Q = rho + np.einsum('ijk,k->ij', gamma * tau, V)
        V_new = np.max(Q, axis=1)
        delta = np.max(np.abs(V - V_new))
        V = V_new
        cur_iter = cur_iter + 1 if (cur_iter != -1) else -1
    return V
