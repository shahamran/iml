from valueIteration import valueIteration
import matplotlib.pyplot as plt
from enum import IntEnum
import numpy as np

GAMMA = 0.75


class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    COUNT = 2


class States:
    START = 4
    COUNT = 5


class Prob:
    SUCESS = 1
    FAIL = 0


def construct_model():
    tau = np.ones((States.COUNT, Action.COUNT, States.COUNT)) * Prob.FAIL
    leftables = np.arange(4) + 1
    rightables = leftables - 1
    tau[leftables, Action.LEFT, leftables - 1] = Prob.SUCESS
    tau[rightables, Action.RIGHT, rightables + 1] = Prob.SUCESS
    tau[0, Action.LEFT, States.START] = Prob.SUCESS
    tau[-1, Action.RIGHT, States.START] = Prob.SUCESS

    rho = np.zeros((States.COUNT, Action.COUNT))
    rho[0, Action.LEFT] = 6
    rho[-1, Action.RIGHT] = 1
    return tau, rho


def run_val_iter(tau, rho, gamma=GAMMA):
    # V = valueIteration(States.COUNT, Action.COUNT,
    #                    tau, rho, gamma)
    # print('Converged V:')
    # print(V)
    plt.subplots(3, 1)
    plt.suptitle('Value Iteration -- Patience, dear')
    for i, gamma in enumerate([0.5, 0.75, 0.85]):
        plt.subplot(3, 1, i+1)
        V = valueIteration(States.COUNT, Action.COUNT, tau, rho, gamma)
        print('V for gamma = %f:' % gamma)
        print(V)
        plt.imshow(V.reshape((1, -1)), cmap='hot')
        plt.title(r'$\gamma = %f$' % gamma)
        plt.axis('off')


def plot_graph(tau, rho):
    gammas = np.arange(0.5, 0.99, 0.01)
    V = np.zeros(gammas.shape)
    for i, gamma in enumerate(gammas):
        V[i] = valueIteration(States.COUNT, Action.COUNT, tau, rho, gamma)[-1]
    plt.figure()
    plt.plot(gammas, V)
    plt.title(r'$s_0$ for different $\gamma$ values')

def main():
    tau, rho = construct_model()
    run_val_iter(tau, rho)
    plot_graph(tau, rho)
    plt.show()


if __name__ == '__main__':
    main()