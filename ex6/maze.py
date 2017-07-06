from valueIteration import valueIteration
import matplotlib.pyplot as plt
from enum import IntEnum
import numpy as np

GAMMA = 0.75


class Action(IntEnum):
    LEFT = 0
    DOWN = 1
    UP = 2
    RIGHT = 3
    COUNT = 4


class States:
    LEFT_COL = np.array([0, 3, 6])
    MID_COL = LEFT_COL + 1
    RIGHT_COL = MID_COL + 1
    TOP_ROW = np.array([0, 1, 2])
    MID_ROW = TOP_ROW + 3
    BOTTOM_ROW = MID_ROW + 3
    ABSORB = 3
    CENTER = 4
    COUNT = 9


class Prob:
    SUCESS = 0.8
    FAIL = 0.2


def construct_maze():
    tau = np.zeros((States.COUNT, Action.COUNT, States.COUNT))
    leftables = np.append(States.RIGHT_COL, States.MID_COL)
    rightables = np.append(States.LEFT_COL, States.MID_COL)
    downables = np.append(States.TOP_ROW, States.MID_ROW)
    upables = np.append(States.BOTTOM_ROW, States.MID_ROW)
    directions = [leftables, rightables, downables, upables]
    actions = [Action.LEFT, Action.RIGHT, Action.DOWN, Action.UP]
    operations = [lambda x: x-1, lambda x: x+1,
                  lambda x: x+3, lambda x: x-3]
    # define all actions as if there are no walls or absorbing state
    for i, direction in enumerate(directions):
        action = actions[i]
        operation = operations[i]
        result = operation(direction)
        tau[direction, action, result] = Prob.SUCESS
        tau[direction, action, direction] = Prob.FAIL
    # define walls and edges
    left_stuck = np.append(States.LEFT_COL, [5])
    right_stuck = np.append(States.RIGHT_COL, [4])
    down_stuck = np.append(States.BOTTOM_ROW, [0, 4])
    up_stuck = np.append(States.TOP_ROW, [7])
    for i, stuck in enumerate([left_stuck, right_stuck, down_stuck, up_stuck]):
        action = actions[i]
        tau[stuck, action, :] = 0
        tau[stuck, action, stuck] = 1
    # define absorbing state
    tau[States.ABSORB, :, :] = 0
    tau[States.ABSORB, :, States.ABSORB] = 1

    rho = np.ones((States.COUNT, Action.COUNT)) * (-6)
    # absorbing
    rho[States.ABSORB, :] = 0
    # expected reward for legal moves
    reward = Prob.SUCESS * (-1) + Prob.FAIL * (-6)
    rho[upables, Action.UP] = reward
    rho[downables, Action.DOWN] = reward
    rho[leftables, Action.LEFT] = reward
    rho[rightables, Action.RIGHT] = reward
    return tau, rho


def run_val_iter(tau, rho, gamma=GAMMA):
    V = valueIteration(States.COUNT, Action.COUNT,
                       tau, rho, gamma)
    print('Converged V:')
    print(V.reshape((3,3)))
    plt.subplots(2, 2)
    plt.suptitle('Value Iteration -- Maze')
    for iters in range(1,5):
        plt.subplot(2, 2, iters)
        V = valueIteration(States.COUNT, Action.COUNT,
                           tau, rho, gamma, iters)
        plt.imshow(V.reshape((3,3)), cmap='hot')
        plt.title(str(iters) + ' iterations')
        plt.axis('off')
    plt.show()


def main():
    tau, rho = construct_maze()
    run_val_iter(tau, rho)


if __name__ == '__main__':
    main()