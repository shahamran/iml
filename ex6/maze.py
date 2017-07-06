from valueIteration import valueIteration as vi
import matplotlib.pyplot as plt
from enum import IntEnum
import numpy as np


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
    MID_ROW = TOP_ROW + 1
    BOTTOM_ROW = MID_ROW + 1
    ABSORB = 3
    CENTER = 4
    COUNT = 9


class Prob:
    SUCESS = 0.8
    FAIL = 0.2


def construct_maze():
    tau = np.zeros((States.COUNT, Action.COUNT, States.COUNT))
    leftables = States.RIGHT_COL + States.MID_COL
    rightables = States.LEFT_COL + States.MID_COL
    downables = States.TOP_ROW + States.MID_ROW
    upables = States.BOTTOM_ROW + States.MID_ROW
    directions = [leftables, rightables, downables, upables]
    actions = [Action.LEFT, Action.RIGHT, Action.DOWN, Action.UP]
    operations = [lambda x: x-1, lambda x: x+1,
                  lambda x: x+3, lambda x: x-3]
    # define all actions as if there are no walls or absorbing state
    for direction, action, operation in directions, actions, operations:
        result = operation(direction)
        tau[direction, action, result] = Prob.SUCESS
        tau[direction, action, direction] = Prob.FAIL
    # define walls and edges
    left_stuck = States.LEFT_COL + [5]
    right_stuck = States.RIGHT_COL + [4]
    down_stuck = States.BOTTOM_ROW + [0, 4]
    up_stuck = States.TOP_ROW + [7]
    for stuck, action in [left_stuck, right_stuck, down_stuck, up_stuck], \
                         actions:
        tau[stuck, action, :] = 0
        tau[stuck, action, stuck] = 1
    return tau


def run_val_iter(maze):
    pass


def plot_values(V):
    pass


def main():
    maze = construct_maze()
    V = run_val_iter(maze)
    plot_values(V)


if __name__ == '__main__':
    main()