from valueIteration import valueIteration as vi
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np


class Action(Enum):
    LEFT = 0
    DOWN = 1
    UP = 2
    RIGHT = 3
    COUNT = 4


class States(Enum):
    LEFT_COL = np.array([0, 3, 6])
    MID_COL = LEFT_COL + 1
    RIGHT_COL = MID_COL + 1
    TOP_ROW = np.array([0, 1, 2])
    MID_ROW = TOP_ROW + 1
    BOTTOM_ROW = MID_ROW + 1
    CENTER = 4
    COUNT = 9


def construct_maze():
    tau = np.zeros((States.COUNT, Action.COUNT, States.COUNT))
    tau[States.LEFT_COL, Action.RIGHT, States.MID_COL] = 0.8
    tau[States.LEFT_COL, Action.RIGHT, States.LEFT_COL] = 0.2
    pass


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