import numpy as np
import matplotlib.pyplot as plt

n = 100000
d = 1000
p = 0.25
rows = 5

data = np.random.binomial(1, p, (n, d))

for epsilon in [0.5, 0.25, 0.1, 0.01, 0.001]:
    estimate = data[:rows, :]

