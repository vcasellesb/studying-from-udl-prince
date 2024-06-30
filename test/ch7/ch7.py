from typing import List
import numpy as np

def f(x, theta: List[np.ndarray]):
    h1 = np.dot(theta[0][0], x) + theta[0][1]
    h2 = np.dot(theta[1][0], h1) + theta[1][1]
    h3 = np.dot(theta[2][0], h2) + theta[2][1]
    return np.dot(theta[3][0], h3) + theta[3][1]

in_dim = 100
hidden_dim = 1000
out_dim = 1
weights = []
biases = []

init_f = np.random.randn

for i in range(4):
    if i == 0:
        weights.append(np.random.randn(hidden_dim, in_dim))
        biases.append(np.random.randn(hidden_dim, 1))
    elif i == 3:
        weights.append(init_f(out_dim, hidden_dim))
        biases.append(init_f(out_dim, 1))
    else:
        weights.append(init_f(hidden_dim, hidden_dim))
        biases.append(init_f(hidden_dim, 1))


x = np.random.randn(in_dim, 1)
theta = [[w, b] for w, b in zip(weights, biases)]
print(f(x, theta).shape)