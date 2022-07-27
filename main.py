#%%
import numpy as np
import matplotlib.pylab as plt

#%%
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = AND(*x)
    print(f"{x} -> {y}")
#%%
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = OR(*x)
    print(f"{x} -> {y}")
#%%
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = NAND(*x)
    print(f"{x} -> {y}")
#%%
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = XOR(*x)
    print(f"{x} -> {y}")

#%%
def step_function(x):
    return np.array(x > 0, dtype=int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()