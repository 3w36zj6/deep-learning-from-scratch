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
#%%
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


#%%
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(Y)

#%%
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print(A2, Z2)

#%%
def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


#%%
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


#%%
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


print(numerical_gradient(lambda x: x[0] ** 2 + x[1] ** 2, np.array([3.0, 4.0])))
#%%
def gradient_descent(f, init_x, lr=0.01, step_num=1000):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


print(gradient_descent(lambda x: x[0] ** 2 + x[1] ** 2, init_x=np.array([-3.0, 4.0])))

#%%
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
