import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Входные данные
x = np.array([0.5, -0.2])
y = 1.0

# Начальные веса и смещения
W1 = np.array([[0.1, 0.2],
               [0.3, -0.4]])
b1 = np.array([0.05, -0.1])

W2 = np.array([[0.6, -0.5]])
b2 = np.array([0.2])

# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------
z1 = W1 @ x + b1
a1 = sigmoid(z1)

z2 = W2 @ a1 + b2
a2 = sigmoid(z2)

loss = -(y * np.log(a2) + (1 - y) * np.log(1 - a2))

print("z1 =", z1)
print("a1 =", a1)
print("z2 =", z2)
print("a2 =", a2)
print("loss =", loss)

# ------------------------------------------------------------
# Backward pass и обновление весов
# ------------------------------------------------------------
dz2 = a2 - y

dW2 = dz2.reshape(1, 1) @ a1.reshape(1, -1)
db2 = dz2

da1 = W2.T @ dz2.reshape(1, 1)
da1 = da1.ravel()

dz1 = da1 * a1 * (1 - a1)

dW1 = np.outer(dz1, x)
db1 = dz1

lr = 0.5

W2_new = W2 - lr * dW2
b2_new = b2 - lr * db2

W1_new = W1 - lr * dW1
b1_new = b1 - lr * db1

print("dz2 =", dz2)
print("dW2 =", dW2)
print("db2 =", db2)
print("dz1 =", dz1)
print("dW1 =", dW1)
print("db1 =", db1)

print("W2_new =", W2_new)
print("b2_new =", b2_new)
print("W1_new =", W1_new)
print("b1_new =", b1_new)