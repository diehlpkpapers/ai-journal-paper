import numpy as np

def matmul(A, p):
    return np.dot(A, p)

def dot_product(a, b):
    return np.dot(a, b)

def gaussian_elimination(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, :] -= factor * A[i, :]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

n = 3
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
b = np.array([1, 2, 3])
x_cg = np.zeros(n)
x_ge = np.zeros(n)
r = b.copy()
p = r.copy()
Ap = np.zeros(n)
tol = 1e-5
rho = np.dot(r, r)
max_iter = 100

for i in range(max_iter):
    Ap = matmul(A, p)
    alpha = rho / dot_product(p, Ap)
    x_cg += alpha * p
    r -= alpha * Ap
    rho_new = np.dot(r, r)
    if np.sqrt(rho_new) < tol:
        break
    beta = rho_new / rho
    p = r + beta * p
    rho = rho_new

print("Solution using Conjugate Gradient solver: ", x_cg)

x_ge = gaussian_elimination(A.copy(), b.copy())
print("Solution using Gaussian elimination method: ", x_ge)

if np.allclose(x_cg, x_ge, atol=tol):
    print("The solutions are equal!")
else:
    print("The solutions are not equal!")
