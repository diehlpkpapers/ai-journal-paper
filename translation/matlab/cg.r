import numpy as np

# Define the linear equation system
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])  # matrix
b = np.array([1, 2, 3])  # right-hand side vector

# Define the tolerance and maximum number of iterations
tol = 1e-6
max_iter = 100

# Initialize the solution vector
x = np.zeros_like(b)

# Conjugate Gradient solver
r = b - A @ x  # initial residual
p = r.copy()  # initial search direction
rho = np.dot(r, r)  # initial residual norm
for k in range(max_iter):
    Ap = A @ p  # matrix-vector product
    alpha = rho / np.dot(p, Ap)  # step size
    x += alpha * p  # update solution
    r -= alpha * Ap  # update residual
    rho_new = np.dot(r, r)  # new residual norm
    if np.sqrt(rho_new) < tol:
        break
    beta = rho_new / rho  # update beta
    p = r + beta * p  # update search direction
    rho = rho_new  # update residual norm

# Validate the solution
residual = np.linalg.norm(b - A @ x)
print(f'Residual norm: {residual:.2e}')
if residual < tol:
    print('Solution validated!')
else:
    print('Solution not validated!')

# Compare with built-in solver (optional)
x_builtin = np.linalg.solve(A, b)
print('Built-in solver solution:')
print(x_builtin)
print('CG solver solution:')
print(x)
