import numpy as np

def conjugate_gradient(A, b, tol=1e-5, max_iter=100):
    """
    Conjugate Gradient solver

    Parameters:
    A (numpy array): matrix
    b (numpy array): right-hand side vector
    tol (float): tolerance for convergence
    max_iter (int): maximum number of iterations

    Returns:
    x (numpy array): solution vector
    """
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x)
    p = r.copy()
    rho = np.dot(r, r)

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rho / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rho_new = np.dot(r, r)
        beta = rho_new / rho
        p = r + beta * p
        rho = rho_new

        if np.linalg.norm(r) < tol:
            break

    return x

# Define the matrix A and the right-hand side vector b
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
b = np.array([1, 2, 3])

# Solve the linear equation system using the Conjugate Gradient solver
x_cg = conjugate_gradient(A, b)

print("Solution using Conjugate Gradient solver:", x_cg)

# Validate the solution using the numpy linalg.solve function
x_numpy = np.linalg.solve(A, b)

print("Solution using numpy linalg.solve function:", x_numpy)

# Check if the solutions are equal
np.testing.assert_allclose(x_cg, x_numpy)
print("The solutions are equal!")

