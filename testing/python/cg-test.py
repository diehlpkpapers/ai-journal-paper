import unittest
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

class TestConjugateGradient(unittest.TestCase):
    def test_conjugate_gradient(self):
        # Test the Conjugate Gradient solver
        A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
        b = np.array([1, 2, 3])
        x_cg = conjugate_gradient(A, b)
        x_numpy = np.linalg.solve(A, b)
        np.testing.assert_allclose(x_cg, x_numpy, atol=1e-5)

    def test_conjugate_gradient_tol(self):
        # Test the Conjugate Gradient solver with a different tolerance
        A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
        b = np.array([1, 2, 3])
        x_cg = conjugate_gradient(A, b, tol=1e-8)
        x_numpy = np.linalg.solve(A, b)
        np.testing.assert_allclose(x_cg, x_numpy, atol=1e-8)

    def test_conjugate_gradient_max_iter(self):
        # Test the Conjugate Gradient solver with a different maximum number of iterations
        A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
        b = np.array([1, 2, 3])
        x_cg = conjugate_gradient(A, b, max_iter=50)
        x_numpy = np.linalg.solve(A, b)
        np.testing.assert_allclose(x_cg, x_numpy, atol=1e-5)

    def test_conjugate_gradient_singular_matrix(self):
        # Test the Conjugate Gradient solver with a singular matrix
        A = np.array([[1, 1], [1, 1]])
        b = np.array([1, 1])
        with self.assertRaises(np.linalg.LinAlgError):
            conjugate_gradient(A, b)

if __name__ == '__main__':
    unittest.main()

