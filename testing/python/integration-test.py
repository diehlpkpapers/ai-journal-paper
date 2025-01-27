import unittest
import numpy as np

def f(x):
    return np.sin(x)

class TestRiemannSum(unittest.TestCase):
    def test_f(self):
        # Test the sin(x) function
        self.assertAlmostEqual(np.sin(np.pi/2), f(np.pi/2), places=5)
        self.assertAlmostEqual(np.sin(0), f(0), places=5)

    def test_riemann_sum(self):
        # Test the Riemann sum computation
        a = -np.pi
        b = 2*np.pi/3
        n = 1000
        dx = (b - a) / n
        sum = 0
        for i in range(n):
            x = a + i * dx
            sum += f(x) * dx
        expected_area = 1.230959  # approximate value of the integral
        self.assertAlmostEqual(sum, expected_area, places=3)

if __name__ == '__main__':
    unittest.main()

