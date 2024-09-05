import numpy as np
from scipy.integrate import quad

# define the function sin(x)
def f(x):
    return np.sin(x)

# define the limits of integration
a = -np.pi
b = 2*np.pi/3

# compute the integral using quad
result, error = quad(f, a, b)

print("The area between -π and 2/3π for sin(x) is: ", result)
