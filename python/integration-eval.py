import numpy as np

# define the function sin(x)
def f(x):
    return np.sin(x)

# define the limits of integration
a = -np.pi
b = 2*np.pi/3

# compute the exact value of the integral
exact_result = -np.cos(b) + np.cos(a)

print("The exact area between -π and 2/3π for sin(x) is: ", exact_result)
