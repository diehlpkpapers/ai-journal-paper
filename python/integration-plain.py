import numpy as np

# define the function sin(x)
def f(x):
    return np.sin(x)

# define the limits of integration
a = -np.pi
b = 2*np.pi/3

# define the number of intervals for the Riemann sum
n = 1000

# compute the width of each interval
dx = (b - a) / n

# initialize the sum
sum = 0

# compute the Riemann sum
for i in range(n):
    x = a + i * dx
    sum += f(x) * dx

print("The area between -π and 2/3π for sin(x) is: ", sum)
