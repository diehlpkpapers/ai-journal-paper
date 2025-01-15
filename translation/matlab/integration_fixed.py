import math
import numpy as np

# Define the limits of integration
a = -math.pi
b = 2/3 * math.pi

x = np.linspace(a, b, 100)

# Compute the area using the trapezoidal rule
area = np.trapz(np.sin(x), x)

# Print the result
print("Area: {:.2f}".format(area))
