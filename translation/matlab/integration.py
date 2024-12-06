import math
import numpy as np

# Define the limits of integration
a = -math.pi
b = 2/3 * math.pi

# Compute the area using the definite integral
area = np.trapz(np.sin(x), a, b)

# Print the result
print("Area: {:.2f}".format(area))
