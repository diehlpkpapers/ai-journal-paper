# Define the limits of integration
a <- -pi
b <- 2/3 * pi

# Compute the area using the definite integral
area <- integrate(function(x) sin(x), a, b)

# Print the result
cat("Area:", area$value, "\n")
