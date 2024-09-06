# Define the limits of integration
a <- -pi
b <- 2*pi/3

# Define the number of intervals
n <- 1000

# Create a vector of x values
x <- seq(a, b, length.out = n)

# Compute the function values
f <- sin(x)

# Compute the area using a Riemann sum
area_riemann <- sum(f * (b - a) / n)

# Print the result
cat("The area between -π and 2/3π for sin(x) is:", area_riemann, "\n")

# Compute the exact value of the integral
exact_result <- -cos(b) + cos(a)

# Print the exact result
cat("The exact area between -π and 2/3π for sin(x) is:", exact_result, "\n")
