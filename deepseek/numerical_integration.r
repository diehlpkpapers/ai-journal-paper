# Function to compute the exact integral of sin(x) from a to b
exact_integral <- function(a, b) {
  return(-cos(b) + cos(a))
}

# Function to compute Riemann sum approximation of sin(x) from a to b
riemann_sum <- function(a, b, n = 1000000) {
  dx <- (b - a) / n  # Width of each rectangle
  x <- seq(a + dx/2, b - dx/2, length.out = n)  # Midpoints of all intervals
  sum(sin(x) * dx)  # Sum of areas of all rectangles
}

# Main computation
a <- -pi
b <- (2/3) * pi
n <- 1000000  # Number of rectangles

# Compute Riemann sum approximation
approx_area <- riemann_sum(a, b, n)

# Compute exact integral
exact_area <- exact_integral(a, b)

# Calculate errors
abs_error <- abs(approx_area - exact_area)
rel_error <- (abs_error / abs(exact_area)) * 100

# Print results
cat(sprintf("Calculating area of sin(x) from %.6f to %.6f\n\n", a, b))
cat(sprintf("Riemann sum approximation (%,d rectangles): %.12f\n", n, approx_area))
cat(sprintf("Exact integral value: %.12f\n", exact_area))
cat(sprintf("Absolute error: %.12f\n", abs_error))
cat(sprintf("Relative error: %.6f %%\n", rel_error))

# Visualization (optional)
x_vals <- seq(a, b, length.out = 1000)
y_vals <- sin(x_vals)
plot(x_vals, y_vals, type = "l", col = "blue", lwd = 2,
     main = expression(paste("Area under ", sin(x), " from -", pi, " to ", 2/3, pi)),
     xlab = "x", ylab = "sin(x)")
abline(h = 0, col = "gray")
polygon(c(x_vals, rev(x_vals)), c(y_vals, rep(0, length(y_vals))), 
        col = rgb(0, 0, 1, 0.3), border = NA)
