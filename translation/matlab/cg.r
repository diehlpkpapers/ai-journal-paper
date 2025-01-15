# Define the linear equation system
A <- matrix(c(4, -1, 0, -1, 4, -1, 0, -1, 4), nrow = 3, byrow = TRUE)
b <- c(1, 2, 3)

# Define the tolerance and maximum number of iterations
tol <- 1e-6
max_iter <- 100

# Initialize the solution vector
x <- rep(0, length(b))

# Conjugate Gradient solver
r <- b - A %*% x
p <- r
rho <- sum(r * r)
for (k in 1:max_iter) {
  Ap <- A %*% p
  alpha <- rho / sum(p * Ap)
  x <- x + alpha * p
  r <- r - alpha * Ap
  rho_new <- sum(r * r)
  if (sqrt(rho_new) < tol) {
    break
  }
  beta <- rho_new / rho
  p <- r + beta * p
  rho <- rho_new
}

# Validate the solution
residual <- sqrt(sum((b - A %*% x)^2))
cat("Residual norm:", residual, "\n")
if (residual < tol) {
  cat("Solution validated!\n")
} else {
  cat("Solution not validated!\n")
}

# Compare with built-in solver (optional)
x_builtin <- solve(A, b)
cat("Built-in solver solution: \n")
print(x_builtin)
cat("CG solver solution: \n")
print(x)
