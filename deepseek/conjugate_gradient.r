# Conjugate Gradient Solver
conjugate_gradient <- function(A, b, x0 = rep(0, length(b)), tol = 1e-10, max_iter = 100) {
  x <- x0
  r <- b - A %*% x
  p <- r
  rsold <- t(r) %*% r
  residuals <- numeric(max_iter + 1)
  residuals[1] <- sqrt(rsold)
  
  for (iter in 1:max_iter) {
    Ap <- A %*% p
    alpha <- as.numeric(rsold / (t(p) %*% Ap))
    x <- x + alpha * p
    r <- r - alpha * Ap
    rsnew <- t(r) %*% r
    residuals[iter + 1] <- sqrt(rsnew)
    
    # Check for convergence
    if (sqrt(rsnew) < tol) {
      residuals <- residuals[1:(iter + 1)]
      break
    }
    
    p <- r + as.numeric(rsnew / rsold) * p
    rsold <- rsnew
  }
  
  list(x = as.vector(x), iterations = iter, residuals = residuals)
}

# Define the system
A <- matrix(c(4, -1, 0, 
              -1, 4, -1, 
              0, -1, 4), nrow = 3, byrow = TRUE)
b <- c(1, 2, 3)

# Solve using our CG implementation
cg_result <- conjugate_gradient(A, b)

# Solve using R's solve function for validation
x_direct <- solve(A, b)

# Print results
cat("Our Conjugate Gradient Solution:\n")
print(cg_result$x)
cat("Number of iterations:", cg_result$iterations, "\n\n")

cat("Direct Solution (solve(A, b)):\n")
print(x_direct)
cat("\n")

# Compute error
error <- norm(cg_result$x - x_direct, type = "2")
cat("Error between CG and direct solution:", error, "\n")

# Plot convergence
plot(cg_result$residuals, type = "b", log = "y", 
     xlab = "Iteration", ylab = "Residual Norm (log scale)",
     main = "Convergence of Conjugate Gradient Method")
grid()
