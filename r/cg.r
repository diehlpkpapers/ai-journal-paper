# Define the matrix A and the right-hand side vector b
A <- matrix(c(4, -1, 0, -1, 4, -1, 0, -1, 4), nrow = 3, byrow = TRUE)
b <- c(1, 2, 3)

# Define the Conjugate Gradient solver function
conjugate_gradient <- function(A, b, tol = 1e-5, max_iter = 100) {
  n <- nrow(A)
  x <- rep(0, n)
  r <- b
  p <- r
  rho <- sum(r^2)
  
  for (i in 1:max_iter) {
    Ap <- A %*% p
    alpha <- rho / sum(p * Ap)
    x <- x + alpha * p
    r <- r - alpha * Ap
    rho_new <- sum(r^2)
    
    if (sqrt(rho_new) < tol) {
      break
    }
    
    beta <- rho_new / rho
    p <- r + beta * p
    rho <- rho_new
  }
  
  return(x)
}

# Solve the linear equation system using the Conjugate Gradient solver
x_cg <- conjugate_gradient(A, b)

# Print the solution using the Conjugate Gradient solver
print(paste("Solution using Conjugate Gradient solver: ", x_cg))

# Validate the solution using the solve function
x_solve <- solve(A, b)

# Print the solution using the solve function
print(paste("Solution using solve function: ", x_solve))

# Check if the solutions are equal
if (all(abs(x_cg - x_solve) < 1e-5)) {
  print("The solutions are equal!")
} else {
  print("The solutions are not equal!")
}

