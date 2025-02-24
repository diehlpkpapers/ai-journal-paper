# Define the matrix A and the right-hand side vector b
# A is a 3x3 matrix representing the coefficient matrix of the linear equation system
# b is a 3-element vector representing the right-hand side of the linear equation system
A <- matrix(c(4, -1, 0, -1, 4, -1, 0, -1, 4), nrow = 3, byrow = TRUE)
b <- c(1, 2, 3)

# Define the Conjugate Gradient solver function
# This function solves a linear equation system Ax = b using the Conjugate Gradient method
# Parameters:
#   A: the coefficient matrix of the linear equation system
#   b: the right-hand side vector of the linear equation system
#   tol: the tolerance for convergence (default: 1e-5)
#   max_iter: the maximum number of iterations (default: 100)
conjugate_gradient <- function(A, b, tol = 1e-5, max_iter = 100) {
  # Get the number of equations (rows of A)
  n <- nrow(A)
  
  # Initialize the solution vector x and the residual vector r
  x <- rep(0, n)
  r <- b
  p <- r
  rho <- sum(r^2)
  
  # Iterate until convergence or maximum number of iterations
  for (i in 1:max_iter) {
    # Compute the matrix-vector product Ap
    Ap <- A %*% p
    # Compute the step size alpha
    alpha <- rho / sum(p * Ap)
    # Update the solution vector x and the residual vector r
    x <- x + alpha * p
    r <- r - alpha * Ap
    # Compute the new residual norm rho_new
    rho_new <- sum(r^2)
    
    # Check for convergence
    if (sqrt(rho_new) < tol) {
      break
    }
    
    # Update the search direction p
    beta <- rho_new / rho
    p <- r + beta * p
    rho <- rho_new
  }
  
  # Return the solution vector x
  return(x)
}

# Solve the linear equation system using the Conjugate Gradient solver
# x_cg is the solution vector obtained using the Conjugate Gradient solver
x_cg <- conjugate_gradient(A, b)

# Print the solution using the Conjugate Gradient solver
print(paste("Solution using Conjugate Gradient solver: ", x_cg))

# Validate the solution using the solve function
# x_solve is the solution vector obtained using the solve function
x_solve <- solve(A, b)

|

