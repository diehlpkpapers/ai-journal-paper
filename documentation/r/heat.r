# Load the parallel library
library(parallel)

# Define the heat equation function
# Computes the derivative of the solution u with respect to time t using the heat equation
heat_equation <- function(u, dx, dt, alpha) {
  # Get the number of grid points
  nx <- length(u)
  # Initialize the derivative vector
  du_dt <- numeric(nx)
  
  # Compute the derivative using the heat equation formula
  for (i in 2:(nx-1)) {
    du_dt[i] <- alpha * (u[i+1] - 2*u[i] + u[i-1]) / dx^2
  }
  
  # Return the derivative vector
  return(du_dt)
}

# Define the Euler step function
# Updates the solution u using the Euler method
euler_step <- function(u, du_dt, dt) {
  # Update the solution using the Euler formula
  return(u + dt * du_dt)
}

# Define the parallel solver function
# Solves the 1D heat equation using the Euler method with parallel processing
solve_heat_equation_parallel <- function(u0, dx, dt, alpha, t_end, num_cores) {
  # Get the number of grid points
  nx <- length(u0)
  # Compute the number of time steps
  nt <- ceiling(t_end / dt)
  # Initialize the solution vector
  u <- u0
  
  # Create a parallel cluster with num|

