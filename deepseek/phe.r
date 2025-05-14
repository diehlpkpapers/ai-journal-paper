library(parallel)
library(doParallel)
library(foreach)

solve_heat_equation_parallel <- function() {
  # Parameters
  alpha <- 0.1       # Thermal diffusivity
  L <- 1             # Length of the domain
  T_final <- 0.1     # Total simulation time
  n <- 100           # Number of spatial points
  nt <- 1000         # Number of time steps
  
  # Derived parameters
  h <- L/(n-1)       # Spatial step size
  dt <- T_final/nt   # Time step size
  x <- seq(0, L, length.out = n)  # Spatial grid
  
  # Initial condition
  u <- sin(pi * x)
  
  # Analytical solution
  analytical_solution <- function(t) {
    sin(pi * x) * exp(-alpha * pi^2 * t)
  }
  
  # Set up parallel backend
  n_cores <- detectCores()
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  
  # Pre-compute coefficient
  r <- alpha * dt / h^2
  
  # Time-stepping loop
  system.time({
    for (k in 1:nt) {
      u_new <- foreach(i = 1:n, .combine = 'c') %dopar% {
        if (i == 1 || i == n) {
          # Boundary conditions (Dirichlet)
          0
        } else {
          # Interior points (finite difference)
          u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
        }
      }
      u <- u_new
    }
  }) -> compute_time
  
  # Stop parallel cluster
  stopCluster(cl)
  
  # Get final solution
  u_final <- u
  
  # Analytical solution at final time
  u_exact <- analytical_solution(T_final)
  
  # Compute error
  max_error <- max(abs(u_final - u_exact))
  l2_error <- sqrt(sum((u_final - u_exact)^2) / n)
  
  # Print results
  cat("Parallel 1D Heat Equation Solver in R\n")
  cat("------------------------------------\n")
  cat(sprintf("Spatial points: %d, Time steps: %d\n", n, nt))
  cat(sprintf("Computation time: %.4f seconds\n", compute_time[3]))
  cat(sprintf("Maximum error at t=%.2f: %.6e\n", T_final, max_error))
  cat(sprintf("L2 error at t=%.2f: %.6e\n", T_final, l2_error))
  
  # Plot results
  par(mfrow = c(1, 2))
  
  # Solution plot
  plot(x, u_final, type = 'l', col = 'blue', lwd = 2,
       xlab = 'Position x', ylab = 'Temperature u(x)',
       main = 'Numerical vs Analytical Solution')
  lines(x, u_exact, col = 'red', lty = 2, lwd = 2)
  legend('topright', legend = c('Numerical', 'Analytical'),
         col = c('blue', 'red'), lty = c(1, 2), lwd = 2)
  grid()
  
  # Error plot
  plot(x, abs(u_final - u_exact), type = 'l', col = 'darkgreen', lwd = 2,
       xlab = 'Position x', ylab = 'Absolute Error',
       main = 'Error Distribution')
  grid()
}

# Run the solver
solve_heat_equation_parallel()
