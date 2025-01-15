# Load the parallel package
library(parallel)

# Define the heat equation function
heat_equation <- function(u, dx, dt, alpha, nx) {
  du_dt <- numeric(nx)  # Initialize du_dt as numeric
  
  for (i in 2:(nx-1)) {
    du_dt[i] <- alpha * (u[i+1] - 2 * u[i] + u[i-1]) / (dx^2)
  }
  
  return(du_dt)
}

# Define the Euler step function
euler_step <- function(u, du_dt, dt) {
  u_new <- u + dt * du_dt  # Compute the new u using Euler's method
  return(u_new)
}

# Define the parallel solver function
solve_heat_equation_parallel <- function(u0, dx, dt, alpha, t_end, num_workers) {
  nx <- length(u0)
  nt <- ceiling(t_end / dt)
  u <- u0
  
  # Use the parallel package for parallel processing
  cl <- makeCluster(num_workers)  # Create a cluster
  
  # Export the variables that do not change during iterations
  clusterExport(cl, list("dx", "alpha", "nx"))
  
  for (i in 1:nt) {
    # Parallelize the heat equation calculation
    du_dt <- parSapply(cl, 1:(nx-2), function(i) {
      alpha * (u[i+1] - 2 * u[i] + u[i-1]) / (dx^2)
    })
    
    # Ensure du_dt is numeric before proceeding
    du_dt <- as.numeric(du_dt)
    
    # Convert the result of parSapply into the full vector (including boundary conditions)
    du_dt_full <- c(0, du_dt, 0)  # Add boundary conditions
    
    # Perform Euler step to update the solution
    u <- euler_step(u, du_dt_full, dt)
  }
  
  stopCluster(cl)  # Stop the cluster
  
  return(u)
}

# Set up the problem parameters
L <- 1.0
nx <- 100
dx <- L / (nx - 1)
dt <- 0.001  # Set time step (dt)
t_end <- 0.1
alpha <- 0.1
num_workers <- 4

# Set up the initial condition
x <- seq(0, L, length.out = nx)  # Use seq() to create a sequence of x values
u0 <- sin(pi * x)

# Solve the heat equation in parallel
u <- solve_heat_equation_parallel(u0, dx, dt, alpha, t_end, num_workers)

# Plot the numerical solution
# plot(x, u, type = "l", col = "blue", xlab = "x", ylab = "u(x,t)", 
#      main = "Solution of the 1D Heat Equation")

# Define the exact solution
exact_solution <- function(x, t, alpha) {
  sin(pi * x) * exp(-alpha * pi^2 * t)
}

# Compute the exact solution at t_end
u_exact <- exact_solution(x, t_end, alpha)

# Add the exact solution to the plot
# lines(x, u_exact, col = "red")
plot(x, u_exact, type = "l", col = "blue", xlab = "x", ylab = "u(x,t)", 
     main = "Solution of the 1D Heat Equation")
legend("topright", legend = c("Exact Solution"), 
       col = c("blue"), lty = 1)

plot(x, u, type = "l", col = "red", xlab = "x", ylab = "u(x,t)", 
     main = "Solution of the 1D Heat Equation")

legend("topright", legend = c("Numerical Solution"), 
       col = c("red"), lty = 1)
