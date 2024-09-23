library(parallel)

# Define the heat equation function
heat_equation <- function(u, dx, dt, alpha) {
  nx <- length(u)
  du_dt <- numeric(nx)
  
  for (i in 2:(nx-1)) {
    du_dt[i] <- alpha * (u[i+1] - 2*u[i] + u[i-1]) / dx^2
  }
  
  return(du_dt)
}

# Define the Euler step function
euler_step <- function(u, du_dt, dt) {
  return(u + dt * du_dt)
}

# Define the parallel solver function
solve_heat_equation_parallel <- function(u0, dx, dt, alpha, t_end, num_cores) {
  nx <- length(u0)
  nt <- ceiling(t_end / dt)
  u <- u0
  
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  foreach(i = 1:nt, .combine = "+") %dopar% {
    du_dt <- heat_equation(u, dx, dt, alpha)
    u <- euler_step(u, du_dt, dt)
  }
  
  stopCluster(cl)
  return(u)
}

# Set up the problem parameters
L <- 1.0
nx <- 100
dx <- L / (nx - 1)
t_end <- 0.1
alpha <- 0.1
num_cores <- 4
dt <- 0.001  # Set a small time step size, was not defined in the ai-generated code and added manually 


# Set up the initial condition
x <- seq(0, L, by = dx)
u0 <- sin(pi * x)

# Solve the heat equation in parallel
u <- solve_heat_equation_parallel(u0, dx, dt, alpha, t_end, num_cores)

# Plot the solution
plot(x, u, type = "l", xlab = "x", ylab = "u(x,t)", main = "Solution of the 1D Heat Equation")

# Validate the solution
exact_solution <- function(x, t, alpha) {
  return(sin(pi * x) * exp(-alpha * pi^2 * t))
}

t <- t_end
u_exact <- exact_solution(x, t, alpha)

lines(x, u_exact, col = "red")

legend("topright", legend = c("Numerical Solution", "Exact Solution"), col = c("black", "red"), lty = 1)

