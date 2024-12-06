# Define the heat equation function
heat_equation <- function(u, dx, dt, alpha) {
  nx <- length(u)
  du_dt <- rep(0, nx)
  
  for (i in 2:(nx-1)) {
    du_dt[i] <- alpha * (u[i+1] - 2*u[i] + u[i-1]) / dx^2
  }
  return(du_dt)
}

# Define the Euler step function
euler_step <- function(u, du_dt, dt) {
  return(u +
