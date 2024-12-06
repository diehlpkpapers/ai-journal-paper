import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def heat_equation(u, dx, dt, alpha):
    nx = len(u)
    du_dt = np.zeros(nx)
    for i in range(1, nx-1):
        du_dt[i] = alpha * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
    return du_dt

def euler_step(u, du_dt, dt):
    return u + dt * du_dt

def solve_heat_equation_parallel(u0, dx, dt, alpha, t_end, num_workers):
    nx = len(u0)
    nt = int(np.ceil(t_end / dt))
    u = u0.copy()
    
    with Pool(processes=num_workers) as pool:
        for i in range(nt):
            du_dt = heat_equation(u, dx, dt, alpha)
            u = euler_step(u, du_dt, dt)
    
    return u

# Set up the problem parameters
L = 1.0
nx = 100
dx = L / (nx - 1)
t_end = 0.1
alpha = 0.1
num_workers = 4

# Set up the initial condition
x = np.arange(0, L+dx, dx)
u0 = np.sin(np.pi * x)

# Solve the heat equation in parallel
u = solve_heat_equation_parallel(u0, dx, dt, alpha, t_end, num_workers)

# Plot the solution
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Solution of the 1D Heat Equation')

# Validate the solution
def exact_solution(x, t, alpha):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

t = t_end
u_exact = exact_solution(x, t, alpha)

plt.hold(True)
plt.plot(x, u_exact, 'r')
plt.legend(['Numerical Solution', 'Exact Solution'])
plt.show()
