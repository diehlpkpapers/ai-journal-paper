import numpy as np
import multiprocessing as mp

def exact_solution(x, t, alpha):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


def heat_equation(u, dx, dt, alpha):
    """
    Compute the right-hand side of the 1D heat equation using finite differences.

    Parameters:
    u (array): current solution
    dx (float): spatial grid spacing
    dt (float): time step
    alpha (float): thermal diffusivity

    Returns:
    du_dt (array): time derivative of the solution
    """
    nx = len(u)
    du_dt = np.zeros(nx)

    for i in range(1, nx-1):
        du_dt[i] = alpha * (u[i+1] - 2*u[i] + u[i-1]) / dx**2

    return du_dt

def euler_step(u, du_dt, dt):
    """
    Perform one time step using the Euler method.

    Parameters:
    u (array): current solution
    du_dt (array): time derivative of the solution
    dt (float): time step

    Returns:
    u_new (array): updated solution
    """
    return u + dt * du_dt

def solve_heat_equation(u0, dx, dt, alpha, t_end, num_procs):
    """
    Solve the 1D heat equation using parallel processing.

    Parameters:
    u0 (array): initial condition
    dx (float): spatial grid spacing
    dt (float): time step
    alpha (float): thermal diffusivity
    t_end (float): final time
    num_procs (int): number of processes to use

    Returns:
    u (array): solution at final time
    """
    nx = len(u0)
    nt = int(t_end / dt)
    u = u0.copy()

    with mp.Pool(processes=num_procs) as pool:
        for i in range(nt):
            du_dt = pool.apply(heat_equation, args=(u, dx, dt, alpha))
            u = euler_step(u, du_dt, dt)

    return u

if __name__ == '__main__':
    # Problem parameters
    L = 1.0
    nx = 100
    dx = L / (nx - 1)
    t_end = 0.1
    alpha = 0.1
    num_procs = 4
    dt = 0.1 / 1000 

    # Initial condition
    x = np.linspace(0, L, nx)
    u0 = np.sin(np.pi * x)

    # Solve the heat equation
    u = solve_heat_equation(u0, dx, dt, alpha, t_end, num_procs)

    # Plot the solution
    import matplotlib.pyplot as plt
    plt.plot(x, u, label="Computed")
    plt.plot(x, exact_solution(x, t_end, alpha), label="Analytic")
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Solution of the 1D Heat Equation')
    plt.legend()
    plt.grid()
    plt.show()

