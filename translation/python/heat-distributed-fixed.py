import numpy as np
from mpi4py import MPI

def heat_equation_mpi(u, du_dt, dx, alpha, nx):
    for i in range(1, nx - 1):
        du_dt[i] = alpha * (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2

def euler_step_mpi(u, du_dt, dt, nx):
    for i in range(nx):
        u[i] += dt * du_dt[i]


nx = 100
nt = 1
L = 1.0
alpha = 0.1
t_end = 0.1

comm = MPI.COMM_WORLD
rank = comm.rank
num_procs = comm.size

dx = L / (nx - 1)
dt = t_end / nt

u = np.zeros(nx)
du_dt = np.zeros(nx)
u_exact = np.zeros(nx)

if rank == 0:
    for i in range(nx):
        u[i] = np.sin(np.pi * i * dx)

comm.Bcast(u, root=0)

for j in range(nt):
    heat_equation_mpi(u, du_dt, dx, alpha, nx)
    euler_step_mpi(u, du_dt, dt, nx)

if rank == 0:
    for i in range(nx):
        u_exact[i] = np.sin(np.pi * i * dx) * np.exp(-alpha * np.pi ** 2 * t_end)

    print("Numerical solution:")
    for i in range(nx):
        print(f"u({i}) = {u[i]}")

    print("Exact solution:")
    for i in range(nx):
        print(f"u_exact({i}) = {u_exact[i]}")

