import numpy as np
from mpi4py import MPI

# Define the heat equation parameters
L = 1.0  # length of the domain
T = 1.0  # total time
alpha = 0.1  # thermal diffusivity
nx = 100  # number of spatial grid points
nt = 100  # number of time steps

# Define the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Calculate the spatial grid spacing
dx = L / (nx - 1)

# Calculate the time step
dt = T / nt

# Initialize the solution array
u = np.zeros((nx,))

# Initialize the boundary conditions
u[0] = 1.0
u[-1] = 0.0

# Distribute the grid points among processors
n_local = nx // size
start = rank * n_local
end = start + n_local

# Initialize the local solution array
u_local = np.zeros((n_local,))

# Communicate the boundary values
if rank == 0:
    comm.send(u[1], dest=1, tag=0)
if rank == size - 1:
    comm.send(u[-2], dest=size - 2, tag=0)
if rank > 0:
    u_local[0] = comm.recv(source=rank - 1, tag=0)
if rank < size - 1:
    u_local[-1] = comm.recv(source=rank + 1, tag=0)

# Time-stepping loop
for n in range(nt):
    # Exchange boundary values
    if rank > 0:
        comm.send(u_local[0], dest=rank - 1, tag=1)
    if rank < size - 1:
        comm.send(u_local[-1], dest=rank + 1, tag=1)
    if rank > 0:
        u_local[-1] = comm.recv(source=rank - 1, tag=1)
    if rank < size - 1:
        u_local[0] = comm.recv(source=rank + 1, tag=1)

    # Update the solution using Euler method
    for i in range(1, n_local - 1):
        u_local[i] = u_local[i] + alpha * dt / dx**2 * (u_local[i-1] - 2*u_local[i] + u_local[i+1])

# Gather the solution from all processors
u_global = np.zeros((nx,))
comm.Allgather(u_local, u_global)

# Validate the solution
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
u_exact = np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t[-1])
error = np.linalg.norm(u_global - u_exact)
print("Error:", error)

