import numpy as np
from numba import njit, prange

@njit(parallel=True)
def heat_equation_1d_parallel(NX, NT, LX, T_FINAL, ALPHA):
    DX = LX / (NX - 1)
    DT = T_FINAL / NT
    M_PI = 4 * np.arctan(1)

    u = np.zeros(NX)
    u_new = np.zeros(NX)

    # Initialize the grid
    for i in prange(NX):
        u[i] = np.sin(M_PI * i * DX)

    # Time-stepping loop
    for j in range(NT):
        for i in prange(1, NX - 1):
            u_new[i] = u[i] + ALPHA * DT / DX / DX * (u[i + 1] - 2 * u[i] + u[i - 1])

        # Update the grid
        u = u_new.copy()

    return u

NX = 100
NT = 1000
LX = 1.0
T_FINAL = 0.1
ALPHA = 0.1

u = heat_equation_1d_parallel(NX, NT, LX, T_FINAL, ALPHA)

# Output the final solution
for i in range(NX):
    print(u[i])
