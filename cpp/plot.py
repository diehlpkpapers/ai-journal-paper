import numpy as np
import matplotlib.pyplot as plt

alpha=0.1
t=0.1

def exact_solution(x, t, alpha):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

u_shared = np.loadtxt("result_heat_shared.txt",dtype=float)

print(u_shared)

x = np.linspace(0,1,len(u_shared))

plt.plot(x,u_shared,label="Computed (Shared)")
plt.plot(x, exact_solution(x, t, alpha), label="Analytic")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Solution of the 1D Heat Equation')
plt.legend()
plt.grid()
plt.show()

