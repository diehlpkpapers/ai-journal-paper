#include <iostream>
#include <cmath>

#define NX 100  // Number of grid points in x-direction
#define NT 1  // Number of time steps
#define LX 1.0  // Length of the domain in x-direction
#define T_FINAL 0.1  // Final time
#define ALPHA 0.1  // Thermal diffusivity

double DT = T_FINAL / NT;
double DX = LX / (NX -1);


void heat_equation_1d_parallel() {
  int i, j;
  double *u, *u_new;
  u = new double[NX];
  u_new = new double[NX];
  u_new[NX-1] = 0;

  // Initialize the grid
  for (i = 0; i < NX; i++) {
    u[i] = sin(M_PI * i * DX);  // Initial condition
  }

  // Time-stepping loop
  for (j = 0; j < NT; j++) {
    #pragma omp parallel for private(i)
    for (i = 1; i < NX - 1; i++) {
      u_new[i] = u[i] + ALPHA * DT / DX / DX * (u[i + 1] - 2 * u[i] + u[i - 1]);
    }

    // Update the grid
    #pragma omp parallel for private(i)
    for (i = 1; i < NX - 1; i++) {
      u[i] = u_new[i];
    }
  }

  // Output the final solution
  for (i = 0; i < NX; i++) {
    std::cout  << u[i] << std::endl;
  }

  delete[] u;
  delete[] u_new;
}

int main() {
  heat_equation_1d_parallel();
  return 0;
}

