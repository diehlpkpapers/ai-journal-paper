/**
 * @file heatEquation1D.cpp
 * @brief Implementation of the 1D heat equation solver using parallel computing
 */

#include <iostream>
#include <cmath>

/**
 * @def NX
 * @brief Number of grid points in x-direction
 */
#define NX 100

/**
 * @def NT
 * @brief Number of time steps
 */
#define NT 1

/**
 * @def LX
 * @brief Length of the domain in x-direction
 */
#define LX 1.0

/**
 * @def T_FINAL
 * @brief Final time
 */
#define T_FINAL 0.1

/**
 * @def ALPHA
 * @brief Thermal diffusivity
 */
#define ALPHA 0.1

/**
 * @var DT
 * @brief Time step size
 */
double DT = T_FINAL / NT;

/**
 * @var DX
 * @brief Spatial step size
 */
double DX = LX / (NX - 1);

/**
 * @fn heat_equation_1d_parallel
 * @brief Solves the 1D heat equation using parallel computing
 * 
 * This function implements the 1D heat equation solver using parallel computing.
 * It initializes the grid, performs time-stepping using the finite difference method,
 * and outputs the final solution.
 */
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

/**
 * @fn main
 * @brief Main function
 * 
 * This function calls the heat_equation_1d_parallel function to solve the 1D heat equation.
 */
int main() {
  heat_equation_1d_parallel();
  return 0;
}

