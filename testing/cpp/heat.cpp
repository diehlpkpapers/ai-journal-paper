#include <gtest/gtest.h>
#include "heat_equation_1d_parallel.h" // Include the original code file

const static size_t NX = 1000;
const static double DX = 0.1;
const static double NT = 100;
const static double DT = 0.001;
const static double ALPHA = 0.1;


void heat_equation_1d_parallel_init(double *u) {
  for (int i = 0; i < NX; i++) {
    u[i] = sin(M_PI * i * DX);  // Initial condition
  }
}


void heat_equation_1d_parallel_single_timestep(double *u, double *u_new) {
  #pragma omp parallel for private(i)
  for (int i = 1; i < NX - 1; i++) {
    u_new[i] = u[i] + ALPHA * DT / DX / DX * (u[i + 1] - 2 * u[i] + u[i - 1]);
  }
}

TEST(HeatEquation1DParallelTest, InitialCondition) {
  double *u = new double[NX];
  heat_equation_1d_parallel_init(u); // Initialize the grid
  for (int i = 0; i < NX; i++) {
    EXPECT_DOUBLE_EQ(u[i], sin(M_PI * i * DX));
  }
  delete[] u;
}

TEST(HeatEquation1DParallelTest, SingleTimeStep) {
  double *u = new double[NX];
  double *u_new = new double[NX];
  heat_equation_1d_parallel_init(u); // Initialize the grid

  heat_equation_1d_parallel_single_timestep(u, u_new);

  for (int i = 1; i < NX - 1; i++) {
    double expected = u[i] + ALPHA * DT / DX / DX * (u[i + 1] - 2 * u[i] + u[i - 1]);
    EXPECT_DOUBLE_EQ(u_new[i], expected);
  }

  delete[] u;
  delete[] u_new;
}

TEST(HeatEquation1DParallelTest, MultipleTimeSteps) {
  double *u = new double[NX];
  double *u_new = new double[NX];
  heat_equation_1d_parallel_init(u); // Initialize the grid

  for (int j = 0; j < NT; j++) {
    heat_equation_1d_parallel_single_timestep(u, u_new);
    // Update the grid
    for (int i = 1; i < NX - 1; i++) {
      u[i] = u_new[i];
    }
  }

  // Check the final solution
  for (int i = 0; i < NX; i++) {
    // You can add a more specific assertion here, e.g., checking against an analytical solution
    EXPECT_TRUE(std::isfinite(u[i]));
  }

  delete[] u;
  delete[] u_new;
}

