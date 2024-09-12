#include <iostream>
#include <cmath>
#include <mpi.h>

// Function to compute the right-hand side of the heat equation
void heatEquation(double *u, double *du_dt, double dx, double alpha, int nx) {
  for (int i = 1; i < nx-1; i++) {
    du_dt[i] = alpha * (u[i+1] - 2*u[i] + u[i-1]) / dx / dx;
  }
}

// Function to perform one time step using the Euler method
void eulerStep(double *u, double *du_dt, double dt, int nx) {
  for (int i = 0; i < nx; i++) {
    u[i] += dt * du_dt[i];
  }
}

int main() {
  // Initialize MPI
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Get the rank of this process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Set up the problem parameters
  int nx = 100;
  double L = 1.0;
  double dx = L / (nx - 1);
  double t_end = 0.1;
  double alpha = 0.1;
  double dt = t_end / 100;

  // Set up the initial condition
  double *u = new double[nx];
  for (int i = 0; i < nx; i++) {
    u[i] = sin(M_PI * i * dx);
  }

  // Set up the MPI data types
  MPI_Datatype mpi_double;
  MPI_Type_contiguous(nx, MPI_DOUBLE, &mpi_double);
  MPI_Type_commit(&mpi_double);

  // Solve the heat equation in parallel
  for (int i = 0; i < 100; i++) {
    double *du_dt = new double[nx];
    heatEquation(u, du_dt, dx, alpha, nx);
    eulerStep(u, du_dt, dt, nx);
    delete[] du_dt;
  }

  // Validate the solution
  double *u_exact = new double[nx];
  for (int i = 0; i < nx; i++) {
    u_exact[i] = sin(M_PI * i * dx) * exp(-alpha * M_PI * M_PI * t_end);
  }

  // Print the results
  if (rank == 0) {
    for (int i = 0; i < nx; i++) {
      std::cout << "u[" << i << "] = " << u[i] << ", u_exact[" << i << "] = " << u_exact[i] << std::endl;
    }
  }

  // Clean up
  delete[] u;
  delete[] u_exact;
  MPI_Type_free(&mpi_double);
  MPI_Finalize();

  return 0;
}

