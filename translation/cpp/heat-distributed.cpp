#include <mpi.h>
#include <iostream>
#include <cmath>

const int nx = 100;
const int nt = 100;
const double L = 1.0;
const double alpha = 0.1;
const double t_end = 0.1;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int ierr, num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double dx = L / (nx - 1);
    double dt = t_end / nt;

    double* u = new double[nx];
    double* du_dt = new double[nx];
    double* u_exact = new double[nx];

    if (rank == 0) {
        for (int i = 0; i < nx; i++) {
            u[i] = sin(M_PI * i * dx);
        }
    }

    MPI_Bcast(u, nx, MPI_DOUBLE, 0, MPI_COMM_WORLD, &ierr);

    for (int j = 0; j < nt; j++) {
        heatEquationMpi(u, du_dt, dx, alpha, nx);
        eulerStepMpi(u, du_dt, dt, nx);
    }

    if (rank == 0) {
        for (int i = 0; i < nx; i++) {
            u_exact[i] = sin(M_PI * i * dx) * exp(-alpha * M_PI * M_PI * t_end);
        }

        std::cout << "Numerical solution:" << std::endl;
        for (int i = 0; i < nx; i++) {
            std::cout << "u(" << i << ") = " << u[i] << std::endl;
        }

        std::cout << "Exact solution:" << std::endl;
        for (int i = 0; i < nx; i++) {
            std::cout << "u_exact(" << i << ") = " << u_exact[i] << std::endl;
        }
    }

    MPI_Finalize(&ierr);

    delete[] u;
    delete[] du_dt;
    delete[] u_exact;

    return 0;
}

void heatEquationMpi(double* u, double* du_dt, double dx, double alpha, int nx) {
    for (int i = 1; i < nx - 1; i++) {
        du_dt[i] = alpha * (u[i + 1] - 2 * u[i] + u[i - 1]) / dx / dx;
    }
}

void eulerStepMpi(double* u, double* du_dt, double dt, int nx) {
    for (int i = 0; i < nx; i++) {
        u[i] += dt * du_dt[i];
    }
}
