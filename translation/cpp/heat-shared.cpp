#include <iostream>
#include <cmath>
#include <omp.h>

int main() {
    const int NX = 100;  // Number of grid points in x-direction
    const int NT = 1000;  // Number of time steps
    const double LX = 1.0;  // Length of the domain in x-direction
    const double T_FINAL = 0.1;  // Final time
    const double ALPHA = 0.1;  // Thermal diffusivity
    const double DX = LX / (NX - 1);  // Grid spacing in x-direction
    const double DT = T_FINAL / NT;  // Time step
    const double M_PI = 4 * atan(1);

    double* u = new double[NX];
    double* u_new = new double[NX];

    // Initialize the grid
    for (int i = 0; i < NX; i++) {
        u[i] = sin(M_PI * i * DX);  // Initial condition
    }

    // Time-stepping loop
    for (int j = 0; j < NT; j++) {
        #pragma omp parallel for
        for (int i = 1; i < NX - 1; i++) {
            u_new[i] = u[i] + ALPHA * DT / DX / DX * (u[i + 1] - 2 * u[i] + u[i - 1]);
        }

        // Update the grid
        for (int i = 0; i < NX; i++) {
            u[i] = u_new[i];
        }
    }

    // Output the final solution
    for (int i = 0; i < NX; i++) {
        std::cout << u[i] << std::endl;
    }

    delete[] u;
    delete[] u_new;

    return 0;
}
