// Write a parallel CUDA code to solve the one dimensional heat equation using a finite difference scheme for the discretization in space and the Euler method for time integration and validate it
#include <iostream>
#include <cuda_runtime.h>

#define NX 1024 // number of grid points in x-direction
#define NT 1000 // number of time steps
#define DX 0.01 // grid spacing in x-direction
#define DT 0.001 // time step
#define ALPHA 0.1 // thermal diffusivity

__global__ void heatEquationKernel(float *u, float *u_new, int nx) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nx) {
    int i = idx;
    float u_xx = (u[i+1] - 2*u[i] + u[i-1]) / (DX*DX);
    u_new[i] = u[i] + ALPHA * DT * u_xx;
  }
}

int main() {
  // Allocate memory on host and device
  float *u, *u_new;
  cudaMallocHost((void **)&u, NX * sizeof(float));
  cudaMallocHost((void **)&u_new, NX * sizeof(float));
  float *d_u, *d_u_new;
  cudaMalloc((void **)&d_u, NX * sizeof(float));
  cudaMalloc((void **)&d_u_new, NX * sizeof(float));

  // Initialize the solution array
  for (int i = 0; i < NX; i++) {
    u[i] = sin(M_PI * i * DX);
  }
  cudaMemcpy(d_u, u, NX * sizeof(float), cudaMemcpyHostToDevice);

  // Set up the kernel launch parameters
  int blockSize = 256;
  int numBlocks = (NX + blockSize - 1) / blockSize;

  // Time-stepping loop
  for (int n = 0; n < NT; n++) {
    heatEquationKernel<<<numBlocks, blockSize>>>(d_u, d_u_new, NX);
    cudaDeviceSynchronize();

    // Swap the solution arrays
    float *temp = d_u;
    d_u = d_u_new;
    d_u_new = temp;
  }

  // Copy the solution back to the host
  cudaMemcpy(u, d_u, NX * sizeof(float), cudaMemcpyDeviceToHost);

  // Validate the solution
  float L2_error = 0.0;
  for (int i = 0; i < NX; i++) {
    float exact = sin(M_PI * i * DX) * exp(-ALPHA * NT * DT * M_PI * M_PI);
    L2_error += (u[i] - exact) * (u[i] - exact);
  }
  L2_error = sqrt(L2_error / NX);
  std::cout << "L2 error: " << L2_error << std::endl;

  // Free memory
  cudaFreeHost(u);
  cudaFreeHost(u_new);
  cudaFree(d_u);
  cudaFree(d_u_new);

  return 0;
}
