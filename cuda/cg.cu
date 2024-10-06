// Write a CUDA code to solve the a linear equation system using the conjugate gradient solver and validate it.
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Kernel function to perform matrix-vector multiplication
__global__ void matVecMulKernel(int n, float *A, float *x, float *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[idx * n + i] * x[i];
        }
        b[idx] = sum;
    }
}

// Kernel function to perform dot product
__global__ void dotProductKernel(int n, float *a, float *b, float *result) {
    __shared__ float temp;
    temp = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp += a[idx] * b[idx];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(result, temp);
    }
}

// Kernel function to perform vector addition
__global__ void vecAddKernel(int n, float *a, float *b, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// Kernel function to perform scalar multiplication
__global__ void scalarMultKernel(int n, float alpha, float *x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= alpha;
    }
}

// Conjugate Gradient solver
void conjugateGradient(int n, float *A, float *b, float *x, int maxIter, float tol) {
    float *r, *p, *Ap;
    cudaMalloc((void **)&r, n * sizeof(float));
    cudaMalloc((void **)&p, n * sizeof(float));
    cudaMalloc((void **)&Ap, n * sizeof(float));

    // Initialize r and p
    cudaMemcpy(r, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p, b, n * sizeof(float), cudaMemcpyHostToDevice);

    float rho, rhoOld, alpha, beta;
    int iter = 0;

    while (iter < maxIter) {
        // Compute Ap
        matVecMulKernel<<<(n + 255) / 256, 256>>>(n, A, p, Ap);

        // Compute rho
        float rhoResult;
        dotProductKernel<<<(n + 255) / 256, 256>>>(n, r, r, &rhoResult);
        cudaMemcpy(&rho, &rhoResult, sizeof(float), cudaMemcpyDeviceToHost);

        // Check for convergence
        if (sqrt(rho) < tol) break;

        // Compute alpha
        float alphaResult;
        dotProductKernel<<<(n + 255) / 256, 256>>>(n, p, Ap, &alphaResult);
        cudaMemcpy(&alpha, &alphaResult, sizeof(float), cudaMemcpyDeviceToHost);
        alpha = rho / alpha;

        // Update x and r
        scalarMultKernel<<<(n + 255) / 256, 256>>>(n, alpha, p);
        vecAddKernel<<<(n + 255) / 256, 256>>>(n, x, p, x);
        vecAddKernel<<<(n + 255) / 256, 256>>>(n, r, Ap, r);

        // Update p
        beta = rho / rhoOld;
        scalarMultKernel<<<(n + 255) / 256, 256>>>(n, beta, p);
        vecAddKernel<<<(n + 255) / 256, 256>>>(n, r, p, p);

        rhoOld = rho;
        iter++;
    }

    cudaFree(r);
    cudaFree(p);
    cudaFree(Ap);
}

int main() {
    int n = 1000; // size of the linear system
    float *A, *b, *x;
    A = (float *)malloc(n * n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));
    x = (float *)malloc(n * sizeof(float));

    // Initialize A and b
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i == j) ? 2 : 1;
        }
        b[i] = (i % 2 == 0) ? 1 : -1;
    }

    // Initialize x to zero
    for (int i = 0; i < n; i++) {
        x[i] = 0;
    }

    float *d_A, *d_b, *d_x;
    cudaMalloc((void **)&d_A, n * n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_x, n * sizeof(float));

    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    int maxIter = 1000;
    float tol = 1e-6;

    conjugateGradient(n, d_A, d_b, d_x, maxIter, tol);

    cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate the solution
    float residual = 0;
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        residual += pow(sum - b[i], 2);
    }
    residual = sqrt(residual);

    std::cout << "Residual: " << residual << std::endl;

    if (residual < tol) {
        std::cout << "Solution validated!" << std::endl;
    } else {
        std::cout << "Solution not validated!" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);

    free(A);
    free(b);
    free(x);

    return 0;
}

