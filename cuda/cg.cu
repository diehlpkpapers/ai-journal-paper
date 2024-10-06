// Write a CUDA code to solve the a linear equation system using the conjugate gradient solver and validate it.
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Matrix-vector product kernel
__global__ void matVecKernel(float *A, float *x, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[idx * N + i] * x[i];
        }
        b[idx] = sum;
    }
}

// Conjugate Gradient solver kernel
__global__ void cgKernel(float *A, float *b, float *x, int N, int maxIter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float r[N]; // residual
        float p[N]; // search direction
        float Ap[N]; // matrix-vector product

        // Initialize residual and search direction
        r[idx] = b[idx];
        p[idx] = r[idx];

        for (int iter = 0; iter < maxIter; iter++) {
            // Matrix-vector product
            matVecKernel<<<N / 256, 256>>>(A, p, Ap, N);

            // Compute alpha
            float alpha = 0.0f;
            for (int i = 0; i < N; i++) {
                alpha += r[i] * p[i];
            }
            alpha /= dotProduct(p, Ap, N);

            // Update x and residual
            x[idx] += alpha * p[idx];
            r[idx] -= alpha * Ap[idx];

            // Compute beta
            float beta = 0.0f;
            for (int i = 0; i < N; i++) {
                beta += r[i] * r[i];
            }
            beta /= dotProduct(p, Ap, N);

            // Update search direction
            p[idx] = r[idx] + beta * p[idx];
        }
    }
}

// Dot product kernel
__global__ void dotProductKernel(float *a, float *b, float *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, a[idx] * b[idx]);
    }
}

// Host function to solve the linear system
void conjugateGradient(float *A, float *b, float *x, int N, int maxIter) {
    // Allocate device memory
    float *d_A, *d_b, *d_x;
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_x, N * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, N * sizeof(float));

    // Launch CG kernel
    cgKernel<<<N / 256, 256>>>(d_A, d_b, d_x, N, maxIter);

    // Copy result back to host
    cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
}

// Validate the result
void validate(float *A, float *b, float *x, int N) {
    float residual[N];
    for (int i = 0; i < N; i++) {
        residual[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            residual[i] += A[i * N + j] * x[j];
        }
        residual[i] -= b[i];
    }

    float maxResidual = 0.0f;
    for (int i = 0; i < N; i++) {
        maxResidual = fmaxf(maxResidual, fabsf(residual[i]));
    }

    std::cout << "Maximum residual: " << maxResidual << std::endl;
}

int main() {
    int N = 1024; // size of the linear system
    int maxIter = 100; // maximum number of iterations

    // Allocate host memory
    float *A, *b, *x;
    A = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    x = (float *)malloc(N * sizeof(float));

    // Initialize matrix A and vector b
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0f : 1.0f;
        }
        b[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    // Initialize solution vector x
    for (int i = 0; i < N; i++) {
        x[i] = 0.0f;
    }

    // Solve the linear system using Conjugate Gradient
    conjugateGradient(A, b, x, N, maxIter);

    // Validate the result
    validate(A, b, x, N);

    // Free host memory
    free(A);
    free(b);
    free(x);

    return 0;
}

