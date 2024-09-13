// Write a CUDA code to compute the area between -pi and 2pi/3 for sin(x) and validate it.
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void integrateSinKernel(float *d_area, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x, h, sum = 0.0f;

    if (idx < N) {
        h = (2.0f * M_PI / 3.0f - (-M_PI)) / N;
        x = -M_PI + idx * h;
        sum = 0.5f * (sin(x) + sin(x + h));
        for (int i = 1; i < N; i++) {
            x += h;
            sum += sin(x);
        }
        sum *= h;
        atomicAdd(d_area, sum);
    }
}

int main() {
    int N = 1000000; // number of intervals
    float h, area;
    float *d_area;

    // allocate memory on device
    cudaMalloc((void **)&d_area, sizeof(float));

    // initialize area to 0
    cudaMemset(d_area, 0, sizeof(float));

    // launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    integrateSinKernel<<<numBlocks, blockSize>>>(d_area, N);

    // copy result from device to host
    cudaMemcpy(&area, d_area, sizeof(float), cudaMemcpyDeviceToHost);

    // validate result
    std::cout << "Approximated area: " << area << std::endl;
    std::cout << "Exact area: " << 2.0f - sqrt(3.0f) / 2.0f << std::endl;

    // free memory
    cudaFree(d_area);

    return 0;
}
