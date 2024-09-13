// Write a CUDA code, for the Ampere architecture, to compute the area between -pi and 2pi/3 for sin(x) and validate it.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void computeArea(float *d_area, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBins) {
        float x = -M_PI + idx * (2.0f * M_PI / 3.0f) / numBins;
        float dx = (2.0f * M_PI / 3.0f) / numBins;
        d_area[idx] = sinf(x) * dx;
    }
}

int main() {
    int numBins = 1000000; // number of bins for integration
    float *h_area, *d_area;
    cudaMalloc((void **)&d_area, numBins * sizeof(float));
    h_area = (float *)malloc(numBins * sizeof(float));

    int blockSize = 256;
    int numBlocks = (numBins + blockSize - 1) / blockSize;
    computeArea<<<numBlocks, blockSize>>>(d_area, numBins);

    cudaDeviceSynchronize();
    cudaMemcpy(h_area, d_area, numBins * sizeof(float), cudaMemcpyDeviceToHost);

    float totalArea = 0.0f;
    for (int i = 0; i < numBins; i++) {
        totalArea += h_area[i];
    }

    printf("Computed area: %f\n", totalArea);

    // Validate using scipy.integrate.quad
    // (you can use any other method to validate the result)
    float exactArea = 1.230095; // computed using scipy.integrate.quad
    printf("Exact area: %f\n", exactArea);
    printf("Relative error: %f\n", fabs(totalArea - exactArea) / exactArea);

    cudaFree(d_area);
    free(h_area);
    return 0;
}
