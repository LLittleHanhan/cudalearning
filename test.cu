#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;

void initval(int s[], int n) {
    for (int i = 0; i < n; i++) {
        s[i] = i;
    }
}

__global__ void sumArrayOnGpu(int* ga, int* gb, int* gres) {
    int i = threadIdx.x;
    gres[i] = ga[i] + gb[i];
}

int main() {
    int dev = 0;
    // cout << cudaGetDeviceCount();
    cudaSetDevice(dev);
    int n = 32;
    int nbyte = sizeof(int) * n;

    int* a = new int[n];
    int* b = new int[n];
    int* res = new int[n];
    initval(a, n);
    initval(b, n);

    int *ga, *gb, *gres;
    cudaMalloc(&ga, nbyte);
    cudaMalloc(&gb, nbyte);
    cudaMalloc(&gres, nbyte);

    cudaMemcpy(ga, a, nbyte, cudaMemcpyHostToDevice);
    cudaMemcpy(gb, b, nbyte, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(1);

    sumArrayOnGpu<<<grid, block>>>(ga, gb, gres);
    cudaMemcpy(res, gres, nbyte, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        cout << res[i];
    }

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gres);
}