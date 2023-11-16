#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>
using namespace std;

void initval(int s[], int n) {
    for (int i = 0; i < n; i++) {
        s[i] = 1;
    }
}

__global__ void sumTwoArrayOnGpu(int* ga, int* gb, int* gres) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    gres[i] = ga[i] + gb[i];
}

void sumTwoMatrix(int tx, int ty) {
    int l = 1024;
    int n = l * l;
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
    cudaMemcpy(gres, a, nbyte, cudaMemcpyHostToDevice);

    dim3 block(tx, ty);
    dim3 grid(l / tx, l / ty);
    cout << tx << " " << ty << " " << l / tx << " " << l / ty;
    sumTwoArrayOnGpu<<<grid, block>>>(ga, gb, gres);
    cudaMemcpy(res, gres, nbyte, cudaMemcpyDeviceToHost);

    cout << res[n - l] << endl;

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gres);
    delete[] a;
    delete[] b;
    delete[] res;
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    cout << prop.name << endl;
    cout << "num of mutipro:" << prop.multiProcessorCount << endl;
    cout << "max num threads of per block:" << prop.maxThreadsPerBlock << endl;
    cout << "max num threads of per mutipro:" << prop.maxThreadsPerMultiProcessor << endl;
    cout << "max num warps of per mutipro:" << prop.maxThreadsPerMultiProcessor / 32 << endl;

    int tx = atoi(argv[1]);
    int ty = atoi(argv[2]);
    sumTwoMatrix(tx, ty);
}