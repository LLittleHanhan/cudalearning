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

__global__ void sumArrayOnGpu(int* ga, int* gb, int* gres, int tx, int ty, int bx, int by) {
    int i = (blockIdx.y * by + blockIdx.x) * tx * ty + threadIdx.y * ty + threadIdx.x;
    gres[i] = ga[i] + gb[i];
}

void test(int tx, int ty) {
    int l = 32 * 1024;
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

    dim3 block(tx, ty);
    dim3 grid(l / tx, l / ty);

    sumArrayOnGpu<<<grid, block>>>(ga, gb, gres, tx, ty, l / tx, l / ty);
    cudaMemcpy(res, gres, nbyte, cudaMemcpyDeviceToHost);

    cout << res[10666] << endl;

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
    cout << tx << " " << ty;
    test(tx, ty);
}