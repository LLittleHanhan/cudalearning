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

__global__ void sumTwoArrayOnGpu(int* ga, int* gb, int* gres, int tx, int ty, int bx, int by) {
    int i = (blockIdx.y * bx + blockIdx.x) * tx * ty + threadIdx.y * tx + threadIdx.x;
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
    sumTwoArrayOnGpu<<<grid, block>>>(ga, gb, gres, tx, ty, l / tx, l / ty);
    cudaMemcpy(res, gres, nbyte, cudaMemcpyDeviceToHost);

    cout << res[n - l] << endl;

    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gres);
    delete[] a;
    delete[] b;
    delete[] res;
}

__global__ void reduce1(int* ga, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = n / 2; stride > 0; stride = (stride + 1) / 2) {
        if (i < stride) {
            ga[i] += ga[i + stride];
        }
        ga[stride] = 0;
        if (stride == 1)
            break;
    }
}

void sumArray(int tx) {
    int n = 1024;
    int nbyte = sizeof(int) * n;
    int* a = new int[n];
    int* res = new int;
    initval(a, n);

    int* ga;
    cudaMalloc(&ga, nbyte);
    dim3 block(tx);
    dim3 grid(n / tx);
    cout << tx << " " << n / tx;
    reduce1<<<grid, block>>>(ga, n);
    cudaMemcpy(res, ga, sizeof(int), cudaMemcpyDeviceToHost);

    cout << *res << endl;

    cudaFree(ga);
    delete[] a;
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
    // sumTwoMatrix(tx, ty);
    sumArray(tx);
}