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

__global__ void reduce(int* ga, int* gres, int m) {
    int tid = threadIdx.x;
    int* arr = blockIdx.x * blockDim.x * m + ga;
    for (int i = 1; i < m; i++) {
        arr[tid] += arr[tid + blockDim.x * i];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            arr[tid] += arr[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        gres[blockIdx.x] = arr[0];
}

void sumOneArray(int tx, int m) {
    int n = 1024 * 1024;
    int* a = new int[n];
    int* res = new int[n / (tx * m)];
    initval(a, n);

    int *ga, *gres;
    cudaMalloc(&ga, sizeof(int) * n);
    cudaMalloc(&gres, sizeof(int) * n / (tx * m));
    cudaMemcpy(ga, a, sizeof(int) * n, cudaMemcpyHostToDevice);

    dim3 block(tx);
    dim3 grid(n / (tx * m));
    cout << n / (tx * m) << " " << tx << " " << m << endl;
    reduce2<<<grid, block>>>(ga, gres, m);
    cudaMemcpy(res, gres, sizeof(int) * n / (tx * m), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < n / (tx * m); i++)
        sum += res[i];
    cout << sum << endl;

    cudaFree(ga);
    cudaFree(gres);
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
    int m = atoi(argv[2]);
    sumOneArray(tx, m);
}