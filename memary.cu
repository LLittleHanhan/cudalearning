#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

__device__ int ga = 8;
__device__ int gb;

__global__ void var() {
    int a = 0;          // 寄存器
    int s[1000] = {0};  // 本地内存
    printf("%p %p", &a, &s);
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
    int test = 100;
    cudaMemcpyToSymbol(gb, &test, sizeof(int));
    var<<<1, 8>>>();
}