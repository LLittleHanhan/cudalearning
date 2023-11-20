#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
    cudaSharedMemConfig conf;
    cudaDeviceGetSharedMemConfig(&conf);
    cout << conf;
}