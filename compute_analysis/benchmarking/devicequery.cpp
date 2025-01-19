/*
Some of the code is adopted from device query benchmark
from CUDA SDK
*/

#include <cuda_runtime.h>
// #include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
            static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

     // memory
    char msg[256];
    snprintf(msg, sizeof(msg),
             "  Global memory size                        : %.0f GB\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1073741824.0f));
    printf("%s", msg);
    printf("  Memory Clock rate                           : %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width                            : %d bit\n",
           deviceProp.memoryBusWidth);
    return 0;
}
