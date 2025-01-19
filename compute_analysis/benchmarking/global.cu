#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define L2_CACHE_SIZE (3 * 1024 * 1024)
#define THREADS 1024
#define BLOCKS 1
#define TOTAL_THREADS (THREADS * BLOCKS)
#define MEM_BITWIDTH 192
#define MEM_CLK_FREQUENCY 7501

__global__ void mem_discrete(float *A, uint32_t *startClk, uint32_t *stopClk, unsigned ARRAY_SIZE, unsigned stride)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float c = 0;
    int offset = TOTAL_THREADS * stride;
    // synchronize all threads
    asm volatile("bar.sync 0;");

    // start timing
    uint32_t start = 0;
    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    for (int i = idx; i < ARRAY_SIZE; i += offset) {
        c += A[i];
    }

    // synchronize all threads
    asm volatile("bar.sync 0;");
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
    // float sum = 0.f;
    // for (int i = 0; i < stride; i++) {
    //     sum += c[i];
    // }
    A[idx] = c;

    // write time and data back to memory
    startClk[idx] = start;
    stopClk[idx] = stop;
}

__global__ void mem_continuous(float *A, float *B, uint32_t *startClk, uint32_t *stopClk, unsigned ARRAY_SIZE)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = TOTAL_THREADS;
     // synchronize all threads
    asm volatile("bar.sync 0;");

    // start timing
    uint32_t start = 0;
    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");
    _Pragma("unroll 64")
    for (unsigned i = idx; i < (ARRAY_SIZE / 4); i+=stride) {
        float4 c = reinterpret_cast<float4 *>(A)[i];
        reinterpret_cast<float4 *>(B)[i] = c;
    }
    // synchronize all threads
    asm volatile("bar.sync 0;");
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
    
    // write time and data back to memory
    startClk[idx] = start;
    stopClk[idx] = stop;
}

int main() {
    // Array size has to exceed L2 size to avoid L2 cache residence
    unsigned ARRAY_SIZE = (L2_CACHE_SIZE / sizeof(float)) * 64;

    uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
    uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));

    float *A = (float *)malloc(ARRAY_SIZE * sizeof(float));

    uint32_t *startClk_g;
    uint32_t *stopClk_g;
    float *A_g, *B_g;

    for (int i = 0; i < ARRAY_SIZE; i++) {
        A[i] = (float)i;
    }

    cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t));
    cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t));
    cudaMalloc(&A_g, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&B_g,ARRAY_SIZE * sizeof(float));

    cudaMemcpy(A_g, A, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int gpu_freq = deviceProp.clockRate;
    printf("GPU Frequency = %u KHz\n", gpu_freq);
    for (int i = 1; i < 2; i++) {
        int stride = 32;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        mem_discrete<<<1, TOTAL_THREADS>>>(A_g, startClk_g, stopClk_g, ARRAY_SIZE, stride);
        // mem_continuous<<<BLOCKS, THREADS>>>(A_g, B_g,
        //         startClk_g, stopClk_g, ARRAY_SIZE); 

        cudaGetDeviceProperties(&deviceProp, dev);
        gpu_freq = deviceProp.clockRate;
        printf("GPU Frequency = %u KHz\n", gpu_freq);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        cudaPeekAtLastError();
        cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
        cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);

        // DDR: double-data-rate 
        float max_bw = (float)MEM_BITWIDTH * MEM_CLK_FREQUENCY * 2 / 1e3 / 8; // in GB/s
        unsigned N = ARRAY_SIZE / stride * sizeof(float); // in bytes
        // unsigned N = 2 * ARRAY_SIZE * sizeof(float); // in bytes

        uint32_t total_time = *std::max_element(stopClk, stopClk + TOTAL_THREADS) - *std::min_element(startClk, startClk + TOTAL_THREADS);
        float mem_bw = (float)(N) / (float)(total_time);
        float average_clock = (float)total_time / (float)(N/TOTAL_THREADS);
        printf("stride %d, Load %d Bytes, using %f ms, average %f clock\n", stride, N, elapsedTime, average_clock);
        printf("Mem BW= %f (Byte/Clk) = %f (GB/sec)\n", mem_bw, mem_bw * gpu_freq / 1e6);
        printf("Mem BW= %f (GB/sec)\n", (float)N / elapsedTime / 1e6);
        printf("Max Theortical Mem BW= %f (GB/sec)\n", max_bw);
        printf("Mem Efficiency = %f %%\n", (mem_bw * gpu_freq / 1e6 / max_bw) * 100);
        printf("/////////////////////////////////////\n");
    }
}
