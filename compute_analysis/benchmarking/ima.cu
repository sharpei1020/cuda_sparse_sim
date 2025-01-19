#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define REPEAT_ITERS 1024

template <class T>
__global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, T *data1,
                        T *data2, T *res) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    register T s1 = data1[gid];
    register T s2 = data2[gid];
    register T result[REPEAT_ITERS];

    // synchronize all threads
    asm volatile("bar.sync 0;");

    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    for (int j = 0; j < REPEAT_ITERS; ++j) {
        asm volatile("{\t\n"
                 "rem.s32 %0, %1, %2;\n\t"
                //  "div.s32 %0, %1, %2;\n\t"
                //  "div.s32 %0, %1, %2;\n\t"
                //  "div.s32 %0, %1, %2;\n\t"
                 "}"
                 : "=r"(result[j])
                 : "r"(s1), "r"(s2));
    }
    // synchronize all threads
    asm volatile("bar.sync 0;");

    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    // write time and data back to memory
    startClk[gid] = start;
    stopClk[gid] = stop;
    int sum = 0;
    for (int j = 0; j < REPEAT_ITERS; ++j) {
        sum += result[j];
    }
    res[gid] = sum;
}

int ipu_max_flops_and_latency(int THREADS_PER_BLOCK) {
    int BLOCKS_NUM = 1;
    int TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

    uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
    uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
    int *data1 = (int *)malloc(TOTAL_THREADS * sizeof(int));
    int *data2 = (int *)malloc(TOTAL_THREADS * sizeof(int));
    int *res = (int *)malloc(TOTAL_THREADS * sizeof(int));

    uint32_t *startClk_g;
    uint32_t *stopClk_g;
    int *data1_g;
    int *data2_g;
    int *res_g;

    for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
        data1[i] = (int)i;
        data2[i] = (int)2;
    }

    cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t));
    cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t));
    cudaMalloc(&data1_g, TOTAL_THREADS * sizeof(int));
    cudaMalloc(&data2_g, TOTAL_THREADS * sizeof(int));
    cudaMalloc(&res_g, TOTAL_THREADS * sizeof(int));

    cudaMemcpy(data1_g, data1, TOTAL_THREADS * sizeof(int),
                        cudaMemcpyHostToDevice);
    cudaMemcpy(data2_g, data2, TOTAL_THREADS * sizeof(int),
                        cudaMemcpyHostToDevice);

    max_flops<int><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g,
                                                        data1_g, data2_g, res_g);
    cudaPeekAtLastError();

    cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
    cudaMemcpy(res, res_g, TOTAL_THREADS * sizeof(int),
                        cudaMemcpyDeviceToHost);

    // printf("res[0] = %f\n", res[0]);
    // for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
    //     if (res[i] != res[0])
    //         printf("res[%d] = %f\n", i, res[i]);
    // }

    float latency;
    uint32_t max_stop_clk = *std::max_element(stopClk, stopClk + TOTAL_THREADS);
    uint32_t min_start_clk = *std::min_element(startClk, startClk + TOTAL_THREADS);
    latency = (float)(max_stop_clk - min_start_clk) / (float)(REPEAT_ITERS);
    printf("float-precision FPU ii = %f (clk)\n", latency);
    printf("Total Clk number = %u \n", max_stop_clk - min_start_clk);

    return 0;
}

int main() {
    for (int i = 1; i <= 1024; i *= 2) {
        printf("///////////////////////////////////////\n");
        printf("THREADS_PER_BLOCK = %d\n", i);
        ipu_max_flops_and_latency(i);
    }
    return 0;
}

