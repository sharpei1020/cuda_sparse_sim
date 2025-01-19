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
    register T s3 = 0.0001;
    register T s4 = s1 - s2;
    register T result1 = 0;
    register T result2 = 0;
    register T result3 = 0;
    register T result4 = 0;

    // synchronize all threads
    asm volatile("bar.sync 0;");

    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");
    // "fma.rn.f32 %0, %1, %2, %3;\n\t"
    // "fma.rn.f32 %0, %1, %2, %3;\n\t"
    // "fma.rn.f32 %0, %1, %2, %3;\n\t"

    for (int j = 0; j < REPEAT_ITERS; ++j) {
        asm volatile("{\t\n"
                    "fma.rn.f32 %0, %1, %2, %0;\n\t"
                    "}"
                    : "+f"(result1) : "f"(s1), "f"(s1));
        asm volatile("{\t\n"
                    "fma.rn.f32 %0, %1, %2, %0;\n\t"
                    "}"
                    : "+f"(result2) : "f"(s2), "f"(s2));
        asm volatile("{\t\n"
                    "fma.rn.f32 %0, %1, %2, %0;\n\t"
                    "}"
                    : "+f"(result3) : "f"(s3), "f"(s3));
        asm volatile("{\t\n"
                    "fma.rn.f32 %0, %1, %2, %0;\n\t"
                    "}"
                    : "+f"(result4) : "f"(s4), "f"(s4));
        asm volatile("{\t\n"
                    "fma.rn.f32 %0, %1, %2, %0;\n\t"
                    "}"
                    : "+f"(s1) : "f"(result1), "f"(result1));
        asm volatile("{\t\n"
                    "fma.rn.f32 %0, %1, %2, %0;\n\t"
                    "}"
                    : "+f"(s2) : "f"(result2), "f"(result2));
        asm volatile("{\t\n"
                    "fma.rn.f32 %0, %1, %2, %0;\n\t"
                    "}"
                    : "+f"(s3) : "f"(result3), "f"(result3));
        asm volatile("{\t\n"
                    "fma.rn.f32 %0, %1, %2, %0;\n\t"
                    "}"
                    : "+f"(s4) : "f"(result4), "f"(result4));
    }
    // synchronize all threads
    asm volatile("bar.sync 0;");

    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    // write time and data back to memory
    startClk[gid] = start;
    stopClk[gid] = stop;
    res[gid] = result1 + result2 + result3 + result4;
}

int fpu_max_flops_and_latency(int THREADS_PER_BLOCK) {
    int BLOCKS_NUM = 1;
    int TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

    uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
    uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
    float *data1 = (float *)malloc(TOTAL_THREADS * sizeof(float));
    float *data2 = (float *)malloc(TOTAL_THREADS * sizeof(float));
    float *res = (float *)malloc(TOTAL_THREADS * sizeof(float));

    uint32_t *startClk_g;
    uint32_t *stopClk_g;
    float *data1_g;
    float *data2_g;
    float *res_g;

    for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
        data1[i] = (float)0.0001;
        data2[i] = (float)0.0001;
    }

    cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t));
    cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t));
    cudaMalloc(&data1_g, TOTAL_THREADS * sizeof(float));
    cudaMalloc(&data2_g, TOTAL_THREADS * sizeof(float));
    cudaMalloc(&res_g, TOTAL_THREADS * sizeof(float));

    cudaMemcpy(data1_g, data1, TOTAL_THREADS * sizeof(float),
                        cudaMemcpyHostToDevice);
    cudaMemcpy(data2_g, data2, TOTAL_THREADS * sizeof(float),
                        cudaMemcpyHostToDevice);

    max_flops<float><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g,
                                                        data1_g, data2_g, res_g);
    cudaPeekAtLastError();

    cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
    cudaMemcpy(res, res_g, TOTAL_THREADS * sizeof(float),
                        cudaMemcpyDeviceToHost);

    printf("res[0] = %f\n", res[0]);
    for (uint32_t i = 0; i < TOTAL_THREADS; i++) {
        if (res[i] != res[0])
            printf("res[%d] = %f\n", i, res[i]);
    }

    float latency;
    uint32_t max_stop_clk = *std::max_element(stopClk, stopClk + TOTAL_THREADS);
    uint32_t min_start_clk = *std::min_element(startClk, startClk + TOTAL_THREADS);
    latency = (float)(max_stop_clk - min_start_clk) / (float)(REPEAT_ITERS * 8);
    printf("float-precision FPU ii = %f (clk)\n", latency);
    printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

    return 0;
}

int main() {
    for (int i = 1; i <= 1024; i *= 2) {
        printf("///////////////////////////////////////\n");
        printf("THREADS_PER_BLOCK = %d\n", i);
        fpu_max_flops_and_latency(i);
    }
    return 0;
}

