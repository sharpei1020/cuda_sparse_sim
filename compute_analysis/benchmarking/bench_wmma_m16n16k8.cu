#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#define REPEAT_ITERS 4096

#define M 16
#define N 16
#define K 8

using namespace nvcuda;


template <class T, class R>
__global__ void tensor_latency(uint64_t *startClk, uint64_t *stopClk, T *a,
                               T *b, R *res) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = (threadIdx.x >> 5);

    // register T result = 0;

    wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, R> c_frag;

    wmma::load_matrix_sync(a_frag, a + warp_id * M * K, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(b_frag, b + warp_id * K * N, 16);

    #pragma unroll
    for (unsigned t = 0; t < a_frag.num_elements; t++) {
        a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
    }

    #pragma unroll
    for (unsigned t = 0; t < b_frag.num_elements; t++) {
        b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
    }

    // synchronize all threads
    asm volatile("bar.sync 0;");

    // start timing
    uint64_t start = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

    for (int j = 0; j < REPEAT_ITERS; ++j) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // synchronize all threads
    asm volatile("bar.sync 0;");

    // stop timing
    uint64_t stop = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

    wmma::store_matrix_sync(res, c_frag, 16, wmma::mem_row_major);

    // write time and data back to memory
    startClk[gid] = start;
    stopClk[gid] = stop;
}

int main() {
    int BLOCK_NUM = 1;
    int TOTAL_THREADS;
    for (int i = 1; i < 16; i++) {
        TOTAL_THREADS = i * 32;
        uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
        uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));

        float * data1 = (float *)malloc(M * K * i * sizeof(float));
        float * data2 = (float *)malloc(K * N * i * sizeof(float));
        float * res = (float *)malloc(M * N * i * sizeof(float));

        uint64_t *startClk_g, *stopClk_g;
        float *data1_g, *data2_g, *res_g;

        for (int j = 0; j < M * N; j++) {
            if (j < M * K)
                data1[j] = (float)j;
            if (j < K * N)
                data2[j] = (float)j;
        }
        cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t));
        cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t));
        cudaMalloc(&data1_g, M * K * i * sizeof(float));
        cudaMalloc(&data2_g, K * N * i * sizeof(float));
        cudaMalloc(&res_g, M * N * i * sizeof(float));

        cudaMemcpy(data1_g, data1, M * K * i * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(data2_g, data2, K * N * i * sizeof(float), cudaMemcpyHostToDevice);

        tensor_latency<float, float><<<BLOCK_NUM, TOTAL_THREADS>>>(startClk_g, stopClk_g, data1_g, data2_g, res_g);

        cudaPeekAtLastError();

        cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);
        cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);

        uint64_t totalTime = *std::max_element(stopClk, stopClk + TOTAL_THREADS) - *std::min_element(startClk, startClk + TOTAL_THREADS);
        float wmma = (float)totalTime / (float)REPEAT_ITERS;
        std::cout << "wmma latency = " << wmma << "(clk)\n";
        std::cout << "Total Clk number = " << totalTime << "\n";
        std::cout << "Total number of threads = " << TOTAL_THREADS << "\n";
        cudaFree(startClk_g);
        cudaFree(stopClk_g);
        cudaFree(data1_g);
        cudaFree(data2_g);
        cudaFree(res_g);
        free(startClk);
        free(stopClk);
        free(data1);
        free(data2);
        free(res);
    }
    return 0;
}