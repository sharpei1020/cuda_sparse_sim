#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

__global__ void test(
    const int* __restrict__ A,
    const int* __restrict__ B,
    int* __restrict__ C,
    uint* __restrict__ smid,
    int n
) {
    uint sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        C[tid] = A[tid] + B[tid];
    }
    if (threadIdx.x == 0)
        smid[blockIdx.x] = sm_id;
}

int main() {
    unsigned ARRAY_SIZE = 300 * 1024 * 256;

    int* A = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int* B = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int* C = (int*)malloc(ARRAY_SIZE * sizeof(int));

    for (int i = 0; i < ARRAY_SIZE; i++) {
        A[i] = i;
        B[i] = i;
    }

    int* A_g, *B_g, *C_g;
    cudaMalloc((void**)&A_g, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&B_g, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&C_g, ARRAY_SIZE * sizeof(int));

    cudaMemcpy(A_g, A, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_g, B, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(512, 1, 1);
    dim3 grid_dim(200, 1, 1);
    int* max_block = (int*)malloc(sizeof(int));
    

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(max_block, test, 512, 0);
    printf("Max blocks per SM: %d\n", *max_block);

    uint* smid_arr = (uint*)malloc(ARRAY_SIZE / 256 * sizeof(uint));
    uint* smid_g;
    cudaMalloc((void**)&smid_g, ARRAY_SIZE / 256 * sizeof(uint));
    test<<<grid_dim, block_dim>>>(A_g, B_g, C_g, smid_g, ARRAY_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(C, C_g, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(smid_arr, smid_g, ARRAY_SIZE / 256 * sizeof(uint), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 200; i++) {
        printf("(%d, %d) ", i, smid_arr[i]);
    }

    cudaFree(A_g);
    cudaFree(B_g);
    cudaFree(C_g);
    cudaFree(smid_g);

    free(A);
    free(B);
    free(C);
    free(smid_arr);

    return 0;
}

