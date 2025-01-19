#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda.h>
#include <mma.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace nvcuda;
#define BLK_H 16
#define BLK_W 8

enum DataSaveLayer {
  L1,
  L2,
  DRAM
};

typedef enum {
  TCGNN_Spmm,
  DTC_Spmm
} SpmmAlgo_t;

typedef enum {
  TCGNN_Sim,
  DTC_Sim
} simulationAlgo_t;

size_t calculate_data_size(int param_num, void* params[]) {
  size_t total_size = 0;
  for (int i = 0; i < param_num; i++) {
    size_t size = 0;
    cuMemGetAddressRange(NULL, &size, (CUdeviceptr)params[i]);
    total_size += size;
  }
  return total_size;
}


__global__ void spmm_time_simulation(uint64_t* duration, uint32_t* iters_offset, uint32_t* block_offset, 
    int* SparseCol_aToX, int* node_ptr, int block_high, int block_width, int dimTileNum, 
    int row_num, int numNodes, int hit_latency, int miss_latency_low, int miss_latency_high) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= row_num) return;
    // int block_iter_start = iters_offset[tid];
    // int block_iter_end = iters_offset[tid + 1];
    int iter_num = iters_offset[tid + 1] - iters_offset[tid];
    // int col_idx[8];
    int shared_latency = 29;
    int neigbor_num = (node_ptr[min((tid + 1) * block_high, numNodes)] - node_ptr[tid * block_high] + 8*32 - 1) / (8*32);
    // 未考虑线程数对读写的影响
    // read sparse neigbor_num * 3 * 200(miss)
    int cycle = (3 * miss_latency_high + 2 * shared_latency) * neigbor_num;
    // dense 200(miss) * 2 + 47(hit) * 2 cache_line 128
    for (int i = 0; i < iter_num; i++) { 
        // init shared mem 3 * 29, read sparse neigbor_num * 200(miss) 
        cycle += 3 * shared_latency + hit_latency * neigbor_num;
        // read dense (4 load/store unit)
        cycle += hit_latency * 2 + miss_latency_high * 4 + 8 * shared_latency;
        // int col_idx_start = block_offset[block_iter_start + i];
        // int col_idx_end = block_offset[block_iter_start + i + 1];
        // for (int j = col_idx_start; j < col_idx_end; j++) {
        //     col_idx[j - col_idx_start] = SparseCol_aToX[j];
        // }
        // smem load 8 * 29 matrix
        cycle += 8 * shared_latency;
        // matirx multiply
        cycle += 64 * ((dimTileNum + 3) / 4);
    }
    // save result 290 * 4
    cycle += miss_latency_high * 4 + hit_latency * 4;
    duration[tid] = (uint64_t)cycle;
    // iter_test[tid] = iter_num;
}

__global__ void mark_L1_hit(
   const uint32_t* __restrict__ block_offset,
   const int* __restrict__ cache_line_id,
   const int* __restrict__ node_ptr,
   const uint32_t* __restrict__ block_iter_nonzero_num,
   const uint32_t* __restrict__ block_iter_nonzero_num_offset,
   int* __restrict__ L1_hit_mark,
   int row_num,
   int block_high,
   int nodes_num
) {
  int block_id = blockIdx.x;
  if (block_id >= row_num) return;
  int cache_line_start = node_ptr[block_id * block_high];
  int cache_line_end = node_ptr[min((block_id + 1) * block_high, nodes_num)];
  extern __shared__ int cache_line_hit[];
  int cache_line_num = cache_line_end / 8 - cache_line_start / 8 + 1;
  for (int i = threadIdx.x; i < cache_line_num; i += blockDim.x) {
    cache_line_hit[i] = 0;
  }
  int block_start = block_offset[block_id];
  int block_end = block_offset[block_id + 1];
  for (int i = block_start; i < block_end; i++) {
    int nz_num = block_iter_nonzero_num[i];
    int ptr_off = block_iter_nonzero_num_offset[i];
    int is_block_iter_hit = 1;
    for (int tid = threadIdx.x; tid < nz_num; tid += blockDim.x) {
      int is_hit = atomicExch(&cache_line_hit[cache_line_id[ptr_off + tid] - cache_line_start / 8], 1);
      is_block_iter_hit &= is_hit;
    }
    atomicAnd(&L1_hit_mark[i], is_block_iter_hit);
  }
}

__global__ void get_nonzero_per_block(
  const int* __restrict__ node_ptr,
  int* __restrict__ nonzero_per_block,
  int row_num,
  int block_high,
  int numNodes
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= row_num) return;
  int cache_line_start = node_ptr[idx * block_high];
  int cache_line_end = node_ptr[min((idx + 1) * block_high, numNodes)];
  nonzero_per_block[idx] = cache_line_end / 8 - cache_line_start / 8 + 1;
}

__global__ void calculate_simulation_time(
    const int* __restrict__ L1_hit_mark,
    const uint32_t* __restrict__ block_offset,
    uint64_t* __restrict__ duratiion,
    DataSaveLayer save_layer,
    int row_num
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= row_num) return;
  int block_start = block_offset[tid];
  int block_end = block_offset[tid + 1];
  uint64_t cycle = 0;
  for (int i = block_start; i < block_end; i++) {
    cycle += 3*29;
    if (L1_hit_mark[i]) {
      cycle += (3 * 75 + 75);
    } else {
      if (save_layer < DRAM) {
        cycle += (3 * 250 + 75);
      } else {
        cycle += (3 * 550 + 75);
      }
    }
    cycle += (29 + 10) * 4 + 4 * 2;
    if (save_layer < DRAM) {
      cycle += (2 * 250 + 75 * 2);
    } else {
      cycle += (2 * 550 + 75 * 2);
    }
  }
  if (save_layer < DRAM) {
    cycle += (250 + 75) * 4;
  } else {
    cycle += (550 + 75) * 4;
  }
  duratiion[tid] = cycle;
}

void spmm_kernel_simulation(
    const uint64_t* duration,
    const int max_block_num_per_sm,
    const int sm_num,
    const int blocks_num,
    uint64_t &total_cycle
) {
  if (blocks_num <= max_block_num_per_sm * sm_num) {
    uint64_t* duration_tmp;
    cudaMalloc((void**)&duration_tmp, sizeof(uint64_t)*blocks_num);
    thrust::copy_n(thrust::device, duration, blocks_num, duration_tmp);
    thrust::sort(thrust::device, duration_tmp, duration_tmp + blocks_num);
    cudaMemcpy(&total_cycle, duration_tmp + blocks_num - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(duration_tmp);
  } else {
    uint64_t* duration_tmp;
    cudaMalloc((void**)&duration_tmp, sizeof(uint64_t)*max_block_num_per_sm*sm_num);
    thrust::copy_n(thrust::device, duration, max_block_num_per_sm*sm_num, duration_tmp);
    thrust::sort(thrust::device, duration_tmp, duration_tmp + max_block_num_per_sm*sm_num);
    int i = max_block_num_per_sm * sm_num;
    while (i < blocks_num) {
      uint64_t* tmp;
      int copy_len = min(max_block_num_per_sm*sm_num, blocks_num - i);
      cudaMalloc((void**)&tmp, sizeof(uint64_t)*copy_len);
      thrust::transform(thrust::device, duration_tmp, duration_tmp + copy_len, duration + i, tmp, 
          [=] __device__ (uint64_t a, uint64_t b) { return a + b; });
      auto min_ptr = thrust::min_element(thrust::device, tmp, tmp + copy_len);
      uint64_t update_min;
      cudaMemcpy(&update_min, min_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
      auto less_ptr = thrust::upper_bound(thrust::device, duration_tmp, duration_tmp + copy_len, update_min, thrust::less<uint64_t>());
      less_ptr = less_ptr == duration_tmp ? duration_tmp + 1 : less_ptr;
      int update_len = (uint64_t*)less_ptr - (uint64_t*)duration_tmp;
      thrust::copy_n(thrust::device, tmp, update_len, duration_tmp);
      i += update_len;
      thrust::sort(thrust::device, duration_tmp, duration_tmp + max_block_num_per_sm*sm_num);
      cudaFree(tmp);
    }
    cudaMemcpy(&total_cycle, duration_tmp + max_block_num_per_sm*sm_num - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(duration_tmp);
  }
}

void spmm_time_simulation_wrapper(
    const int* node_ptr,
    const int* edgeToColumn,
    const int* blockPartition,
    const uint32_t* block_offset,
    uint64_t* duration,
    int L1_cache_size,
    int L2_cache_size,
    int read_write_bytesize,
    int block_high,
    int nodes_num,
    int edges_num,
    int row_num,
    int max_blocks
) {
  int *cache_line_id, *block_iter_idx, *block_id, *block_row_idx;
  cudaMalloc((void**)&cache_line_id, sizeof(int)*edges_num);
  cudaMalloc((void**)&block_iter_idx, sizeof(int)*edges_num);
  thrust::sequence(thrust::device, cache_line_id, cache_line_id + edges_num);
  thrust::transform(thrust::device, cache_line_id, cache_line_id + edges_num, 
      cache_line_id, [=] __device__ (int i) { return i / 8; });
  thrust::transform(thrust::device, edgeToColumn, edgeToColumn + edges_num, 
      block_iter_idx, [=] __device__ (int i) { return i / 8; });
  cudaMalloc((void**)&block_id, sizeof(int)*edges_num);
  cudaMalloc((void**)&block_row_idx, sizeof(int)*(nodes_num-1));
  thrust::fill_n(thrust::device, block_id, edges_num, 0);
  thrust::fill_n(thrust::device, block_row_idx, nodes_num, 1);
  thrust::scatter(thrust::device, block_row_idx, block_row_idx+nodes_num-1, node_ptr+1, block_id);
  thrust::inclusive_scan(thrust::device, block_id, block_id + edges_num, block_id);
  cudaFree(block_row_idx);
  thrust::transform(thrust::device, block_id, block_id + edges_num, block_iter_idx,
      block_iter_idx, [=] __device__ (int a, int b) { return a / block_high * max_blocks + b; });
  cudaFree(block_id);
  thrust::stable_sort_by_key(thrust::device, block_iter_idx, block_iter_idx + edges_num, 
      cache_line_id);
  uint32_t *mask, *block_iter_nonzero_num, *block_iter_nonzero_num_offset;
  int* reduced_block_iter_idx;
  cudaMalloc((void**)&reduced_block_iter_idx, sizeof(int)*edges_num);
  cudaMalloc((void**)&mask, sizeof(uint32_t)*edges_num);
  cudaMalloc((void**)&block_iter_nonzero_num, sizeof(uint32_t)*edges_num);
  thrust::fill_n(thrust::device, mask, edges_num, 1);
  auto pair_ends = thrust::reduce_by_key(thrust::device, block_iter_idx, block_iter_idx + edges_num, mask, 
      reduced_block_iter_idx, block_iter_nonzero_num);
  int block_num = pair_ends.first - reduced_block_iter_idx;
  cudaMalloc((void**)&block_iter_nonzero_num_offset, sizeof(uint32_t)*(block_num+1));
  cudaMemset(block_iter_nonzero_num_offset, 0, sizeof(uint32_t));
  thrust::inclusive_scan(thrust::device, block_iter_nonzero_num, block_iter_nonzero_num + block_num, 
      block_iter_nonzero_num_offset + 1);
  int* L1_hit_mark;
  cudaMalloc((void**)&L1_hit_mark, sizeof(int)*block_num);
  thrust::fill_n(thrust::device, L1_hit_mark, block_num, 1);
  int* nonzero_per_block;
  cudaMalloc((void**)&nonzero_per_block, sizeof(int)*row_num);
  int grid = (block_num + 127) / 128;
  get_nonzero_per_block<<<grid, 128>>>(node_ptr, nonzero_per_block, row_num, block_high, nodes_num);
  cudaDeviceSynchronize();
  auto max_cache_line_num = thrust::max_element(thrust::device, nonzero_per_block, nonzero_per_block + row_num);
  int max_cacheline_num;
  cudaMemcpy(&max_cacheline_num, max_cache_line_num, sizeof(int), cudaMemcpyDeviceToHost);
  int block_dim = 32;
  int grid_dim = row_num;
  mark_L1_hit<<<grid_dim, block_dim, sizeof(int)*max_cacheline_num>>>(
      block_offset, cache_line_id, node_ptr, block_iter_nonzero_num, block_iter_nonzero_num_offset,
      L1_hit_mark, row_num, block_high, nodes_num);
  cudaDeviceSynchronize();
  DataSaveLayer save_layer;
  if (read_write_bytesize <= L2_cache_size) {
    save_layer = L2;
  } else {
    save_layer = DRAM;
  }
  grid = (row_num + 127) / 128;
  calculate_simulation_time<<<grid, 128>>>(
      L1_hit_mark, block_offset, duration, save_layer, row_num);
  printf("get duration done\n");
  cudaFree(cache_line_id);
  cudaFree(block_iter_idx);
  cudaFree(mask);
  cudaFree(block_iter_nonzero_num);
  cudaFree(block_iter_nonzero_num_offset);
  cudaFree(L1_hit_mark);
  cudaFree(nonzero_per_block);
}

__global__ void calculate_DTC_simulation_time(
    const uint32_t* __restrict__ RowWindow_offset,
    const int* __restrict__ TCblock_offset,
    uint64_t* __restrict__ duratiion,
    DataSaveLayer save_layer,
    int row_num
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= row_num) return;
  uint32_t row_start = RowWindow_offset[tid];
  uint32_t row_end = RowWindow_offset[tid + 1];
  uint64_t cycle = 0;
  uint64_t miss_latency;
  if (save_layer < DRAM) {
    miss_latency = 250;
  } else {
    miss_latency = 550;
  }
  cycle += (row_end - row_start + 7) / 8 * (miss_latency - 75);
  int block_start = TCblock_offset[row_start];
  int block_end = TCblock_offset[row_start + 1];
  if((block_start / 8) == (block_end / 8) && (block_start % 8 > 0)) {
    cycle += (29 * 2 + 75 + miss_latency * 3);
  } else {
    cycle += (29 * 2 + 75 * 3 + miss_latency);
  }
  for (uint32_t i = row_start + 1; i < row_end; i++) {
    cycle += miss_latency * 4;
    uint64_t tmp_cycle = 0;
    block_start = TCblock_offset[i];
    block_end = TCblock_offset[i + 1];
    if((block_start / 8) == (block_end / 8) && (block_start % 8 > 0)) {
      tmp_cycle += (29 * 2 + 75 + miss_latency * 3);
    } else {
      tmp_cycle += (29 * 2 + 75 * 3 + miss_latency);
    }
    cycle += max(tmp_cycle, (uint64_t)(29 * 8 + 32 * 2));
  }
  cycle += (miss_latency * 4 + 32 * 2);
  cycle += (75 * 2 + miss_latency * 2) * 2;
  duratiion[tid] = cycle;
}

void spmm_DTC_time_simulation_wrapper(
    const uint32_t* RowWindow_offset,
    const int* TCblock_offset,
    uint64_t* duartion,
    int L1_cache_size,
    int L2_cache_size,
    int read_write_bytesize,
    int row_num
) {
  DataSaveLayer save_layer;
  if (read_write_bytesize <= L2_cache_size) {
    save_layer = L2;
  } else {
    save_layer = DRAM;
  }
  int grid = (row_num + 127) / 128;
  calculate_DTC_simulation_time<<<grid, 128>>>(
      RowWindow_offset, TCblock_offset, duartion, save_layer, row_num);
}

__global__ void standard_deviation(const uint32_t* iters_offset, const uint32_t* block_offset, 
    const int* SparseCol_aToX, float* dev, int row_num) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= row_num) return;
    float var = 0.0f, mean = 0.f;
    uint32_t block_iter_start = iters_offset[tid];
    uint32_t block_iter_end = iters_offset[tid + 1];
    int nnz_num = block_offset[block_iter_end] - block_offset[block_iter_start];
    for (uint32_t i = block_iter_start; i < block_iter_end; i++) {
        uint32_t col_idx_start = block_offset[i];
        uint32_t col_idx_end = block_offset[i + 1];
        for (uint32_t j = col_idx_start; j < col_idx_end; j++) {
            mean += float(SparseCol_aToX[j] / nnz_num);
            var += float(SparseCol_aToX[j] / nnz_num) * float(SparseCol_aToX[j] / nnz_num);
        }
    }
    dev[tid] = sqrtf(fabsf(var * nnz_num - mean * mean));
}

////////////////////////////////////
/// SPMM forward (AGNN, AGNN)
///////////////////////////////////
__global__ void spmmAGNN_forward_cuda_kernel(
	const int * __restrict__    nodePointer,		// node pointer.
	const int *__restrict__     edgeList,			// edge list.
  const float *__restrict__   edgeAttention,	    // edge attention.
	const int *__restrict__     blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__     edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__     edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
  const int row_num,						    // number of row_windows.
	const float *__restrict__   input,		    // input feature matrix.
	float *output							    // aggreAGNNed output feature matrix.
    // uint64_t* timer,
    // int* smid_arr,
    // int* sm_order,
    // int* sm_counter
) {
  // uint smid;
  // asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  // int sm_ord;
  // if (threadIdx.x == 0 && threadIdx.y == 0) {
  //   sm_ord = atomicAdd(sm_counter + smid, 1);
  // }
  // uint64_t start, stop;
  // // synchronize all threads
  // asm volatile("bar.sync 0;");
  // // start timing
  // asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  const unsigned bid = blockIdx.x;								// block_index == row_window_index
  if (bid >= row_num) return;									// if row_window_index out of range.
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
	const unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
	
	const unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
	const unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
	const unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
	const unsigned dense_bound = numNodes * embedding_dim;

	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum, BLK_W, BLK_H]
	extern __shared__ float dense_X[];

	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);
  // float D[BLK_W] = {0.0f};

	// Processing TC_blocks along the column dimension of Sparse A.
	for (unsigned i = 0; i < num_TC_blocks; i++){

		// Init A_colToX_row with dummy values.
		if (tid < BLK_W){
			sparse_AToX_index[tid] = numNodes + 1;
		}

		__syncthreads();

		// Init sparse_A with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
			sparse_A[idx] = 0;
		}

		// Init dense_X with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
			dense_X[idx] = 0;
		}

		// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
		// currently fetch all neighbors of the current nodes.
		// then to see whether it can fit into current TC_block frame of column.		
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
			unsigned col = edgeToColumn[eIdx];
			if (i * BLK_W <= col && col < (i + 1) * BLK_W){			// if the edge in the current TC_block frame of column.
				unsigned row_local = edgeToRow[eIdx] % BLK_H;
				unsigned col_local = col % BLK_W;
				sparse_A[row_local * BLK_W + col_local] = edgeAttention[eIdx];		// sparse_A according to edge_features.
				sparse_AToX_index[col_local] = edgeList[eIdx];		                // record the mapping from sparse_A colId to rowId of dense_X.
			}		
		}

		__syncthreads();

		// Initialize dense_X by column-major store,
		// Threads of a warp for fetching a dense_X.
		// each warp identify by wid.
		if (wid < dimTileNum)
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
				unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
				unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
				unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
				unsigned target_idx = wid * BLK_W * BLK_H + idx;
				// boundary test.
				if (source_idx >= dense_bound)
					dense_X[target_idx] = 0;
				else
					dense_X[target_idx] = input[source_idx];
			}

		__syncthreads();

		if (wid < dimTileNum)
		{
      // for (int j = 0; j < 2; j++) {
      //   for (int k = 0; k < 4; k++) {
      //     for (int l = 0; l < 8; l++) {
      //       D[j*4+k] += sparse_A[j*64+(laneid>>2)*8+l]*dense_X[wid*BLK_W*BLK_H+k*32+(laneid&3)*8+l];
      //     }
      //   }
      // }
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);

			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}

			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}
	}

	if (wid < dimTileNum)
		// Store the matrix to output matrix.
		// * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
		wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
    // for (int j = 0; j < 2; j++) 
    //   for (int k = 0; k < 4; k++) 
    //     output[bid*BLK_H*embedding_dim+(j*8+(laneid>>2))*embedding_dim+wid*BLK_H+k*4+(laneid&3)] = D[j*4+k];

  // stop timing
  // synchronize all threads
  // asm volatile("bar.sync 0;");
  // asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  // if (threadIdx.x == 0 && threadIdx.y == 0) {
  //     timer[bid] = stop - start;
  //     smid_arr[bid] = (int)smid;
  //     sm_order[bid] = sm_ord;
  // }
}

__global__ void spmmAGNN_forward_cuda_kernel_single(
	const int * __restrict__    nodePointer,		// node pointer.
	const int *__restrict__     edgeList,			// edge list.
  const float *__restrict__   edgeAttention,	    // edge attention.
	const int *__restrict__     blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__     edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__     edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
  const int row_num,						    // number of row_windows.
	const float *__restrict__   input,		    // input feature matrix.
	float * __restrict__ output,							    // aggreAGNNed output feature matrix
  uint64_t* __restrict__ read_latency,
  uint8_t* __restrict__ read_token,
  uint8_t* __restrict__ threadIdx_record,
  uint32_t* __restrict__ record_len_ptr
) {
  const unsigned bid = blockIdx.x;								// block_index == row_window_index
  if (bid >= row_num) return;									// if row_window_index out of range.
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
	const unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
	
	const unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
	const unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
	const unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
	const unsigned dense_bound = numNodes * embedding_dim;

	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum, BLK_W, BLK_H]
	extern __shared__ float dense_X[];
  __shared__ uint32_t record_len[1];
  if (threadIdx.x == 0 && threadIdx.y == 0)
    record_len[0] = 0;

	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);
  // float D[BLK_W] = {0.0f};

	// Processing TC_blocks along the column dimension of Sparse A.
	for (unsigned i = 0; i < num_TC_blocks; i++){

		// Init A_colToX_row with dummy values.
		if (tid < BLK_W){
			sparse_AToX_index[tid] = numNodes + 1;
		}

		__syncthreads();

		// Init sparse_A with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
			sparse_A[idx] = 0;
		}

		// Init dense_X with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
			dense_X[idx] = 0;
		}

		// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
		// currently fetch all neighbors of the current nodes.
		// then to see whether it can fit into current TC_block frame of column.		
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
      
      uint64_t start, stop;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
			unsigned col = edgeToColumn[eIdx];
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
      if (bid == (row_num - 100)) {
        int id = atomicAdd(record_len, 1);
        read_latency[id] = stop - start;
        read_token[id] = 0;
        threadIdx_record[id] = threadIdx.x + threadIdx.y * blockDim.x;
      }

			if ((i * BLK_W <= col) && (col < (i + 1) * BLK_W)){			// if the edge in the current TC_block frame of column.
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
				unsigned row_local = edgeToRow[eIdx];
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
        if (bid == (row_num - 100)) {
          int id = atomicAdd(record_len, 1);
          read_latency[id] = stop - start;
          read_token[id] = 1;
          threadIdx_record[id] = threadIdx.x + threadIdx.y * blockDim.x;
        }

        row_local = row_local % BLK_H;
				unsigned col_local = col % BLK_W;

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
        sparse_A[row_local * BLK_W + col_local] = edgeAttention[eIdx];		// sparse_A according to edge_features.
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
        if (bid == (row_num - 100)) {
          int id = atomicAdd(record_len, 1);
          read_latency[id] = stop - start;
          read_token[id] = 2;
          threadIdx_record[id] = threadIdx.x + threadIdx.y * blockDim.x;
        }

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
				sparse_AToX_index[col_local] = edgeList[eIdx];		                // record the mapping from sparse_A colId to rowId of dense_X.
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
        if (bid == (row_num - 100)) {
          int id = atomicAdd(record_len, 1);
          read_latency[id] = stop - start;
          read_token[id] = 3;
          threadIdx_record[id] = threadIdx.x + threadIdx.y * blockDim.x;
        }
			}		
		}

		__syncthreads();

		// Initialize dense_X by column-major store,
		// Threads of a warp for fetching a dense_X.
		// each warp identify by wid.
		if (wid < dimTileNum)
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
				unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
				unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
				unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
				unsigned target_idx = wid * BLK_W * BLK_H + idx;
				// boundary test.
				if (source_idx >= dense_bound)
					dense_X[target_idx] = 0;
				else {
          uint64_t start, stop;
          asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
          dense_X[target_idx] = input[source_idx];
          asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
          if (bid == (row_num - 100)) {
            int id = atomicAdd(record_len, 1);
            read_latency[id] = stop - start;
            read_token[id] = 4;
            threadIdx_record[id] = threadIdx.x + threadIdx.y * blockDim.x;
          }
        }
			}

		__syncthreads();

		if (wid < dimTileNum)
		{
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);

			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}

			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}
	}

	if (wid < dimTileNum)
		// Store the matrix to output matrix.
		// * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
		wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);

  if (threadIdx.x == 0 && threadIdx.y == 0 && bid == (row_num - 100)) 
    record_len_ptr[0] = record_len[0];
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer(
	const uint32_t *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
  // uint64_t start, stop;
  // // synchronize all threads
  // asm volatile("bar.sync 0;");
  // // start timing
  // asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
  uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
  uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	 
		// if (tid < BLK_W) {
		//   sparse_AToX_index[tid] = numNodes + 1;
		// }
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
    int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;

		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
	
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		}
	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(&valuesA[eIdx]));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(&sparse_AToX_idx[sparse_AToX_idx_start + tid]));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );


	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
    // stop timing
    // synchronize all threads
    // asm volatile("bar.sync 0;");
    // asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
    // if (threadIdx.x == 0) {
    //     timer[bid] = stop - start;
    // }
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_single(
	const uint32_t *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output,							    // output feature matrix.
  uint64_t *__restrict__ read_latency,
  uint8_t* __restrict__ read_token,
  uint8_t* __restrict__ threadIdx_record,
  uint32_t* __restrict__ record_len_ptr,
  int record_block
) {
  // uint64_t start, stop;
  // // synchronize all threads
  // asm volatile("bar.sync 0;");
  // // start timing
  // asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
  __shared__ uint32_t record_len[1];
  int record_blockid = record_block;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
  uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
  uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
  if (threadIdx.x == 0 && threadIdx.y == 0)
    record_len[0] = 0;
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	 
		// if (tid < BLK_W) {
		//   sparse_AToX_index[tid] = numNodes + 1;
		// }
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
    uint64_t start, stop;
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
		  sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
      if (bid == record_blockid) {
        int id = atomicAdd(record_len, 1);
        read_latency[id] = stop - start;
        read_token[id] = 0;
        threadIdx_record[id] = tid;
      }
		}
		if (tid < BLK_W) {
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
      if (bid == record_blockid) {
        int id = atomicAdd(record_len, 1);
        read_latency[id] = stop - start;
        read_token[id] = 1;
        threadIdx_record[id] = tid;
      }
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
    int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;

		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
      uint64_t start, stop;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
	
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
      if (bid == record_blockid) {
        int id = atomicAdd(record_len, 1);
        read_latency[id] = stop - start;
        read_token[id] = 2;
        threadIdx_record[id] = tid;
      }
		}
	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
      uint64_t start, stop;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(&valuesA[eIdx]));	
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
      if (bid == record_blockid) {
        int id = atomicAdd(record_len, 1);
        read_latency[id] = stop - start;
        read_token[id] = 3;
        threadIdx_record[id] = tid;
      }
	  }
		if (tid < BLK_W) {	
      uint64_t start, stop;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(&sparse_AToX_idx[sparse_AToX_idx_start + tid]));	
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
      if (bid == record_blockid) {
        int id = atomicAdd(record_len, 1);
        read_latency[id] = stop - start;
        read_token[id] = 4;
        threadIdx_record[id] = tid;
      }
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
    uint64_t start, stop;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
    if (bid == record_blockid) {
      int id = atomicAdd(record_len, 1);
      read_latency[id] = stop - start;
      read_token[id] = 2;
      threadIdx_record[id] = tid;
    }
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );


	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
  if (threadIdx.x == 0 && threadIdx.y == 0 && bid == record_blockid)
    record_len_ptr[0] = record_len[0];
    // stop timing
    // synchronize all threads
    // asm volatile("bar.sync 0;");
    // asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
    // if (threadIdx.x == 0) {
    //     timer[bid] = stop - start;
    // }
}

/*Generate TC offset, tileid and AtoB*/
// __global__ void generate_tcoffset_id_atob(
//     int *nodePointer, uint32_t *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
//     int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id,
//     int *sparseatob, int max_block, int num_nodes, int blockSize_h,
//     int blockSize_w, int num_row_windows) {
//   extern __shared__ int pos_ptr[];
//   int tid = threadIdx.x;
//   int winId = blockIdx.x; // each warp one window
//   unsigned block_start = rowwindow_offset[winId];
//   unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
//   unsigned num_blocks = block_end - block_start;
//   if (num_blocks == 0) {
//     return;
//   }
//   int *tcblock_offset_ptr = pos_ptr + num_blocks;
//   int *tcblock_offset_global_ptr = tcblock_offset + block_start;
//   int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
//   unsigned element_start = nodePointer[winId * blockSize_h];
//   unsigned element_end =
//       nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
//   unsigned num_window_edges = element_end - element_start;
//   if (num_window_edges == 0) {
//     return;
//   }
//   for (int i = 0; i < 2 * num_blocks + 1; i++) {
//     pos_ptr[i] = 0;
//   }
//   for (unsigned e_index = element_start; e_index < element_end; e_index++) {
//     unsigned col = edgeToColumn[e_index]; // new col
//     tcblock_nnz_ptr[col / blockSize_w]++;
//   }
//   for (int i = 0; i < num_blocks; i++) {
//     tcblock_offset_global_ptr[i] = tcblock_nnz_ptr[i];
//   }
//   auto tileid = tcblocktile_id + element_start;
//   auto sparse_AToB = sparseatob + block_start * blockSize_w;
//   for (int i = 0; i < num_blocks; i++) {
//     tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
//   }
//   for (unsigned e_index = element_start; e_index < element_end; e_index++) {
//     unsigned col = edgeToColumn[e_index]; // new col
//     unsigned tcblock_id = col / blockSize_w;
//     unsigned row_local = edgeToRow[e_index] % blockSize_h;
//     unsigned col_local = col % blockSize_w;
//     tileid[tcblock_offset_ptr[tcblock_id] + pos_ptr[tcblock_id]] =
//         (uint8_t)(row_local * blockSize_w + col_local);
//     sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
//     pos_ptr[tcblock_id]++;
//   }
// }

__global__ void generate_tcoffset_id_atob(
    int *nodePointer, uint32_t *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id,
    int *sparseatob, int max_block, int num_nodes, int blockSize_h,
    int blockSize_w, int num_row_windows) {
      // extern __shared__ int pos_ptr[];
      __shared__ unsigned offset[1]; 
      __shared__ unsigned mask[8];
      int tid = threadIdx.x;
      int winId = blockIdx.x; // each warp one window
      unsigned block_start = rowwindow_offset[winId];
      unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
      unsigned num_blocks = block_end - block_start;
      if (num_blocks == 0) 
        return;
      // int *tcblock_offset_ptr = pos_ptr + num_blocks;
      int *tcblock_offset_global_ptr = tcblock_offset + block_start;
      // int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
      unsigned element_start = nodePointer[winId * blockSize_h];
      unsigned element_end =
          nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
      // for (int i = tid; i < 2 * num_blocks + 1; i += blockDim.x) {
      //   pos_ptr[i] = 0;
      // }
      if (threadIdx.x == 0)
        offset[0] = 0;
      __syncthreads();
      auto tileid = tcblocktile_id + element_start;
      for (int i = 0; i < num_blocks; i++) {
        for (unsigned e_index = element_start + tid; e_index < element_end; e_index += blockDim.x) {
          unsigned col = edgeToColumn[e_index]; // new col
          if (i == 0)
            atomicAdd(tcblock_offset_global_ptr + col / blockSize_w, 1);
          if ((threadIdx.x&31)==0)
            mask[threadIdx.x>>5]=0;
          int set = ((col >= blockSize_w * i && col < blockSize_w * (i + 1))<<(threadIdx.x&31));
          for (int j = 1; j < 32; j <<= 1)
            set |= __shfl_xor_sync(0xffffffff, set, j*2-1);
          mask[threadIdx.x>>5] = set;
          __syncthreads();
          if (col >= blockSize_w * i && col < blockSize_w * (i + 1)) {
            unsigned row_local = edgeToRow[e_index] % blockSize_h;
            unsigned col_local = col % blockSize_w;
            unsigned off = __popc(set & ((1<<(threadIdx.x&31))-1));
            for (int j = 0; j < (threadIdx.x>>5); j++)
              off += __popc(mask[j]);
            // unsigned off = atomicAdd(offset, 1);
            tileid[offset[0]+off] = (uint8_t)(row_local * blockSize_w + col_local);
            sparseatob[(block_start + i) * blockSize_w + col_local] = edgeList[e_index];
          }
          __syncthreads();
          if (threadIdx.x == 0)
            for (int j = 0; j < 4; j++)
              offset[0] += __popc(mask[j]);
        }
      }
      
}

void generate_tcoffset_id_atob_cuda(int *nodePointer, uint32_t *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, int *tcblock_offset,
                                    uint8_t *tcblock_tileid, int *sparseatob,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 256;
  int window_count = num_row_windows;
  // const int dynamic_shared_size = (2 * max_block + 1) * sizeof(int);
  // std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
  // if (dynamic_shared_size > 98304) {
  //   int maxbytes = 131072; // 96 KB
  //   cudaFuncSetAttribute(generate_tcoffset_id_atob,
  //                        cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  // } else if (dynamic_shared_size > 65536) {
  //   int maxbytes = 98304; // 96 KB
  //   cudaFuncSetAttribute(generate_tcoffset_id_atob,
  //                        cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  // } else if (dynamic_shared_size > 32768) {
  //   int maxbytes = 65536; // 128 KB
  //   cudaFuncSetAttribute(generate_tcoffset_id_atob,
  //                        cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  // }
  generate_tcoffset_id_atob<<<window_count, block_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
      tcblock_offset, tcblock_tileid, sparseatob, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}  

std::vector<uint64_t> spmm_compare(
   py::array_t<int> row_block_num,
   py::array_t<int> indice,
   py::array_t<int> indptr,
  //  py::array_t<uint64_t> duration_array,
  //  py::array_t<uint64_t> timer_TCGNN_array,
  //  py::array_t<uint64_t> timer_DTC_array,
  //  py::array_t<int> smid_array,
  //  py::array_t<int> sm_order_array,
  //  py::array_t<int> sm_counter_array,
  //  py::array_t<uint64_t> read_latency,
  //  py::array_t<uint32_t> read_offset,
  //  py::array_t<uint8_t> read_token_,
//    py::array_t<int> iter_test,
   py::array_t<float> dev_array,
   torch::Tensor blockPartition,
   torch::Tensor edgeToColumn,
   torch::Tensor edgeToRow,
   int nodes_num,
   int edges_num,
   int block_num,
   int embedding_dim,
   int block_high,
   int block_width
) {
    int *indice_d, *indptr_d, *blockPartution_d, *edgeToColumn_d, *edgeToRow_d;
    uint32_t *row_block_num_d;
    int row_num = (nodes_num + block_high - 1) / block_high;
    printf("row_num: %d\n", row_num);
    cudaMalloc(&row_block_num_d, sizeof(uint32_t) * (row_num + 1));
    cudaMemset(row_block_num_d, 0, sizeof(uint32_t));
    cudaMalloc(&indice_d, sizeof(int) * edges_num);
    cudaMalloc(&indptr_d, sizeof(int) * (nodes_num + 1));
    int blockPartition_len = blockPartition.size(0);
    cudaMalloc(&blockPartution_d, sizeof(int) * row_num);
    cudaMalloc(&edgeToColumn_d, sizeof(int) * edges_num);
    cudaMalloc(&edgeToRow_d, sizeof(int) * edges_num);
    int *Sparsecol_AToX;
    uint32_t *value;
    cudaMalloc(&value, sizeof(uint32_t) * edges_num);
    cudaMalloc(&Sparsecol_AToX, sizeof(int) * edges_num);
    // auto row_block_num_ptr = row_block_num.request().ptr;
    auto indice_ptr = indice.request().ptr;
    auto indptr_ptr = indptr.request().ptr;
    // cudaMemcpy(row_block_num_d, row_block_num_ptr, sizeof(int) * row_num, cudaMemcpyHostToDevice);
    cudaMemcpy(indice_d, indice_ptr, sizeof(int) * edges_num, cudaMemcpyHostToDevice);
    cudaMemcpy(indptr_d, indptr_ptr, sizeof(int) * (nodes_num + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(blockPartution_d, blockPartition.data_ptr<int>(), sizeof(int) * blockPartition_len, cudaMemcpyHostToDevice);
    cudaMemcpy(edgeToColumn_d, edgeToColumn.data_ptr<int>(), sizeof(int) * edges_num, cudaMemcpyHostToDevice);
    cudaMemcpy(edgeToRow_d, edgeToRow.data_ptr<int>(), sizeof(int) * edges_num, cudaMemcpyHostToDevice);
    thrust::copy(thrust::device, indice_d, indice_d + edges_num, Sparsecol_AToX);
    thrust::transform(thrust::device, edgeToRow_d, edgeToRow_d + edges_num, edgeToColumn_d, 
        value, [=]__device__(int row, int col) { return (uint32_t)(row / block_high) * (uint32_t)nodes_num + (uint32_t)col; });
    thrust::stable_sort_by_key(thrust::device, value, value + edges_num, Sparsecol_AToX);
    auto end = thrust::unique_by_key(thrust::device, value, value + edges_num, Sparsecol_AToX);
    int nnz = end.first - value;
    int* mask, *block_nonzero_num;
    uint32_t *block_offset;
    // printf("nnz: %d\n", nnz);
    cudaMalloc(&mask, sizeof(int) * nnz);
    cudaMalloc(&block_nonzero_num, sizeof(int) * nnz);
    cudaMalloc(&block_offset, sizeof(uint32_t) * (block_num + 1));
    thrust::fill_n(thrust::device, mask, nnz, 1);
    thrust::transform(thrust::device, value, value + nnz, value, 
        [=]__device__(uint32_t idx) {return ((uint32_t)(idx % (uint32_t)nodes_num) / block_width) * (uint32_t)block_width + (uint32_t)(idx / nodes_num) * (uint32_t)nodes_num;});
    thrust::reduce_by_key(thrust::device, value, value + nnz, mask, value, block_nonzero_num,
        thrust::equal_to<uint32_t>(), thrust::plus<int>());
    thrust::inclusive_scan(thrust::device, block_nonzero_num, block_nonzero_num + block_num, block_offset + 1,
                        [=]__device__(int a, uint32_t b) {return (uint32_t)a + b;});
    thrust::inclusive_scan(thrust::device, blockPartution_d, blockPartution_d + row_num, row_block_num_d + 1, 
                        [=]__device__(int a, uint32_t b) {return (uint32_t)(a + b);});
    // printf("after copy data\n");
    float* edgeattention_d, *input, *output;
    cudaMalloc(&edgeattention_d, sizeof(float) * edges_num);
    cudaMalloc(&input, sizeof(float) * nodes_num * embedding_dim);
    cudaMalloc(&output, sizeof(float) * block_high * row_num * embedding_dim);
    thrust::fill_n(thrust::device, edgeattention_d, edges_num, 1.0f);
    thrust::fill_n(thrust::device, input, nodes_num * embedding_dim, 1.0f);
    printf("edgeattention_d address: %ld\n", ((long)edgeattention_d & 7));
    /////////////////////////////////TC-GNN////////////////////////////////////////
    uint64_t *duration;
    // cudaMalloc(&timer_d, sizeof(uint64_t) * row_num); * timer_d,
    cudaMalloc(&duration, sizeof(uint64_t) * row_num);
    // int* smid_arr, *sm_order, *sm_counter;
    // cudaMalloc(&smid_arr, sizeof(int) * row_num);
    // cudaMalloc(&sm_order, sizeof(int) * row_num);
    // cudaMalloc(&sm_counter, sizeof(int) * 28);
    // thrust::fill_n(thrust::device, sm_counter, 28, 0);
    // uint64_t* latency_record;
    // uint8_t* threadIdx_record, *read_token;
    // uint32_t* record_len_ptr;
    // cudaMalloc(&latency_record, sizeof(uint64_t) * 3 * 1024 * 1024);
    // cudaMalloc(&threadIdx_record, sizeof(uint8_t) * 3 * 1024 * 1024);
    // cudaMalloc(&read_token, sizeof(uint8_t) * 3 * 1024 * 1024);
    // cudaMalloc(&record_len_ptr, sizeof(uint32_t));
    // cudaMemset(record_len_ptr, 0, sizeof(uint32_t));
    const int dimTileNum = (embedding_dim + block_high - 1) / block_high;
	  const int dynamic_shared_size = dimTileNum * block_width * block_high * sizeof(float); // dynamic shared memory.
    dim3 grid(row_num, 1, 1);
    dim3 block(32, 8, 1);
    //, timer_d, smid_arr, sm_order, sm_counter
    float kernel_elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaDeviceSynchronize();
    spmmAGNN_forward_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
        indptr_d, indice_d, edgeattention_d, blockPartution_d, edgeToColumn_d, edgeToRow_d, 
        nodes_num, edges_num, embedding_dim, row_num, input, output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_elapsed_time, start, stop);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    uint64_t total_real = deviceProp.clockRate * kernel_elapsed_time;
    printf("real cycle for kernel: %d cycles.\n", total_real);
    // int* max_blocks_num_per_sm = (int*)malloc(sizeof(int));
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(max_blocks_num_per_sm, spmmAGNN_forward_cuda_kernel_single, 256, dynamic_shared_size); 
    // size_t* max_Dynamic_SMem_per_sm = (size_t*)malloc(sizeof(size_t));
    // cudaOccupancyAvailableDynamicSMemPerBlock(max_Dynamic_SMem_per_sm, spmmAGNN_forward_cuda_kernel_single, *max_blocks_num_per_sm, 256);
    // printf("max_Dynamic_SMem_per_sm: %d KB\n", (*max_Dynamic_SMem_per_sm * *max_blocks_num_per_sm) / 1024);
    // printf("max_blocks_num_per_sm: %d\n", *max_blocks_num_per_sm);
    // cudaDeviceSynchronize();
    // //*max_blocks_num_per_sm*28(first wave running amount)
    // spmmAGNN_forward_cuda_kernel_single<<<grid, block, dynamic_shared_size>>>(
    //     indptr_d, indice_d, edgeattention_d, blockPartution_d, edgeToColumn_d, edgeToRow_d, 
    //     nodes_num, edges_num, embedding_dim, row_num, input, output, latency_record, 
    //     read_token, threadIdx_record, record_len_ptr);
    // cudaDeviceSynchronize();
    /////////////////////////////////DTC//////////////////////////////////////////////
    int* tcblock_offset, *sparse_AToX_index;
    uint8_t* tcblocktile_id;
    uint64_t* timer_d1;
    cudaMalloc(&timer_d1, sizeof(uint64_t) * row_num);
    cudaMalloc(&tcblock_offset, sizeof(int) * (block_num + 1));
    cudaMalloc(&sparse_AToX_index, sizeof(int) * block_num * block_width);
    cudaMalloc(&tcblocktile_id, sizeof(uint8_t) * edges_num);
    // cudaMemset(tcblock_offset, 0, sizeof(int));
    thrust::fill_n(thrust::device, tcblock_offset, block_num + 1, 0);
    thrust::fill_n(thrust::device, sparse_AToX_index, block_num * block_width, nodes_num);
    // printf("before max_element\n");
    auto max = thrust::max_element(thrust::device, blockPartution_d, blockPartution_d + row_num);
    // printf("after max_element\n");
    int max_blocks;
    cudaMemcpy(&max_blocks, max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("max_blocks: %d\n", max_blocks);
    generate_tcoffset_id_atob_cuda(indptr_d, row_block_num_d, edgeToColumn_d, edgeToRow_d,
        indice_d, tcblock_offset + 1, tcblocktile_id, sparse_AToX_index, max_blocks, nodes_num,
        block_high, block_width, row_num);
    thrust::inclusive_scan(thrust::device, tcblock_offset + 1, tcblock_offset + 1 + block_num, tcblock_offset + 1);
    // printf("after generate_tcoffset_id_atob_cuda\n");
    const int WARPperblock = embedding_dim / block_high;
    dim3 grid_(row_num, 1, 1);
    dim3 block_(32, WARPperblock, 1);
    spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer<<<grid_, block_>>>(
        row_block_num_d, tcblocktile_id, tcblock_offset, sparse_AToX_index, edgeattention_d, nodes_num,
        edges_num, embedding_dim, input, output);
    cudaDeviceSynchronize();
    // printf("after spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer\n");
    /////////////////////////////////////////////////////////////////////////////////////
    // printf("after spmmAGNN_forward_cuda_kernel\n");
    int dimtileNum = (embedding_dim + block_high - 1) / block_high;
    void *params[8] = {indice_d, indptr_d, blockPartution_d, edgeToColumn_d, edgeToRow_d, 
                       edgeattention_d, input, output};
    int total_data_size = calculate_data_size(8, params);
    // int total_read_write = (row_num + (nodes_num + 1) + edges_num * 4) * 4 + nodes_num * embedding_dim * 8;
    printf("total read write size: %d bytes\n", total_data_size);
    spmm_time_simulation_wrapper(
      indptr_d, edgeToColumn_d, blockPartution_d, row_block_num_d, duration,
      28672, 3145728, total_data_size, 16, nodes_num, edges_num, row_num,
      max_blocks);
    printf("after spmm_time_simulation_wrapper\n");
    uint64_t total_time;
    spmm_kernel_simulation(duration, 6, 28, row_num, total_time);
    printf("simulation_total_time: %lu\n", total_time);
    // if (total_read <= 3 * 1024 * 1024) {
    //     hit_latency = 50;
    //     // miss 201 or 491
    //     miss_latency_low = 250;
    //     miss_latency_high = 491;
    //     printf("L2 cache save: %d\n", total_read);
    // } else if (total_read <= 32 * 1024 * 1024) {
    //     hit_latency = 18;
    //     // miss 491 or 871
    //     miss_latency_low = 491;
    //     miss_latency_high = 871;
    //     printf("TLB page save: %d\n", total_read);
    // } else {
    //     hit_latency = 50;
    //     // miss 871 or 1222
    //     miss_latency_low = 250;
    //     miss_latency_high = 1222;
    //     printf("DRAM save: %d\n", total_read);
    // }
//////////////////////////////////////////////////
    // int* iter_test_d;
    // cudaMalloc(&iter_test_d, sizeof(int) * row_num);
//////////////////////////////////////////////////
    // spmm_time_simulation<<<blocks, 1024>>>(duration, row_block_num_d, block_offset,
    //     Sparsecol_AToX, indptr_d, block_high, block_width, dimtileNum, 
    //     row_num, nodes_num, hit_latency, miss_latency_low, miss_latency_high);
    // cudaDeviceSynchronize();
    printf("after spmm_time_simulation\n");
    float* dev_arr;
    cudaMalloc(&dev_arr, sizeof(float) * row_num);
    int blocks = (row_num + 1023) / 1024;
    standard_deviation<<<blocks, 1024>>>(row_block_num_d, block_offset, Sparsecol_AToX,
        dev_arr, row_num);
    cudaDeviceSynchronize();
    printf("after standard_deviation\n");
    // printf("after spmm_time_simulation\n");
/////////////////////////////////////////////////////////
    // int off_len, block_off_len;
    // uint32_t write_len;
    // cudaMemcpy(&off_len, indptr_d + 16, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&block_off_len, blockPartution_d, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&write_len, record_len_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // uint8_t* threadIdx_record_copy;
    // cudaMalloc(&threadIdx_record_copy, sizeof(uint8_t) * write_len);
    // thrust::copy(thrust::device, threadIdx_record, threadIdx_record + write_len, threadIdx_record_copy);
    // thrust::stable_sort_by_key(thrust::device, threadIdx_record, threadIdx_record + write_len, latency_record);
    // thrust::stable_sort_by_key(thrust::device, threadIdx_record_copy, threadIdx_record_copy + write_len, read_token);
    // uint32_t* thread_counts, *counts_mask, *thread_offset;
    // cudaMalloc(&counts_mask, sizeof(uint32_t) * write_len);
    // cudaMalloc(&thread_counts, sizeof(uint32_t) * 256);
    // cudaMalloc(&thread_offset, sizeof(uint32_t) * 257);
    // thrust::fill_n(thrust::device, thread_counts, 256, 0);
    // cudaMemset(thread_offset, 0, sizeof(uint32_t));
    // thrust::fill_n(thrust::device, counts_mask, write_len, 1);
    // thrust::reduce_by_key(thrust::device, threadIdx_record, threadIdx_record + write_len, counts_mask,
    //     threadIdx_record, thread_counts, thrust::equal_to<uint8_t>(), thrust::plus<uint32_t>());  
    // thrust::inclusive_scan(thrust::device, thread_counts, thread_counts + 256, thread_offset + 1);
/////////////////////////////////////////////////////////
    // auto duration_array_ptr = duration_array.request().ptr;
    // cudaMemcpy(duration_array_ptr, duration, sizeof(uint64_t) * row_num, cudaMemcpyDeviceToHost);
    // auto timer_array_ptr = timer_TCGNN_array.request().ptr;
    // cudaMemcpy(timer_array_ptr, timer_d, sizeof(uint64_t) * row_num, cudaMemcpyDeviceToHost);
    // auto timer_array_ptr1 = timer_DTC_array.request().ptr;
    // cudaMemcpy(timer_array_ptr1, timer_d1, sizeof(uint64_t) * row_num, cudaMemcpyDeviceToHost);
    // auto smid_array_ptr = smid_array.request().ptr;
    // cudaMemcpy(smid_array_ptr, smid_arr, sizeof(int) * row_num, cudaMemcpyDeviceToHost);
    // auto sm_order_array_ptr = sm_order_array.request().ptr;
    // cudaMemcpy(sm_order_array_ptr, sm_order, sizeof(int) * row_num, cudaMemcpyDeviceToHost);
    // auto sm_counter_array_ptr = sm_counter_array.request().ptr;
    // cudaMemcpy(sm_counter_array_ptr, sm_counter, sizeof(int) * 28, cudaMemcpyDeviceToHost);
    // auto read_latency_ptr = read_latency.request().ptr;
    // cudaMemcpy(read_latency_ptr, latency_record, sizeof(uint64_t) * write_len, cudaMemcpyDeviceToHost);
    // auto read_offset_ptr = read_offset.request().ptr;
    // cudaMemcpy(read_offset_ptr, thread_offset, sizeof(uint32_t) * 257, cudaMemcpyDeviceToHost);
    // auto read_token_ptr = read_token_.request().ptr;
    // cudaMemcpy(read_token_ptr, read_token, sizeof(uint8_t) * write_len, cudaMemcpyDeviceToHost);
///////////////////////////////////////////////
    // auto iter_test_ptr = iter_test.request().ptr;
    // cudaMemcpy(iter_test_ptr, iter_test_d, sizeof(int) * row_num, cudaMemcpyDeviceToHost);
///////////////////////////////////////////////
    auto dev_array_ptr = dev_array.request().ptr;
    cudaMemcpy(dev_array_ptr, dev_arr, sizeof(float) * row_num, cudaMemcpyDeviceToHost);
    // printf("after copy data\n");
    cudaFree(dev_arr);
    cudaFree(timer_d1);
    cudaFree(tcblock_offset);
    cudaFree(sparse_AToX_index);
    cudaFree(tcblocktile_id);
    // cudaFree(smid_arr);
    // cudaFree(iter_test_d);
    cudaFree(row_block_num_d);
    cudaFree(indice_d);
    cudaFree(indptr_d);
    cudaFree(blockPartution_d);
    cudaFree(edgeToColumn_d);
    cudaFree(edgeToRow_d);
    cudaFree(edgeattention_d);
    cudaFree(input);
    cudaFree(output);
    // cudaFree(timer_d);
    // cudaFree(duration);
    cudaFree(value);
    cudaFree(Sparsecol_AToX);
    cudaFree(mask);
    cudaFree(block_nonzero_num);
    cudaFree(block_offset);
    return {total_time, total_real};
}

void record_single_block_latency(
  py::array_t<int> indice,
  py::array_t<int> indptr,
  torch::Tensor blockPartition,
  torch::Tensor edgeToColumn,
  torch::Tensor edgeToRow,
  py::array_t<uint64_t> read_latency_,
  py::array_t<uint32_t> read_offset,
  py::array_t<uint8_t> read_token_,
  int record_block_id,
  int nodes_num,
  int edges_num,
  int block_num,
  int embedding_dim,
  int block_high,
  int block_width
) {
  int *indice_d, *indptr_d, *blockPartution_d, *edgeToColumn_d, *edgeToRow_d;
  uint32_t *row_block_num_d;
  int row_num = (nodes_num + block_high - 1) / block_high;
  cudaMalloc(&row_block_num_d, sizeof(uint32_t) * (row_num + 1));
  cudaMemset(row_block_num_d, 0, sizeof(uint32_t));
  cudaMalloc(&indice_d, sizeof(int) * edges_num);
  cudaMalloc(&indptr_d, sizeof(int) * (nodes_num + 1));
  int blockPartition_len = blockPartition.size(0);
  cudaMalloc(&blockPartution_d, sizeof(int) * row_num);
  cudaMalloc(&edgeToColumn_d, sizeof(int) * edges_num);
  cudaMalloc(&edgeToRow_d, sizeof(int) * edges_num);
  auto indice_ptr = indice.request().ptr;
  auto indptr_ptr = indptr.request().ptr;
  cudaMemcpy(indice_d, indice_ptr, sizeof(int) * edges_num, cudaMemcpyHostToDevice);
  cudaMemcpy(indptr_d, indptr_ptr, sizeof(int) * (nodes_num + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(blockPartution_d, blockPartition.data_ptr<int>(), sizeof(int) * blockPartition_len, cudaMemcpyHostToDevice);
  cudaMemcpy(edgeToColumn_d, edgeToColumn.data_ptr<int>(), sizeof(int) * edges_num, cudaMemcpyHostToDevice);
  cudaMemcpy(edgeToRow_d, edgeToRow.data_ptr<int>(), sizeof(int) * edges_num, cudaMemcpyHostToDevice);
  thrust::inclusive_scan(thrust::device, blockPartution_d, blockPartution_d + row_num, row_block_num_d + 1, 
                        [=]__device__(int a, uint32_t b) {return (uint32_t)(a + b);});
  float* edgeattention_d, *input, *output;
  cudaMalloc(&edgeattention_d, sizeof(float) * edges_num);
  cudaMalloc(&input, sizeof(float) * nodes_num * embedding_dim);
  cudaMalloc(&output, sizeof(float) * block_high * row_num * embedding_dim);
  thrust::fill_n(thrust::device, edgeattention_d, edges_num, 1.0f);
  thrust::fill_n(thrust::device, input, nodes_num * embedding_dim, 1.0f);
  int* tcblock_offset, *sparse_AToX_index;
  uint8_t* tcblocktile_id;
  cudaMalloc(&tcblock_offset, sizeof(int) * (block_num + 1));
  cudaMalloc(&sparse_AToX_index, sizeof(int) * block_num * block_width);
  cudaMalloc(&tcblocktile_id, sizeof(uint8_t) * edges_num);
  thrust::fill_n(thrust::device, tcblock_offset, block_num + 1, 0);
  thrust::fill_n(thrust::device, sparse_AToX_index, block_num * block_width, nodes_num);
  auto max = thrust::max_element(thrust::device, blockPartution_d, blockPartution_d + row_num);
  int max_blocks;
  cudaMemcpy(&max_blocks, max, sizeof(int), cudaMemcpyDeviceToHost);
  printf("max_blocks: %d\n", max_blocks);
  generate_tcoffset_id_atob_cuda(indptr_d, row_block_num_d, edgeToColumn_d, edgeToRow_d,
      indice_d, tcblock_offset + 1, tcblocktile_id, sparse_AToX_index, max_blocks, nodes_num,
      block_high, block_width, row_num);
  thrust::inclusive_scan(thrust::device, tcblock_offset + 1, tcblock_offset + 1 + block_num, tcblock_offset + 1);
  uint64_t *read_latency;
  uint8_t *read_token, *threadIdx_record;
  uint32_t *read_len;
  cudaMalloc(&read_latency, sizeof(uint64_t) * 3 * 1024 * 1024);
  cudaMalloc(&read_token, sizeof(uint8_t) * 3 * 1024 * 1024);
  cudaMalloc(&threadIdx_record, sizeof(uint8_t) * 3 * 1024 * 1024);
  cudaMalloc(&read_len, sizeof(uint32_t));
  const int WARPperblock = embedding_dim / block_high;
  printf("WARPperblock: %d\n", WARPperblock);
  dim3 grid_(row_num, 1, 1);
  dim3 block_(32, WARPperblock, 1);
  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_single<<<grid_, block_>>>(
      row_block_num_d, tcblocktile_id, tcblock_offset, sparse_AToX_index, edgeattention_d, nodes_num,
      edges_num, embedding_dim, input, output, read_latency, read_token, threadIdx_record, read_len,
      record_block_id);
  cudaDeviceSynchronize();
  uint32_t write_len;
  cudaMemcpy(&write_len, read_len, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  uint8_t* threadIdx_record_copy;
  cudaMalloc(&threadIdx_record_copy, sizeof(uint8_t) * write_len);
  thrust::copy(thrust::device, threadIdx_record, threadIdx_record + write_len, threadIdx_record_copy);
  thrust::stable_sort_by_key(thrust::device, threadIdx_record, threadIdx_record + write_len, read_latency);
  thrust::stable_sort_by_key(thrust::device, threadIdx_record_copy, threadIdx_record_copy + write_len, read_token);
  uint32_t* thread_counts, *counts_mask, *thread_offset;
  cudaMalloc(&counts_mask, sizeof(uint32_t) * write_len);
  cudaMalloc(&thread_counts, sizeof(uint32_t) * 64);
  cudaMalloc(&thread_offset, sizeof(uint32_t) * 65);
  thrust::fill_n(thrust::device, thread_counts, 64, 0);
  cudaMemset(thread_offset, 0, sizeof(uint32_t));
  thrust::fill_n(thrust::device, counts_mask, write_len, 1);
  thrust::reduce_by_key(thrust::device, threadIdx_record, threadIdx_record + write_len, counts_mask,
      threadIdx_record, thread_counts, thrust::equal_to<uint8_t>(), thrust::plus<uint32_t>());  
  thrust::inclusive_scan(thrust::device, thread_counts, thread_counts + 64, thread_offset + 1);
  auto read_latency_ptr = read_latency_.request().ptr;
  cudaMemcpy(read_latency_ptr, read_latency, sizeof(uint64_t) * write_len, cudaMemcpyDeviceToHost);
  auto read_offset_ptr = read_offset.request().ptr;
  cudaMemcpy(read_offset_ptr, thread_offset, sizeof(uint32_t) * 65, cudaMemcpyDeviceToHost);
  auto read_token_ptr = read_token_.request().ptr;
  cudaMemcpy(read_token_ptr, read_token, sizeof(uint8_t) * write_len, cudaMemcpyDeviceToHost);
  cudaFree(row_block_num_d);
  cudaFree(indice_d);
  cudaFree(indptr_d);
  cudaFree(blockPartution_d);
  cudaFree(edgeToColumn_d);
  cudaFree(edgeToRow_d);
  cudaFree(edgeattention_d);
  cudaFree(input);
  cudaFree(output);
  cudaFree(tcblock_offset);
  cudaFree(sparse_AToX_index);
  cudaFree(tcblocktile_id);
  cudaFree(read_latency);
  cudaFree(read_token);
  cudaFree(threadIdx_record);
  cudaFree(read_len);
}

std::vector<uint64_t> simulate(
  py::array_t<int> indice,
  py::array_t<int> indptr,
  torch::Tensor blockPartition,
  torch::Tensor edgeToColumn,
  torch::Tensor edgeToRow,
  int nodes_num,
  int edges_num,
  int block_num,
  int block_high,
  int block_width,
  int embedding_dim
) {
  int *indice_d, *indptr_d, *blockPartution_d, *edgeToColumn_d, *edgeToRow_d;
  uint32_t *row_block_num_d;
  int row_num = (nodes_num + block_high - 1) / block_high;
  cudaMalloc(&row_block_num_d, sizeof(uint32_t) * (row_num + 1));
  cudaMemset(row_block_num_d, 0, sizeof(uint32_t));
  cudaMalloc(&indice_d, sizeof(int) * edges_num);
  cudaMalloc(&indptr_d, sizeof(int) * (nodes_num + 1));
  int blockPartition_len = blockPartition.size(0);
  cudaMalloc(&blockPartution_d, sizeof(int) * row_num);
  cudaMalloc(&edgeToColumn_d, sizeof(int) * edges_num);
  cudaMalloc(&edgeToRow_d, sizeof(int) * edges_num);
  // auto row_block_num_ptr = row_block_num.request().ptr;
  auto indice_ptr = indice.request().ptr;
  auto indptr_ptr = indptr.request().ptr;
  // cudaMemcpy(row_block_num_d, row_block_num_ptr, sizeof(int) * row_num, cudaMemcpyHostToDevice);
  cudaMemcpy(indice_d, indice_ptr, sizeof(int) * edges_num, cudaMemcpyHostToDevice);
  cudaMemcpy(indptr_d, indptr_ptr, sizeof(int) * (nodes_num + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(blockPartution_d, blockPartition.data_ptr<int>(), sizeof(int) * blockPartition_len, cudaMemcpyHostToDevice);
  cudaMemcpy(edgeToColumn_d, edgeToColumn.data_ptr<int>(), sizeof(int) * edges_num, cudaMemcpyHostToDevice);
  cudaMemcpy(edgeToRow_d, edgeToRow.data_ptr<int>(), sizeof(int) * edges_num, cudaMemcpyHostToDevice);
  thrust::inclusive_scan(thrust::device, blockPartution_d, blockPartution_d + row_num, row_block_num_d + 1, 
                        [=]__device__(int a, uint32_t b) {return (uint32_t)(a + b);});
  float* edgeattention_d, *input, *output;
  cudaMalloc(&edgeattention_d, sizeof(float) * edges_num);
  cudaMalloc(&input, sizeof(float) * nodes_num * embedding_dim);
  cudaMalloc(&output, sizeof(float) * block_high * row_num * embedding_dim);
  thrust::fill_n(thrust::device, edgeattention_d, edges_num, 1.0f);
  thrust::fill_n(thrust::device, input, nodes_num * embedding_dim, 1.0f);
  uint64_t *duration;
  cudaMalloc(&duration, sizeof(uint64_t) * row_num);
  int* tcblock_offset, *sparse_AToX_index;
  uint8_t* tcblocktile_id;
  cudaMalloc(&tcblock_offset, sizeof(int) * (block_num + 1));
  cudaMalloc(&sparse_AToX_index, sizeof(int) * block_num * block_width);
  cudaMalloc(&tcblocktile_id, sizeof(uint8_t) * edges_num);
  // cudaMemset(tcblock_offset, 0, sizeof(int));
  thrust::fill_n(thrust::device, tcblock_offset, block_num + 1, 0);
  thrust::fill_n(thrust::device, sparse_AToX_index, block_num * block_width, nodes_num);
  auto max = thrust::max_element(thrust::device, blockPartution_d, blockPartution_d + row_num);
  int max_blocks;
  cudaMemcpy(&max_blocks, max, sizeof(int), cudaMemcpyDeviceToHost);
  printf("max_blocks: %d\n", max_blocks);
  generate_tcoffset_id_atob_cuda(indptr_d, row_block_num_d, edgeToColumn_d, edgeToRow_d,
      indice_d, tcblock_offset + 1, tcblocktile_id, sparse_AToX_index, max_blocks, nodes_num,
      block_high, block_width, row_num);
  thrust::inclusive_scan(thrust::device, tcblock_offset + 1, tcblock_offset + 1 + block_num, tcblock_offset + 1);
  int last_tcblock_offset;
  cudaMemcpy(&last_tcblock_offset, tcblock_offset + block_num, sizeof(int), cudaMemcpyDeviceToHost);
  printf("last_tcblock_offset: %d, edges_num: %d\n", last_tcblock_offset, edges_num);
  const int WARPperblock = embedding_dim / block_high;
  printf("WARPperblock: %d\n", WARPperblock);
  int* max_block_occupancy_per_SM = (int*)malloc(sizeof(int));
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(max_block_occupancy_per_SM, 
      spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer, 
      WARPperblock*32, 1088);
  printf("Max blocks per SM: %d\n", *max_block_occupancy_per_SM);
  dim3 grid_(row_num, 1, 1);
  dim3 block_(32, WARPperblock, 1);
  float kernel_elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaDeviceSynchronize();
  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer<<<grid_, block_>>>(
      row_block_num_d, tcblocktile_id, tcblock_offset, sparse_AToX_index, edgeattention_d, nodes_num,
      edges_num, embedding_dim, input, output);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernel_elapsed_time, start, stop);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  uint64_t total_real = deviceProp.clockRate * kernel_elapsed_time;
  printf("real cycle for kernel: %d cycles.\n", total_real);
  void *params[7] = {row_block_num_d, tcblocktile_id, tcblock_offset, sparse_AToX_index, edgeattention_d, input, output};
  int total_read_write = calculate_data_size(7, params);
  printf("total read write size: %d bytes\n", total_read_write);
  spmm_DTC_time_simulation_wrapper(
      row_block_num_d, tcblock_offset, duration, 28672, 3145728, total_read_write, row_num);
  uint64_t total_time;
  spmm_kernel_simulation(duration, *max_block_occupancy_per_SM, 28, row_num, total_time);
  printf("total time: %d cycles.\n", total_time);
  cudaFree(row_block_num_d);
  cudaFree(indice_d);
  cudaFree(indptr_d);
  cudaFree(blockPartution_d);
  cudaFree(edgeToColumn_d);
  cudaFree(edgeToRow_d);
  cudaFree(edgeattention_d);
  cudaFree(input);
  cudaFree(output);
  cudaFree(duration);
  cudaFree(tcblock_offset);
  cudaFree(sparse_AToX_index);
  cudaFree(tcblocktile_id);
  return {total_time, total_real};
}

PYBIND11_MODULE(sim_ext, m) {
    m.def("spmm_compare", &spmm_compare, "spmm_compare");
    m.def("simulate", &simulate, "simulate");
    m.def("record_single_block_latency", &record_single_block_latency, "record_single_block_latency");
};