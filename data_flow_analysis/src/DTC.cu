#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/unique.h>
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "DTC.cuh"

namespace py = pybind11;
//no balance, no reorder
std::vector<int> get_block_num_and_datasize(py::array_t<int64_t> &edge_list, int block_high, int block_width, size_t num_edges, size_t num_nodes, 
                        py::array_t<int> &row_block_num, py::array_t<int> &row_nonzero_offset, py::array_t<int64_t> &SparseCol_AtoX) 
{
    int64_t *edge_list_ptr = (int64_t*) edge_list.request().ptr;
    int64_t *edge_index_0, *edge_index_1;
    cudaMalloc((void**)&edge_index_0, num_edges * sizeof(int64_t));
    cudaMalloc((void**)&edge_index_1, num_edges * sizeof(int64_t));
    cudaMemcpy(edge_index_0, edge_list_ptr, num_edges * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_index_1, edge_list_ptr + num_edges, num_edges * sizeof(int64_t), cudaMemcpyHostToDevice);
    int64_t* value, *value_;
    cudaMalloc((void**)&value, num_edges * sizeof(int64_t));
    cudaMalloc((void**)&value_, num_edges * sizeof(int64_t));
    thrust::transform(thrust::device, edge_index_0, edge_index_0 + num_edges, edge_index_1, value, 
                            [=]__device__(int64_t a, int64_t b){return (b / (int64_t)block_high) * (int64_t)num_nodes + a;});
    thrust::copy(thrust::device, value, value + num_edges, value_);
    thrust::stable_sort_by_key(thrust::device, value, value + num_edges, edge_index_0);
    thrust::stable_sort_by_key(thrust::device, value_, value_ + num_edges, edge_index_1);
    thrust::transform(thrust::device, edge_index_1, edge_index_1 + num_edges, edge_index_1, 
                            [=]__device__(int64_t a){return (a / (int64_t)block_high);});
    auto new_end = thrust::unique_by_key(thrust::device, value, value + num_edges, edge_index_1);
    auto new_end_ = thrust::unique_by_key(thrust::device, value_, value_ + num_edges, edge_index_0);
    int nonzero_col = new_end.second - edge_index_1;
    cudaMemcpy(SparseCol_AtoX.request().ptr, edge_index_0, nonzero_col * sizeof(int64_t), cudaMemcpyDeviceToHost);
    int row_num = (num_nodes + block_high - 1) / block_high;
    int* mask, *nonzero_colnum, *row_offset, *row_offset_;
    cudaMalloc((void**)&mask, nonzero_col * sizeof(int));
    cudaMalloc((void**)&nonzero_colnum, nonzero_col * sizeof(int));
    cudaMalloc((void**)&row_offset, row_num * sizeof(int));
    cudaMalloc((void**)&row_offset_, row_num * sizeof(int));
    // int* row_block_offset;
    // cudaMalloc((void**)&row_block_offset, (row_num + 1) * sizeof(int));
    thrust::fill_n(thrust::device, row_offset, row_num, 0);
    thrust::fill_n(thrust::device, mask, nonzero_col, 1);
    auto end = thrust::reduce_by_key(thrust::device, edge_index_1, new_end.second, mask, edge_index_1, 
                            nonzero_colnum, thrust::equal_to<int64_t>(), thrust::plus<int>());
    thrust::scatter(thrust::device, nonzero_colnum, end.second, edge_index_1, row_offset);
    thrust::inclusive_scan(thrust::device, row_offset, row_offset + row_num, row_offset_);
    cudaMemcpy((int*)row_nonzero_offset.request().ptr + 1, row_offset_, row_num * sizeof(int), cudaMemcpyDeviceToHost);
    thrust::transform(thrust::device, row_offset, row_offset + row_num, row_offset, 
                            [=]__device__(int a){return (a + block_width - 1) / block_width;});
    int* row_block_num_ptr = (int*) row_block_num.request().ptr;
    cudaMemcpy(row_block_num_ptr, row_offset, row_num * sizeof(int), cudaMemcpyDeviceToHost);
    thrust::inclusive_scan(thrust::device, row_offset, row_offset + row_num, row_offset);
    // cudaDeviceSynchronize();
    int block_num;
    cudaMemcpy(&block_num, row_offset + row_num - 1, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("block_num: %d\n", block_num);
    // int compress_len = 4 * (row_num + 1) + nonzero_col_len * 4 + 4 * (block_num + 1) + num_edges * 5;
    cudaFree(edge_index_0);
    cudaFree(edge_index_1);
    cudaFree(value);
    cudaFree(value_);
    cudaFree(mask);
    cudaFree(nonzero_colnum);
    cudaFree(row_offset);
    // cudaFree(row_block_offset);
    return {nonzero_col, block_num};
}

PYBIND11_MODULE(ext, m) {
    m.def("get_block_num_and_datasize", &get_block_num_and_datasize, "get block number and data size");
}