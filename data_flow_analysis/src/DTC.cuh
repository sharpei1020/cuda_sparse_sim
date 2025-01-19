#ifndef DTC_CUH
#define DTC_CUH
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<int> get_block_num_and_datasize(py::array_t<int64_t> &edge_list, int block_high, int block_width, size_t num_edges, size_t num_nodes, 
                        py::array_t<int> &row_block_num, py::array_t<int> &row_nonzero_offset, py::array_t<int64_t> &SparseCol_AtoX); 

#endif // DTC_CUH