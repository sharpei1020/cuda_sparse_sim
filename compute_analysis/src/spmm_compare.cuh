#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

std::vector<uint64_t> spmm_compare(
   py::array_t<int> row_block_num,
   py::array_t<int> indice,
   py::array_t<int> indptr,
   // py::array_t<uint64_t> duration_array,
   // py::array_t<uint64_t> timer_TCGNN_array,
   // py::array_t<uint64_t> timer_DTC_array,
   // py::array_t<int> smid_array,
   // py::array_t<int> sm_order_array,
   // py::array_t<int> sm_counter_array,
   // py::array_t<uint64_t> read_latency,
   // py::array_t<uint32_t> read_offset,
   // py::array_t<uint8_t> read_token_,
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
);

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
);

void record_single_block_latency(
  py::array_t<int> indice,
  py::array_t<int> indptr,
  torch::Tensor blockPartition,
  torch::Tensor edgeToColumn,
  torch::Tensor edgeToRow,
  py::array_t<uint64_t> read_latency,
  py::array_t<uint32_t> read_offset,
  py::array_t<uint8_t> read_token_,
  int record_block_id,
  int nodes_num,
  int edges_num,
  int block_num,
  int embedding_dim,
  int block_high,
  int block_width
);