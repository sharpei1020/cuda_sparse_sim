from data_flow_analysis.data import TCGNN_dataset, get_CSR_nonzero_len, get_SGT_nonzeroblock_len
import TCGNN
import torch
import numpy as np
import math


class SpMM_Sim:
    def __init__(self, dataset, feat_dim):
        super(SpMM_Sim, self).__init__()
        self.dataset = dataset
        self.feat_dim = feat_dim

    def simulate_time(self, model_name):
        dataset = TCGNN_dataset(self.dataset)
        dataset.init_edges()
        indices, indptr = dataset.get_CSR_data()
        nodes_num = dataset.get("nodes_num")
        edge_index_0 = dataset.get("edge_index_0")
        edge_index_1 = dataset.get("edge_index_1")
        nonzero, compress, row_block_num, row_nonzero_offset, SparseCol_AtoX = get_SGT_nonzeroblock_len(edge_index_0, \
                                        edge_index_1, nodes_num, 16, 8, 'TCGNN')
        MAC_counts = len(edge_index_0) * 2 * self.feat_dim
        valid_compute_cycles = MAC_counts / 64 / 4 / 28
        if model_name == "TCGNN":
            # 1 SM for 6 blocks
            num_row_windows = (nodes_num + 16 - 1) // 16
            edgeToColumn = torch.zeros(len(edge_index_0), dtype=torch.int)
            edgeToRow = torch.zeros(len(edge_index_0), dtype=torch.int)
            blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
            TCGNN.preprocess(torch.IntTensor(indices), torch.IntTensor(indptr), \
                            nodes_num, 16, 8, blockPartition, edgeToColumn, edgeToRow)
            # print(edgeToColumn, SparseCol_AtoX)
            
            import sim_ext
            # print(row_block_num.sum().item())
            time_sim = np.zeros(len(row_block_num), dtype=np.uint64)
            time_real_tcgnn = np.zeros(len(row_block_num), dtype=np.uint64)
            time_real_dtc = np.zeros(len(row_block_num), dtype=np.uint64)
            smid = np.zeros(len(row_block_num), dtype=np.int32)
            sm_order = np.zeros(len(row_block_num), dtype=np.int32)
            sm_counter = np.zeros(28, dtype=np.int32)
            # read_latency = np.zeros(3 * 1024 * 1024, dtype=np.uint64)
            # thread_offset = np.zeros(257, dtype=np.uint32)
            # read_token = np.zeros(3 * 1024 * 1024, dtype=np.uint8)
            ########################################
            # iter_test = np.zeros(len(row_block_num), dtype=np.int32)
            dev = np.zeros(len(row_block_num), dtype=np.float32)
            # print(len(row_block_num)) read_latency, thread_offset, read_token,
            simulate, real = sim_ext.spmm_compare(row_block_num, indices, indptr, 
                                #  time_sim, time_real_tcgnn, \
                                # time_real_dtc, smid, sm_order, sm_counter, \
                                dev, blockPartition, edgeToColumn, edgeToRow, nodes_num, len(edge_index_0), \
                                blockPartition.sum().item(), self.feat_dim, 16, 8)
            utilization_real = (valid_compute_cycles / real) * 100
            utiliation_sim = (valid_compute_cycles / simulate) * 100
            # row_num = len(row_block_num) - 100
            # print(indptr[row_num*16], indptr[row_num*16+16])
            # neighbors = set()
            # for i in range(16):
            #     start = indptr[row_num*16+i]
            #     end = indptr[row_num*16+i+1]
            #     for idx in indices[start:end]:
            #         neighbors.add(idx)

            # sorted_neighbors = sorted(neighbors)
            # for i in range(math.ceil(len(sorted_neighbors) / 8)):
            #     print(np.array(sorted_neighbors[i*8:min((i+1)*8, len(sorted_neighbors))]).var())
            # print(sorted(neighbors), len(neighbors))
            # # print(thread_offset, len(thread_offset))
            # for i in range(64):
            #     print("(", i, "),", " ".join(f"{n:5d}" for n in read_latency[thread_offset[i]:thread_offset[i+1]]))
            #     print("(", i, "),", " ".join(f"{n:5d}" for n in read_token[thread_offset[i]:thread_offset[i+1]]))
            # print(smid[:200])
            # print(dev[:50])
            # print(iter_test[:50])
            # print(blockPartition[:50].numpy())
            # print(time_sim[:50])
            # print(time_real_tcgnn[:50])
            # print(time_real_dtc[:50])
            ##########################################
            # test_condition = np.abs(iter_test - blockPartition.numpy()) > 0
            # if test_condition.any():
            #     print("Error: iter_test and blockPartition do not match!")
            ##########################################
            # time_real = time_real.reshape(-1, 2)[:, 0]
            return dev, blockPartition.numpy(), simulate, real, utiliation_sim, utilization_real
            return time_sim, time_real_tcgnn, time_real_dtc, smid, dev, blockPartition.numpy(), sm_order, sm_counter
        if model_name == "DTC":
            # 1 SM for 6 blocks
            num_row_windows = (nodes_num + 16 - 1) // 16
            edgeToColumn = torch.zeros(len(edge_index_0), dtype=torch.int)
            edgeToRow = torch.zeros(len(edge_index_0), dtype=torch.int)
            blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
            TCGNN.preprocess(torch.IntTensor(indices), torch.IntTensor(indptr), \
                            nodes_num, 16, 8, blockPartition, edgeToColumn, edgeToRow)
            # print(edgeToColumn, SparseCol_AtoX)
            
            import sim_ext
            simulate, real = sim_ext.simulate(indices, indptr, 
                                blockPartition, edgeToColumn, edgeToRow, nodes_num, len(edge_index_0), \
                                blockPartition.sum().item(), 16, 8, self.feat_dim)
            utilization_real = (valid_compute_cycles / real) * 100
            utiliation_sim = (valid_compute_cycles / simulate) * 100
            return blockPartition.numpy(), simulate, real, utiliation_sim, utilization_real

    def record_latency(self, model_name):
        dataset = TCGNN_dataset(self.dataset)
        dataset.init_edges()
        indices, indptr = dataset.get_CSR_data()
        nodes_num = dataset.get("nodes_num")
        edge_index_0 = dataset.get("edge_index_0")
        edge_index_1 = dataset.get("edge_index_1")
        if model_name == "DTC":
            num_row_windows = (nodes_num + 16 - 1) // 16
            edgeToColumn = torch.zeros(len(edge_index_0), dtype=torch.int)
            edgeToRow = torch.zeros(len(edge_index_0), dtype=torch.int)
            blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
            TCGNN.preprocess(torch.IntTensor(indices), torch.IntTensor(indptr), \
                            nodes_num, 16, 8, blockPartition, edgeToColumn, edgeToRow)
            import sim_ext
            read_latency = np.zeros(3 * 1024 * 1024, dtype=np.uint64)
            thread_offset = np.zeros(65, dtype=np.uint32)
            read_token = np.zeros(3 * 1024 * 1024, dtype=np.uint8)
            record_block_id = 1
            sim_ext.record_single_block_latency(
               indices, indptr, blockPartition, edgeToColumn, edgeToRow, 
               read_latency, thread_offset, read_token, record_block_id,
               nodes_num, len(edge_index_0), blockPartition.sum().item(), 
               self.feat_dim, 16, 8)
            row_num = record_block_id
            print(indptr[row_num*16], indptr[row_num*16+16])
            neighbors = set()
            for i in range(16):
                start = indptr[row_num*16+i]
                end = indptr[row_num*16+i+1]
                for idx in indices[start:end]:
                    neighbors.add(idx)

            sorted_neighbors = sorted(neighbors)
            for i in range(math.ceil(len(sorted_neighbors) / 8)):
                print(np.array(sorted_neighbors[i*8:min((i+1)*8, len(sorted_neighbors))]).var())
            print(sorted(neighbors), len(neighbors))
            # print(thread_offset, len(thread_offset))
            for i in range(64):
                print("(", i, "),", " ".join(f"{n:5d}" for n in read_latency[thread_offset[i]:thread_offset[i+1]]))
                print("(", i, "),", " ".join(f"{n:5d}" for n in read_token[thread_offset[i]:thread_offset[i+1]]))
            return read_latency, thread_offset, read_token


