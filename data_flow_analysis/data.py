import torch
import numpy as np
import os
from scipy.sparse import coo_matrix

# tcgnn-ae-graphs
class TCGNN_dataset(torch.nn.Module):
    def __init__(self, dataset_name):
        super(TCGNN_dataset, self).__init__()

        self.data_path = os.path.join("/home/ljq/mine/graphiler/examples/AGNN/tcgnn-ae-graphs", dataset_name + ".npz")
        self.nodes_num = None
        self.edge_index_0 = None
        self.edge_index_1 = None

    def init_edges(self):
        fp = np.load(self.data_path)
        self.edge_index_0 = fp['src_li']
        self.edge_index_1 = fp['dst_li']
        self.nodes_num = fp['num_nodes'].item()
    
    def get_CSR_data(self):
        edge_index = np.stack((self.edge_index_0, self.edge_index_1), dtype= np.int64)
        val = np.ones(len(self.edge_index_0), dtype=np.float32)
        adj = coo_matrix((val, edge_index), shape=(self.nodes_num, self.nodes_num))
        return adj.tocsr().indices, adj.tocsr().indptr

    def get(self, str):
        return getattr(self, str)

#SNAP-dataset   
class SNAP_dataset(torch.nn.Module):
    def __init__(self, dataset_name):
        super(SNAP_dataset, self).__init__()
        self.data_path = os.path.join("/home/ljq/mine/sim_modules/cuda-sparse-sim/data_flow_analysis/dataset", dataset_name + ".txt")
        self.nodes_num = None
        self.edge_index_0 = None
        self.edge_index_1 = None

    def init_edges(self):
        edges = np.array(np.loadtxt(self.data_path), dtype=np.int32)
        self.edge_index_0 = edges[:, 0] - 1
        self.edge_index_1 = edges[:, 1] - 1
        self.nodes_num = np.max(edges)

def get_CSR_nonzero_len(edge_index_0, edge_index_1, nodes_num):
    edge_index = np.stack((edge_index_0, edge_index_1), dtype= np.int64)
    val = np.ones(len(edge_index_0), dtype=np.float32)
    adj = coo_matrix((val, edge_index), shape=(nodes_num, nodes_num))
    data = adj.tocsr().data
    inptr = adj.tocsr().indptr
    indices = adj.tocsr().indices
    csr_len = data.dtype.itemsize * data.shape[0] + inptr.dtype.itemsize * inptr.shape[0] + indices.dtype.itemsize * indices.shape[0]
    return adj.tocsr().data.shape[0], csr_len
    
def get_SGT_nonzeroblock_len(edge_index_0, edge_index_1, nodes_num, block_high, block_width, model_name):
    edge_index = np.stack((edge_index_0, edge_index_1), dtype=np.int64)
    edge_len = len(edge_index_0)
    import ext
    row_num = (nodes_num + block_high - 1) // block_high
    row_block_num = np.zeros(row_num, dtype=np.int32)
    row_nonzero_offset = np.zeros(row_num + 1, dtype=np.int32)
    SparseCol_AtoX = np.zeros(edge_len, dtype=np.int64)
    nonero, block_num = ext.get_block_num_and_datasize(edge_index, block_high, block_width, \
                                                       edge_len, nodes_num, row_block_num, row_nonzero_offset, SparseCol_AtoX)
    # print(nonero, block_num, row_block_num)
    if model_name == 'TCGNN':
        compress = (nodes_num + 1) * 4 + edge_len * 4 * 4 + 4 * ((nodes_num + block_high - 1) // block_high + 1)
    if model_name == 'DTC':
        compress = 4 * ((nodes_num + block_high - 1) // block_high + 1) + block_num * 32 + 4 * (block_num + 1) + edge_len * 5
    if model_name == 'Ours':
        compress = 8 * ((nodes_num + block_high - 1) // block_high + 1) + nonero * 4 + 16 * (block_num + 1) + edge_len * 4
    return nonero, compress, row_block_num, row_nonzero_offset, SparseCol_AtoX[:nonero]

def get_CSR_compute_memoryaccess_rate(edge_index_0, nodes_num, embeddding_dim, mode):
    if mode == 'spmm':
        memory_access = (nodes_num + len(edge_index_0)) * (embeddding_dim + 2)
        compute = len(edge_index_0) * 2 * embeddding_dim
        return compute / memory_access, compute / memory_access

def get_SGT_compute_memoryaccess_rate(edge_index_0, edge_index_1, nodes_num, block_high, block_width, embedding_dim, mode, model_name):
    edge_index = np.stack((edge_index_0, edge_index_1), dtype=np.int64)
    edge_len = len(edge_index_0)
    row_num = (nodes_num + block_high - 1) // block_high
    row_block_num = np.zeros(row_num, dtype=np.int32)
    row_nonzero_offset = np.zeros(row_num + 1, dtype=np.int32)
    SparseCol_AtoX = np.zeros(edge_len, dtype=np.int64)
    import ext
    nonero, block_num = ext.get_block_num_and_datasize(edge_index, block_high, block_width, \
                                                       edge_len, nodes_num, row_block_num, row_nonzero_offset, SparseCol_AtoX)
    if mode == 'spmm':
        if model_name == 'TCGNN':
            memory_access = 2 * row_num + 3 * len(edge_index_0) + nonero * embedding_dim + \
                np.dot(row_block_num, row_nonzero_offset[1:] - row_nonzero_offset[:-1]) + nodes_num * embedding_dim
            compute = 2 * block_high * block_width * block_num * embedding_dim
            valid_compte = len(edge_index_0) * 2 * embedding_dim
            return compute / memory_access, valid_compte / memory_access
        if model_name == 'DTC':
            memory_access = 2 * (row_num + block_num) + 3 * len(edge_index_0) + nonero * embedding_dim + nodes_num * embedding_dim
            compute = 2 * block_high * block_width * block_num * embedding_dim
            valid_compte = len(edge_index_0) * 2 * embedding_dim
            return compute / memory_access, valid_compte / memory_access


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    homo_dataset = {'CR':'citeseer', 'CO':'cora', 'PB':'pubmed', 
                        'PI':'ppi', 'PR':'PROTEINS_full', 'OV':'OVCAR-8H',
                        'YT':'Yeast', 'DD':'DD', 'YH':'YeastH', 'AZ':'amazon0505',
                        'AT':'artist', 'CA':'com-amazon', 'SC':'soc-BlogCatalog',
                        'AO':'amazon0601'}
    print(homo_dataset.keys())
    # block_len = [2, 4, 8, 16, 32, 64]
    dim_len = [32, 64, 128, 256, 512, 1024]
    # for i, block_high in enumerate(block_len):
    for i, dimlen in enumerate(dim_len):
        CSR_norm = []
        SGT_norm = []
        DTC_norm = []
        for dataset in homo_dataset.values():
            data= TCGNN_dataset(dataset)
            data.init_edges()
            edge_index_0 = data.get('edge_index_0')
            edge_index_1 = data.get('edge_index_1')
            nodes_num = data.get('nodes_num')
            # compute_rate0, valid_compute_rate0 = get_CSR_compute_memoryaccess_rate(edge_index_0, nodes_num, 32, 'spmm')
            # compute_rate1, valid_compute_rate1 = get_SGT_compute_memoryaccess_rate(edge_index_0, edge_index_1, nodes_num, block_high, 8, 32, 'spmm', 'TCGNN')
            # compute_rate2, valid_compute_rate2 = get_SGT_compute_memoryaccess_rate(edge_index_0, edge_index_1, nodes_num, block_high, 8, 32, 'spmm', 'DTC')
            compute_rate0, valid_compute_rate0 = get_CSR_compute_memoryaccess_rate(edge_index_0, nodes_num, dimlen, 'spmm')
            compute_rate1, valid_compute_rate1 = get_SGT_compute_memoryaccess_rate(edge_index_0, edge_index_1, nodes_num, 16, 8, dimlen, 'spmm', 'TCGNN')
            compute_rate2, valid_compute_rate2 = get_SGT_compute_memoryaccess_rate(edge_index_0, edge_index_1, nodes_num, 16, 8, dimlen, 'spmm', 'DTC')
            CSR_norm.append(valid_compute_rate0)
            SGT_norm.append(valid_compute_rate1)
            DTC_norm.append(valid_compute_rate2)
            print("(", valid_compute_rate0, valid_compute_rate1, ") ratio: ", valid_compute_rate0/valid_compute_rate1, "(", valid_compute_rate0, valid_compute_rate2, ") ratio: ", valid_compute_rate0/valid_compute_rate2)
        x = np.arange(len(homo_dataset))
        y0 = np.array(CSR_norm)
        y1 = np.array(SGT_norm)
        y2 = np.array(DTC_norm)
        plt.subplot(2, 3, i+1)
        plt.bar(x, y0, width=0.2, label='CSR')
        plt.bar(x+1*0.2, y1, width=0.2, label='SGT-valid')
        plt.bar(x+2*0.2, y2, width=0.2, label='DTC-valid')
        # plt.xlabel(f'Dim32-Blockhigh{block_high}', fontdict={'size': 10})
        plt.xlabel(f'Dim{dimlen}-Blockhigh16', fontdict={'size': 10})
        plt.xticks(x+0.2, homo_dataset.keys())
        plt.ylabel('Compute Memory Access Rate')
    plt.legend()
    plt.savefig('../img/different_block_high_homo_dataset_compute_memory_access_rate.png')
    plt.show()

    # dim_len = [32, 64, 128, 256]
    # work_mode = 'compute memory access rate'
    # for i, dimlen in enumerate(dim_len):
    #     CSR_norm = []
    #     SGT_norm = []
    #     DTC_norm = []
    #     # DTC_norm328 = []
    #     Ours_norm = []
    #     for dataset in homo_dataset:
    #         data= TCGNN_dataset(dataset)
    #         data.init_edges()
    #         edge_index_0 = data.get('edge_index_0')
    #         edge_index_1 = data.get('edge_index_1')
    #         nodes_num = data.get('nodes_num')
    #         if work_mode == 'compress':
    #             nonzero0, compress0 = get_CSR_nonzero_len(edge_index_0, edge_index_1, nodes_num)
    #             nonzero1, compress1, row_block_num, row_nonzero_offset, SparseCol_AtoX = get_SGT_nonzeroblock_len(edge_index_0, edge_index_1, nodes_num, 16, 8, 'TCGNN')
    #             nonzero2, compress2, row_block_num, row_nonzero_offset, SparseCol_AtoX = get_SGT_nonzeroblock_len(edge_index_0, edge_index_1, nodes_num, 16, 8, 'DTC')
    #             nonzero3, compress3, row_block_num, row_nonzero_offset, SparseCol_AtoX = get_SGT_nonzeroblock_len(edge_index_0, edge_index_1, nodes_num, 16, 8, 'Ours')
    #             CSR_norm.append(1)
    #             SGT_norm.append(compress1/compress0)
    #             DTC_norm.append(compress2/compress0)
    #             Ours_norm.append(compress3/compress0)
    #             # print(row_nonzero_offset, SparseCol_AtoX)
    #             print("(", nonzero0, nonzero3, ") ratio: ", nonzero0/nonzero3, "(", compress0, compress3, ") ratio: ", compress0/compress3)
    #         if work_mode == 'compute memory access rate':
    #             compute_rate0, valid_compute_rate0 = get_CSR_compute_memoryaccess_rate(edge_index_0, nodes_num, dimlen, 'spmm')
    #             compute_rate1, valid_compute_rate1 = get_SGT_compute_memoryaccess_rate(edge_index_0, edge_index_1, nodes_num, 16, 8, dimlen, 'spmm', 'TCGNN')
    #             compute_rate2, valid_compute_rate2 = get_SGT_compute_memoryaccess_rate(edge_index_0, edge_index_1, nodes_num, 16, 8, dimlen, 'spmm', 'DTC')
    #             # compute_rate3, valid_compute_rate3 = get_SGT_compute_memoryaccess_rate(edge_index_0, edge_index_1, nodes_num, 32, 8, 128, 'spmm', 'DTC')
    #             CSR_norm.append(compute_rate0)
    #             SGT_norm.append(valid_compute_rate1)
    #             DTC_norm.append(valid_compute_rate2)
    #             # DTC_norm328.append(valid_compute_rate3)
    #             print("(", compute_rate0, valid_compute_rate1, ") ratio: ", compute_rate0/valid_compute_rate1, "(", compute_rate0, valid_compute_rate2, ") ratio: ", compute_rate0/valid_compute_rate2)


    #     x = np.arange(len(homo_dataset))
    #     y0 = np.array(CSR_norm)
    #     y1 = np.array(SGT_norm)
    #     y2 = np.array(DTC_norm)
    #     # y3 = np.array(DTC_norm328)
    #     # y3 = np.array(Ours_norm)
    #     plt.subplot(1, 4, i+1)
    #     plt.bar(x, y0, width=0.2, label='CSR')
    #     plt.bar(x+1*0.2, y1, width=0.2, label='SGT-valid')
    #     plt.bar(x+2*0.2, y2, width=0.2, label='DTC-valid')
    #     # plt.bar(x+3*0.2, y3, width=0.2, label='DTC-valid-48')
    #     # plt.bar(x+3*0.2, y3, width=0.2, label='Ours')
    #     plt.xlabel('Dataset', fontdict={'size': 1})
    #     plt.xticks(x+0.2, homo_dataset)
    # if work_mode == 'compress':
    #     plt.ylabel('Compression Ratio')
    #     plt.title('Compression Ratio of Homograph Datasets')
    #     plt.legend()
    #     plt.savefig('../img/homo_dataset_compress_ratio.png')
    # if work_mode == 'compute memory access rate':
    #     plt.ylabel('Compute Memory Access Rate')
    #     plt.title('Compute Memory Access Rate of Homograph Datasets')
    #     plt.legend()
    #     plt.savefig('../img/homo_dataset_compute_memory_access_rate.png')
    # plt.show()

