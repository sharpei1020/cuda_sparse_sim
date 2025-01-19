import numpy as np
from data import TCGNN_dataset
import ext

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    homo_dataset = {'CR':'citeseer', 'CO':'cora', 'PB':'pubmed', 
                        'PI':'ppi', 'PR':'PROTEINS_full', 'OV':'OVCAR-8H',
                        'YT':'Yeast', 'DD':'DD', 'YH':'YeastH', 'AZ':'amazon0505',
                        'AT':'artist', 'CA':'com-amazon', 'SC':'soc-BlogCatalog',
                        'AO':'amazon0601'}
    print(homo_dataset.keys())
    block_high = 16
    block_width = 8
    dim_len = [32, 64, 128, 256, 512, 1024]
    for i, dimlen in enumerate(dim_len):
        fused_global_trans = []
        no_fused_global_trans = []
        for dataset in homo_dataset.values():
            data = TCGNN_dataset(dataset)
            data.init_edges()
            edge_index_0 = data.get('edge_index_0')
            edge_index_1 = data.get('edge_index_1')
            nodes_num = data.get('nodes_num')
            edge_index = np.stack((edge_index_0, edge_index_1), dtype=np.int64)
            edge_len = len(edge_index_0)
            row_num = (nodes_num + block_high - 1) // block_high
            row_block_num = np.zeros(row_num, dtype=np.int32)
            row_nonzero_offset = np.zeros(row_num + 1, dtype=np.int32)
            SparseCol_AtoX = np.zeros(edge_len, dtype=np.int64)
            nonzero, block_num = ext.get_block_num_and_datasize(edge_index, block_high, block_width,
                                edge_len, nodes_num, row_block_num, row_nonzero_offset, SparseCol_AtoX)
            no_fused = ((row_num + 1) + edge_len * 4 + row_num + nodes_num * dimlen + nonzero * dimlen) + \
                        (edge_len * 2) + (edge_len * 4 + (row_num + 1) + row_num + nonzero * dimlen + nodes_num * dimlen) 
            no_fused_global_trans.append(1)
            fused = nodes_num * dimlen * 2 + nonzero * dimlen + (row_num + 1) + row_num + 3 * edge_len
            # print(f'{dimlen}, {dataset}, {nodes_num * dimlen * 2}, {nonzero * dimlen}, {(row_num + 1) + row_num + 3 * edge_len}')
            # print(f'{dimlen}, {dataset}, {no_fused / fused}')
            fused_global_trans.append(fused / no_fused)
        x = np.arange(len(homo_dataset))
        y0 = np.array(no_fused_global_trans)
        y1 = np.array(fused_global_trans)
        plt.subplot(2, 3, i+1)
        plt.bar(x, y0, width=0.2, label='no fused')
        plt.bar(x, y1, width=0.2, label='fused')
        plt.xlabel(f'Dim{dimlen}-Blockhigh16', fontdict={'size': 10})
        plt.xticks(x+0.2, homo_dataset.keys())
        plt.ylabel('Global memory access')
    plt.legend()
    # plt.savefig('../img/GAT_Global_memory_access(fused/no_fused).png')
    plt.show()