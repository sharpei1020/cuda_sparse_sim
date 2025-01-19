from compute_analysis.spmm import SpMM_Sim
from data_flow_analysis.data import TCGNN_dataset
from matplotlib import pyplot as plt
import numpy as np
import math
import os

if __name__ == '__main__':
    homo_dataset = ['OVCAR-8H',
                        'Yeast', 'DD', 'YeastH', 'amazon0505',
                        'artist', 'com-amazon', 'soc-BlogCatalog',
                        'amazon0601']
    # 'citeseer', 'cora', 'pubmed', 
                        # 'ppi', 'PROTEINS_full', 
    for dataset in homo_dataset:
        # data = TCGNN_dataset(dataset)
        # data.init_edges()
        # fig0, ax0 = plt.subplots()
        # ax0.scatter(data.edge_index_0, data.edge_index_1, c='tab:red')
        spmm_simulation = SpMM_Sim(dataset, 32)
        # time_sim, time_real_tcgnn, time_real_dtc, smid, dev, row_blocks_num, sm_order, sm_counter
        # 0           1               2               3    4    5              6         7
        out = spmm_simulation.simulate_time("TCGNN")
        large_gap = []
        smid_x = np.arange(0, 28)
        smid_array = [0] * 28
        dev_mean = np.mean(out[4]).item()
        time_sim_sub_mean = out[0] - out[0].mean()
        time_real_sub_mean = out[1] - out[1].mean()
        time_sim_2 = np.dot(time_sim_sub_mean, time_sim_sub_mean).item()
        time_real_2 = np.dot(time_real_sub_mean, time_real_sub_mean).item()
        r = np.dot(time_sim_sub_mean, time_real_sub_mean).item() / math.sqrt(time_real_2 * time_sim_2)
        tmp = np.absolute(np.array(out[0], dtype=np.int32)-np.array(out[1], dtype=np.int32))
        print("{}'s error rate:{:.4f} %, latency error mean {} cycles, Pearson correlation coefficient is {:.4f}".\
              format(dataset, np.mean(tmp / out[1]).item() * 100, np.mean(tmp).item(), r))
        for i in range(len(out[0])):
            gap = None
            if out[1][i] > out[0][i]:
                gap = out[1][i] - out[0][i]
            else:
                gap = out[0][i] - out[1][i]
            if gap > 100000:
                large_gap.append((i, out[0][i], out[1][i], out[5][i], out[6][i]))
                smid_array[out[3][i]] += 1
        for it in large_gap:
            print(it)
        fig, ax = plt.subplots(1, 2)
        ax[1].bar(smid_x, smid_array, width=1)
        ax[1].set_xticks(smid_x)
        # os._exit(0)
        start = len(out[1]) // 2
        # x = np.arange(0, len(time_real))
        ax[0].scatter(out[1], out[0], c='tab:blue')
        ax[0].scatter(out[2], out[0], c='tab:orange')
        # ax.scatter(x, time_real, c='tab:orange', label='Real Time')
        ax[0].legend()
        ax[0].grid(True)
        plt.savefig("img/spmm_time_sim.png")
        plt.show()
