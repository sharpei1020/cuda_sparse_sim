from compute_analysis.spmm import SpMM_Sim
from data_flow_analysis.data import TCGNN_dataset
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    # homo_dataset = ['citeseer', 'cora', 'pubmed', 
    #                     'ppi', 'PROTEINS_full', 'OVCAR-8H',
    #                     'Yeast', 'DD', 'YeastH', 'amazon0505',
    #                     'artist', 'com-amazon', 'soc-BlogCatalog',
    #                     'amazon0601']
    # for dataset in homo_dataset:
    spmm_simulation = SpMM_Sim('cora', 32)
    # time_sim, time_real_tcgnn, time_real_dtc, smid, dev, row_blocks_num, sm_order, sm_counter
    # 0           1               2               3    4    5              6         7
    out = spmm_simulation.simulate_time("TCGNN")
    large_gap = {}
    gaps = {}
    workloads = {}
    large_gap_workloads = {}
    for i in range(28):
        if not (i in large_gap.keys()):
            large_gap[i] = []
            gaps[i] = []
            workloads[i] = 0
            large_gap_workloads[i] = 0
    row_num = len(out[0])
    for i in range(len(out[0])):
        gap = None
        gaps[out[3][i]].append(i)
        workloads[out[3][i]] += out[5][i]
        if out[1][i] > out[0][i]:
            gap = out[1][i] - out[0][i]
        else:
            gap = out[0][i] - out[1][i]
        if gap > 100000:
            large_gap[out[3][i]].append((i, out[5][i], out[6][i]))
            large_gap_workloads[out[3][i]] += out[5][i]
    for g in large_gap.keys():
        print(f"{g} : {large_gap[g]}, {large_gap_workloads[g]} / {out[7][g]}")
    # print(time_sim[52468], time_real_tcgnn[52468])

