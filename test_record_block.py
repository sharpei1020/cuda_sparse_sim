from compute_analysis.spmm import SpMM_Sim
import numpy as np

if __name__ == '__main__':
    # homo_dataset = ['OVCAR-8H',
    #                     'Yeast', 'DD', 'YeastH', 'amazon0505',
    #                     'artist', 'com-amazon', 'soc-BlogCatalog',
    #                     'amazon0601']
    
    # for dataset in homo_dataset:
        spmm_sim = SpMM_Sim('OVCAR-8H', 32)
        out = spmm_sim.record_latency("DTC")

