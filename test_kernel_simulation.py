from compute_analysis.spmm import SpMM_Sim
from matplotlib import pyplot as plt
import numpy as np
import math

if __name__ == '__main__':
    homo_dataset = ['citeseer', 'cora', 'pubmed', 
                        'ppi', 'PROTEINS_full', 'OVCAR-8H',
                        'Yeast', 'DD', 'YeastH', 'amazon0505',
                        'artist', 'com-amazon', 'soc-BlogCatalog',
                        'amazon0601']
    model = "DTC"
    simulate_list0 = []
    real_list0 = []
    simulate_list1 = []
    real_list1 = []
    for dataset in homo_dataset:
        spmm_simulation = SpMM_Sim(dataset, 32)
        out = spmm_simulation.simulate_time("TCGNN")
        simulate_list0.append(out[2])
        real_list0.append(out[3])
        print(f"TCGNN {dataset} utiliation: {out[4]} %, {out[5]} %")
        out = spmm_simulation.simulate_time("DTC")
        simulate_list1.append(out[1])
        real_list1.append(out[2])
        print(f"DTC {dataset} utiliation: {out[3]} %, {out[4]} %")
    sim = np.array(simulate_list0 + simulate_list1, dtype=np.int64)
    sim_mean = sim.mean()
    real = np.array(real_list0 + real_list1, dtype=np.int64)
    real_mean = real.mean()
    diff = np.absolute(sim - real)
    MEA = (diff / real).mean() 
    upper = np.dot(sim - sim_mean, real - real_mean).sum().item()
    downer = math.sqrt(np.dot(sim - sim_mean, sim - sim_mean).item() * np.dot(real - real_mean, real - real_mean).item())
    print("MEA: {:.4f}, R: {:.4f}".format(MEA, upper / downer))
    fig, ax = plt.subplots()
    ax.scatter(simulate_list0, real_list0)
    ax.scatter(simulate_list1, real_list1)
    ax.set_xlabel("Simulated Time (cycles)")
    ax.set_ylabel("Real Time (cycles)")
    plt.savefig(f"img/spmm_{model}_simulation.png")
    plt.show()