
import os
import sys
from Bio import SeqIO
from Bio import Align
from Bio.Align import substitution_matrices
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def get_sequences(fasta_path):
    """
    Extract sequences (and optionally headers) from a FASTA file.
    
    Args:
        fasta_path (str): Path to the FASTA file.
        return_headers (bool): If True, also return headers dict.
    
    Returns:
        dict or tuple: seq_dict, or (seq_dict, headers) if return_headers=True.
    """
    seq_dict = {}
    entropies = {}
    
    with open(fasta_path) as f:
        for record in SeqIO.parse(f, 'fasta'):
            header, entropy = record.description.split('|')
            seq_dict[header] = str(record.seq)
            entropies[header] = float(entropy.replace('H=', ''))
    
    return (seq_dict, entropies)

def plot_scatter(
    ax,
    x, 
    y, 
    xlabel='Sequence Identity (%)', 
    ylabel='Delta Entropy (FP32 - BF16)',
    spearman_rho=None,
    pvalue=None,
    hline=None
):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x, y, marker='.', c=z, cmap='viridis')
    #ax.scatter(x, y, marker='.', color='navy')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if(hline):
        ax.axhline(hline, color='black', linestyle='--', lw=1, alpha=0.7)

    if(spearman_rho is not None and pvalue is not None):
        ax.text(
            0.05, 
            0.95, 
            f'Spearman r={spearman_rho:.2f}, p={pvalue:.2e}', 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5)
        )



FP32_FASTA_PATH = "/scicore/home/schwede/pudziu0000/projects/tea/sandbox/multidomain/fp32_r1/sequences_tea.fasta"
BF16_FASTA_PATH = "/scicore/home/schwede/pudziu0000/projects/tea/sandbox/multidomain/bf16_r1/sequences_tea.fasta"

fp32_sequences, fp32_entropies = get_sequences(FP32_FASTA_PATH)
bf16_sequences, bf16_entropies = get_sequences(BF16_FASTA_PATH)

identities = []
fp32_entropies_list = []
bf16_entropies_list = []
delta_entropies = []

for key in fp32_sequences.keys():
    
    fp32_seq = fp32_sequences[key]
    bf16_seq = bf16_sequences[key]

    matches = sum(a == b for a, b in zip(fp32_seq, bf16_seq) if a != '-' and b != '-')

    identity = matches/len(fp32_seq)*100
    identities.append(identity)
    fp32_entropies_list.append(fp32_entropies[key])
    bf16_entropies_list.append(bf16_entropies[key])
    delta_entropies.append(fp32_entropies[key] - bf16_entropies[key])

fp32_res = stats.spearmanr(identities, fp32_entropies_list)
print(fp32_res.statistic, fp32_res.pvalue)

bf16_res = stats.spearmanr(identities, bf16_entropies_list)
print(bf16_res.statistic, bf16_res.pvalue)

entropies_res = stats.spearmanr(fp32_entropies_list, bf16_entropies_list)
print(entropies_res.statistic, entropies_res.pvalue)

delta_entropies_res = stats.spearmanr(identities, delta_entropies)
print(delta_entropies_res.statistic, delta_entropies_res.pvalue)

fig, ax = plt.subplots(3, 1, figsize=(8, 12))

plot_scatter(
    ax[0], 
    identities, 
    delta_entropies, 
    xlabel='Sequence Identity (%)', 
    ylabel='Delta Entropy (FP32 - BF16)', 
    spearman_rho=delta_entropies_res.statistic, 
    pvalue=delta_entropies_res.pvalue
)

plot_scatter(
    ax[1], 
    identities, 
    fp32_entropies_list, 
    xlabel='Sequence Identity (%)', 
    ylabel='FP32 Entropy', 
    spearman_rho=fp32_res.statistic, 
    pvalue=fp32_res.pvalue,
    hline=0.25
)

plot_scatter(
    ax[2], 
    identities, 
    bf16_entropies_list, 
    xlabel='Sequence Identity (%)', 
    ylabel='BF16 Entropy', 
    spearman_rho=bf16_res.statistic, 
    pvalue=bf16_res.pvalue,
    hline=0.25
)

plt.suptitle('Sequence Identity vs Entropy')
plt.tight_layout()
plt.savefig('identity_vs_delta_entropy.png', dpi=300)





    






