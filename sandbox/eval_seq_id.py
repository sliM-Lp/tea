
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
BF16_FASTA_PATH = "/scicore/home/schwede/pudziu0000/projects/tea/sandbox/multidomain/nonquantized_r1/sequences_tea.fasta"

fp32_sequences, fp32_entropies = get_sequences(FP32_FASTA_PATH)
bf16_sequences, bf16_entropies = get_sequences(BF16_FASTA_PATH)

identities = []
lengths = []
fp32_entropies_list = []
bf16_entropies_list = []
delta_entropies = []

num_in_tail = 0
num_in_tail_delta_entropy = 0

for key in fp32_sequences.keys():
    
    fp32_seq = fp32_sequences[key]
    bf16_seq = bf16_sequences[key]

    matches = sum(a == b for a, b in zip(fp32_seq, bf16_seq) if a != '-' and b != '-')

    identity = matches/len(fp32_seq)*100
    identities.append(identity)
    lengths.append(len(fp32_seq))
    fp32_entropies_list.append(fp32_entropies[key])
    bf16_entropies_list.append(bf16_entropies[key])
    delta_entropy = fp32_entropies[key] - bf16_entropies[key]
    delta_entropies.append(delta_entropy)

    if(identity < 90):
        num_in_tail += 1

    if(delta_entropy < -0.1 or delta_entropy > 0.1):
        num_in_tail_delta_entropy += 1

print(f"Number of sequences in tail (identity < 90%): {num_in_tail}")
print(f"Number of sequences in tail (delta entropy < -0.1 or delta entropy > 0.1): {num_in_tail_delta_entropy}")

fp32_res = stats.spearmanr(identities, fp32_entropies_list)
print(f"Spearman correlation between identities and FP32 entropies:")
print(fp32_res.statistic, fp32_res.pvalue)

print(f"Spearman correlation between identities and BF16 entropies:")
bf16_res = stats.spearmanr(identities, bf16_entropies_list)
print(bf16_res.statistic, bf16_res.pvalue)

print(f"Spearman correlation between FP32 and BF16 entropies:")
entropies_res = stats.spearmanr(fp32_entropies_list, bf16_entropies_list)
print(entropies_res.statistic, entropies_res.pvalue)

print(f"Spearman correlation between identities and delta entropies:")
delta_entropies_res = stats.spearmanr(identities, delta_entropies)
print(delta_entropies_res.statistic, delta_entropies_res.pvalue)

print(f"Spearman correlation between delta entropies and sequence lengths:")
delta_length_res = stats.spearmanr(delta_entropies, lengths)
print(delta_length_res.statistic, delta_length_res.pvalue)

fig, ax = plt.subplots(4, 1, figsize=(8, 12))

plot_scatter(
    ax[0], 
    identities, 
    delta_entropies, 
    xlabel='Sequence Identity (%)', 
    ylabel='Delta Entropy (FP32 - nonquantized)', 
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
    ylabel='Nonquantized Entropy', 
    spearman_rho=bf16_res.statistic, 
    pvalue=bf16_res.pvalue,
    hline=0.25
)

plot_scatter(
    ax[3],
    lengths, 
    delta_entropies, 
    xlabel='Sequence Length', 
    ylabel='Delta Entropy (FP32 - nonquantized)', 
    spearman_rho=delta_length_res.statistic,
    pvalue=delta_length_res.pvalue
)

plt.suptitle('Sequence Identity vs Entropy')
plt.tight_layout()
plt.savefig('identity_vs_delta_entropy.png', dpi=300)





    






