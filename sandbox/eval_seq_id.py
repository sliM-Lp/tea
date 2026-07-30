
import os
import sys
from Bio import SeqIO
from Bio import Align
from Bio.Align import substitution_matrices
from scipy import stats

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

FP32_FASTA_PATH = "/scicore/home/schwede/pudziu0000/projects/tea/sandbox/multidomain/fp32_r1/sequences_tea.fasta"
BF16_FASTA_PATH = "/scicore/home/schwede/pudziu0000/projects/tea/sandbox/multidomain/bf16_r1/sequences_tea.fasta"

fp32_sequences, fp32_entropies = get_sequences(FP32_FASTA_PATH)
bf16_sequences, bf16_entropies = get_sequences(BF16_FASTA_PATH)

identities = []
fp32_entropies_list = []
bf16_entropies_list = []

for key in fp32_sequences.keys():
    
    fp32_seq = fp32_sequences[key]
    bf16_seq = bf16_sequences[key]

    matches = sum(a == b for a, b in zip(fp32_seq, bf16_seq) if a != '-' and b != '-')

    identity = matches/len(fp32_seq)*100
    identities.append(identity)
    fp32_entropies_list.append(fp32_entropies[key])
    bf16_entropies_list.append(bf16_entropies[key])

fp32_res = stats.spearmanr(identities, fp32_entropies_list)
print(fp32_res.statistic, fp32_res.pvalue)

bf16_res = stats.spearmanr(identities, bf16_entropies_list)
print(bf16_res.statistic, bf16_res.pvalue)

entropies_res = stats.spearmanr(fp32_entropies_list, bf16_entropies_list)
print(entropies_res.statistic, entropies_res.pvalue)





    






