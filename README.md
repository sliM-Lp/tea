# The Embedding-based Alphabet (TEA)

![Model Architecture](Model_Architecture.png)

This repository contains the code accompanying our pre-print (link coming soon).

It includes:
- [Sequence conversion with AlphaBeta](#sequence-conversion-with-alphabeta)
- [Sequence searches with MMseqs2](#running-searches-with-MMseqs2)

## Environment setup

Download [mamba](https://github.com/conda-forge/miniforge#mambaforge)

```bash
chmod +x Mambaforge.sh
./Mambaforge.sh
```

```bash
module load CUDA/12.1.1
mamba create -n tea
mamba activate tea
mamba install pip
mamba install -c bioconda usalign

pip install lightning torch torch-geometric tensorboard nbformat tqdm wandb pandas biopython scikit-learn numba matplotlib seaborn 'jsonargparse[signatures]' transformers bitsandbytes peft pyarrow sentencepiece deepspeed fair-esm biotite

pip install .
```

# Sequence Conversion with TEA

The `convert_sequences.py` script takes protein sequences from a FASTA file and generates new tea-FASTA. It supports confidence-based sequence output where low-confidence positions are displayed in lowercase.

Alphabeta is derived from ESM2 embeddings, which are used for the sequence conversion. The conversion script relies on the ESM-2 model `esm2_t33_650M_UR50D` to generate embeddings. If you have an active internet connection, the script will automatically download the ESM2 model as needed. If not, you will need to download the model manually and cache it locally before running the conversion. For example:

```python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')" ```

## Parameters

### Required
- `--fasta_file`: Input FASTA file containing protein sequences
- `--output_file`: Output FASTA file for generated sequences

### Optional
- `--checkpoint_path`: Path to AlphaBeta model checkpoint (.ckpt file)
- `--include_confidence`: Enable confidence-based lowercase output
- `--entropy_threshold`: Entropy threshold for lowercase conversion (default: 0.3)
- `--mask_prob`: Probability of masking input tokens (default: 0.0)
- `--save_per_residue_logits`: Save per-residue logits to .pt file
- `--save_avg_entropy`: Save average entropy values to .pt file
- `--estimate_runtime`: Estimate processing time without running conversion
- `--timing_results_file`: Path to timing results for batch size optimization

### Output Files
- `output_sequences.fasta`: Generated protein sequences
- `output_sequences.fasta.masked`: Number of masked tokens per sequence (if `--mask_prob > 0`)
- `output_sequences_logits.pt`: Per-residue logits (if `--save_per_residue_logits`)
- `output_sequences_avg_entropy.pt`: Average entropy values (if `--save_avg_entropy`)

### Basic Usage
```bash
python convert_sequences.py \
    --fasta_file input_sequences.fasta \
    --output_file output_sequences.fasta
```

### Installing MMseqs2
```bash
mamba install -c conda-forge -c bioconda mmseqs2
```

#### Basic usage

```bash
mmseqs easy-search query_alphabeta.fasta target_alphabeta.fasta results.m8 /tmp/mmseqs_tmp \
    --comp-bias-corr 0 \
    --mask 0 \
    --gap-open 12 \
    --gap-extend 1 \
    --mask-lower-case 1 \
    -s 5.7 \
    --max-seqs 2000 \
    -e 10000 \
    --sub-mat alphabeta_matrix_sub_matrix.out \
    --seed-sub-mat alphabeta_matrix_sub_matrix.out \
    -v 2
```

Use as substitution matrix `models/alphabeta_sub_matrix.out`. Add the flag `--exact-kmer-matching 1` for speed improvement with no significant loss in performance expected.