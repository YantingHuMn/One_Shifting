# One Shifting Substantially Improves Deep Learning Model’s Performance on Sparse Data

A computational pipeline for reconstructing and evaluating single-cell count matrices using deep generative models (VAE, DCA, scVI). Designed for epigenomic data from ChIP-seq and CUT&Tag experiments.

## Overview

One_Shifting trains variational autoencoders on count matrices derived from epigenomic assays, reconstructs held-out data, and evaluates reconstruction quality through gene-wise and cell-wise correlation analysis against ground truth replicates.

## Pipeline

```
Step 1: Build count matrix with QC (R)
    ↓
Step 2: Train model — VAE / DCA / scVI (Python)
    ↓
Step 3: Reconstruct count matrix (Python)
    ↓
Step 4: Evaluate via correlation scatter plots (R)
```

## Workflow Details

### 1. Count Matrix Construction (R)

Builds count matrices from epigenomic data (HiPlex, PBMC) with quality control filtering.

### 2. Model Training (Python)

Trains deep generative models with nested cross-validation. 

Configurable hyperparameters include architecture (hidden dims, latent dim, layers), transformation (log1p, sqrt, etc.), beta (KL weight), nonzero weight, and threshold.

### 3. Reconstruction (Python)

Reconstructs count matrices from the best-fold model selected by validation loss/metric.

### 4. Evaluation (R)

Computes Pearson and Spearman correlations between reconstructed outputs and ground truth, with options to filter ground-truth all-zero features. Generates scatter plots comparing reconstruction vs. original input correlations, with diagonal reference lines and highlighted feature pairs (e.g., H3K4me3-H3K27me3).

## Requirements

### Python
- PyTorch
- scvi-tools
- pandas, numpy, pyarrow

### R (>= 4.0.0)
- arrow
- dplyr
- ggplot2
- jsonlite

## Usage

