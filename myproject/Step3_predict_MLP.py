"""
predict_mlp.py
============================================================
Use trained MLP models to predict on the test set, output:
  1. predicted_matrix.feather  (n_test_cells x n_genes) predicted values
  2. true_input_matrix.feather  (n_test_cells x n_genes) target_test values

These files can be directly used for reconstructed-vs-original
correlation plots downstream.

Usage:
  python predict_mlp.py \
    --model_dir results/saved_models \
    --tf_test split_data/tf_test.feather \
    --target_test split_data/target_test.feather \
    --out_pred results/predicted.feather \
    --out_truth results/true_input.feather \
    --norm_factor no_norm \
    --this_trans_factor sqrt+1 \
    --compare_combinations_save_path results/compare_combinations_trans_norm.csv
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import argparse
import os
import json

from scipy.stats import pearsonr, spearmanr


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h)


def _apply_trans(arr, trans):
    arr = np.asarray(arr, dtype=np.float32)

    if trans == "no_trans":
        return arr.copy()
    elif trans == "sqrt":
        return np.sqrt(np.clip(arr, 0, None))
    elif trans == "sqrt+1":
        return np.sqrt(np.clip(arr, 0, None) + 1)
    elif trans == "sqrt+0.00001":
        return np.sqrt(np.clip(arr, 0, None) + 0.00001)
    elif trans == "sqrt+10":
        return np.sqrt(np.clip(arr, 0, None) + 10)
    elif trans == "sqrt+1_then_minus_1":
        return np.sqrt(np.clip(arr, 0, None) + 1) - 1
    elif trans == "log2":
        return np.log2(np.clip(arr, 0, None) + 1)
    elif trans == "log1p":
        return np.log1p(np.clip(arr, 0, None))
    elif trans == "log2_then_add_1":
        return np.log2(np.clip(arr, 0, None) + 1) + 1
    elif trans == "count+1":
        return arr + 1
    elif trans == "log2(count+2)":
        return np.log2(np.clip(arr, 0, None) + 2)
    elif trans == "log2(count+1)+1":
        return np.log2(np.clip(arr, 0, None) + 1) + 1
    else:
        raise ValueError(f"Unknown transformation: {trans}")


def compute_matrix_correlation(pred_df, truth_df):
    pearson_list = []
    spearman_list = []

    common_genes = [g for g in pred_df.columns if g in truth_df.columns]

    for gene in common_genes:
        pred_arr = pred_df[gene].to_numpy(dtype=np.float64)
        truth_arr = truth_df[gene].to_numpy(dtype=np.float64)

        keep = np.isfinite(pred_arr) & np.isfinite(truth_arr)
        pred_arr = pred_arr[keep]
        truth_arr = truth_arr[keep]

        if len(pred_arr) == 0:
            continue

        if np.std(pred_arr) < 1e-10 or np.std(truth_arr) < 1e-10:
            continue

        p, _ = pearsonr(pred_arr, truth_arr)
        s, _ = spearmanr(pred_arr, truth_arr)

        if not np.isnan(p):
            pearson_list.append(p)
        if not np.isnan(s):
            spearman_list.append(s)

    if len(pearson_list) == 0:
        mean_pearson = 0.0
    else:
        mean_pearson = float(np.mean(pearson_list))

    if len(spearman_list) == 0:
        mean_spearman = 0.0
    else:
        mean_spearman = float(np.mean(spearman_list))

    return mean_pearson, mean_spearman

def append_compare_result(compare_path, trans, norm_factor, pearson_corr, spearman_corr):
    row_df = pd.DataFrame([{
        "method": "mlp_TF_gene",
        "trans": trans,
        "norm_factor": norm_factor,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr
    }])

    out_dir = os.path.dirname(compare_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(compare_path):
        old_df = pd.read_csv(compare_path)
        out_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        out_df = row_df

    out_df.to_csv(compare_path, index=False)


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    print("[INFO] Loading test data...")
    tf_test = pd.read_feather(args.tf_test)
    target_test = pd.read_feather(args.target_test)

    for col in ["pos", "index", "Unnamed: 0"]:
        if col in tf_test.columns:
            tf_test.drop(columns=[col], inplace=True)
        if col in target_test.columns:
            target_test.drop(columns=[col], inplace=True)

    X_test_np = tf_test.values.astype(np.float32)
    gene_names = list(target_test.columns)
    n_cells = X_test_np.shape[0]

    print(f"  Test cells: {n_cells}")
    print(f"  TF features: {X_test_np.shape[1]}")
    print(f"  Target genes: {len(gene_names)}")

    # Find all trained gene models
    model_dir = args.model_dir
    available_genes = []

    for gene_name in gene_names:
        gene_dir = os.path.join(model_dir, gene_name)
        config_path = os.path.join(gene_dir, "config.json")
        model_path = os.path.join(gene_dir, "model.pt")

        if os.path.exists(config_path) and os.path.exists(model_path):
            available_genes.append(gene_name)

    print(f"\n[INFO] Found models for {len(available_genes)}/{len(gene_names)} genes")

    if len(available_genes) == 0:
        print("[ERROR] No models found! Please check --model_dir path")
        return

    # Predict per gene
    pred_dict = {}

    for i, gene_name in enumerate(available_genes):
        gene_dir = os.path.join(model_dir, gene_name)

        # Read config
        with open(os.path.join(gene_dir, "config.json"), "r") as f:
            config = json.load(f)

        trans = config.get("trans", "no_trans")
        h1 = int(config.get("h1", 256))
        h2 = int(config.get("h2", 128))
        input_dim = int(config["input_dim"])
        bs = int(config.get("bs", 64))

        # Transform input
        X_transformed = _apply_trans(X_test_np, trans)
        X_t = torch.tensor(X_transformed, dtype=torch.float32)

        if X_t.shape[1] != input_dim:
            raise ValueError(
                f"Input dimension mismatch for gene {gene_name}: "
                f"model expects {input_dim}, but tf_test has {X_t.shape[1]}"
            )

        # Load model
        model = MLP(input_dim, h1, h2, dropout=0.0).to(device)
        model.load_state_dict(
            torch.load(
                os.path.join(gene_dir, "model.pt"),
                map_location=device,
                weights_only=True
            )
        )
        model.eval()

        # Predict
        test_loader = DataLoader(
            TensorDataset(X_t),
            batch_size=bs,
            shuffle=False
        )

        all_pred = []

        with torch.no_grad():
            for (x,) in test_loader:
                x = x.to(device)
                pred = model(x)
                all_pred.append(pred.cpu().numpy())

        pred_arr = np.concatenate(all_pred).flatten()
        pred_dict[gene_name] = pred_arr

        if (i + 1) % 100 == 0 or (i + 1) == len(available_genes):
            print(f"  Predicted {i + 1}/{len(available_genes)} genes...")

    # Ensure output directories exist
    for p in [args.out_pred, args.out_truth]:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    # 1. Predicted matrix
    pred_df = pd.DataFrame(pred_dict, columns=available_genes)
    pred_df.to_feather(args.out_pred)

    # 2. True target matrix, same gene order as predicted
    truth_df = target_test[available_genes].copy()
    truth_df = pd.DataFrame(
        _apply_trans(truth_df.to_numpy(dtype=np.float32), args.this_trans_factor),
        columns=available_genes
    )
    truth_df.to_feather(args.out_truth)
    
    # 3. Correlation between predicted and true target matrix
    pearson_corr, spearman_corr = compute_matrix_correlation(pred_df, truth_df)

    append_compare_result(
        compare_path=args.compare_combinations_save_path,
        trans=args.this_trans_factor,
        norm_factor=args.norm_factor,
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr
    )

    print(f"\n{'=' * 60}")
    print("Prediction Summary:")
    print(f"  Genes predicted: {len(available_genes)}")
    print(f"  Cells: {n_cells}")
    print(f"  Pearson corr: {pearson_corr:.6f}")
    print(f"  Spearman corr: {spearman_corr:.6f}")
    print(f"  Compare summary: {args.compare_combinations_save_path}")

    print(f"\nOutput files:")
    print(f"  {args.out_pred}")
    print(f"  {args.out_truth}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict test set using trained MLP models"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to saved_models directory"
    )
    parser.add_argument(
        "--tf_test",
        type=str,
        required=True,
        help="TF test feather file"
    )
    parser.add_argument(
        "--target_test",
        type=str,
        required=True,
        help="Target gene test feather file"
    )
    parser.add_argument(
        "--out_pred",
        type=str,
        required=True,
        help="Output path for predicted matrix (.feather)"
    )
    parser.add_argument(
        "--out_truth",
        type=str,
        required=True,
        help="Output path for true target matrix (.feather)"
    )
    parser.add_argument(
        "--norm_factor",
        type=str,
        required=True,
        help="Normalization factor used for this run"
    )
    parser.add_argument(
        "--this_trans_factor",
        type=str,
        required=True,
        help="Transformation factor used for this run"
    )
    parser.add_argument(
        "--compare_combinations_save_path",
        type=str,
        required=True,
        help="CSV path to append MLP correlation result"
    )
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)