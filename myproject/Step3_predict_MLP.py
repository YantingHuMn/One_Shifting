"""
predict_mlp.py
============================================================
Use trained MLP models to predict on the test set, output:
  1. predicted_matrix.feather  (n_test_cells x n_genes) predicted values
  2. true_input_matrix.feather  (n_test_cells x n_genes) target_test values
  3. ground_truth_matrix.feather  (n_test_cells x n_genes) ground truth values

These files can be directly used for correlation plots downstream.

Usage:
  python predict_mlp.py \
    --model_dir results/saved_models \
    --tf_test split_data/tf_test.feather \
    --target_test split_data/target_test.feather \
    --ground_test split_data/ground_truth_test.feather \
    --out_pred results/predicted.feather \
    --out_truth results/true_input.feather \
    --out_ground_truth results/ground_truth.feather
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


def apply_trans(arr, trans):
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
        raise ValueError(f"Unknown transform: {trans}")
    

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    print("[INFO] Loading test data...")
    tf_test = pd.read_feather(args.tf_test)
    target_test = pd.read_feather(args.target_test)
    ground_test = pd.read_feather(args.ground_test)

    for col in ['pos', 'index', 'Unnamed: 0']:
        if col in tf_test.columns:
            tf_test.drop(columns=[col], inplace=True)
        if col in target_test.columns:
            target_test.drop(columns=[col], inplace=True)
        if col in ground_test.columns:
            ground_test.drop(columns=[col], inplace=True)

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
    pred_dict = {} # gene_name -> predicted values (n_cells,)

    for i, gene_name in enumerate(available_genes):
        gene_dir = os.path.join(model_dir, gene_name)

        # Read config
        with open(os.path.join(gene_dir, "config.json"), "r") as f:
            config = json.load(f)

        trans = config.get('trans', 'no_trans')
        h1 = int(config.get('h1', 256))
        h2 = int(config.get('h2', 128))
        input_dim = int(config['input_dim'])
        bs = int(config.get('bs', 64))

        # Transform input
        X_transformed = apply_trans(X_test_np, trans)
        X_t = torch.tensor(X_transformed, dtype=torch.float32)

        # Load model
        model = MLP(input_dim, h1, h2, dropout=0.0).to(device)  # no dropout at prediction time
        model.load_state_dict(torch.load(
            os.path.join(gene_dir, "model.pt"),
            map_location=device, weights_only=True
        ))
        model.eval()

        # Predict
        test_loader = DataLoader(
            TensorDataset(X_t, torch.zeros(X_t.shape[0], 1)),  # dummy y
            batch_size=bs, shuffle=False
        )

        all_pred = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                pred = model(x)
                all_pred.append(pred.cpu().numpy())

        pred_arr = np.concatenate(all_pred).flatten()

        pred_dict[gene_name] = pred_arr

        if (i + 1) % 100 == 0 or (i + 1) == len(available_genes):
            print(f"  Predicted {i+1}/{len(available_genes)} genes...")

    # Save results 
    # Ensure output directories exist
    for p in [args.out_pred, args.out_truth, args.out_ground_truth]:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    # 1. Predicted matrix (columns ordered by available_genes)
    pred_df = pd.DataFrame(pred_dict, columns=available_genes)
    pred_df.to_feather(args.out_pred)

    # 2. True input matrix (target_test, same gene order as predicted)
    truth_df = target_test[available_genes].copy()
    truth_df.to_feather(args.out_truth)

    # 3. Ground truth matrix (same gene order as predicted)
    ground_genes = [g for g in available_genes if g in ground_test.columns]
    ground_df = ground_test[ground_genes].copy()
    ground_df.to_feather(args.out_ground_truth)

    print(f"\n{'='*60}")
    print("Prediction Summary:")
    print(f"  Genes predicted: {len(available_genes)}")
    print(f"  Genes in ground truth: {len(ground_genes)}")
    print(f"  Cells: {n_cells}")

    print(f"\nOutput files:")
    print(f"  {args.out_pred}")
    print(f"  {args.out_truth}")
    print(f"  {args.out_ground_truth}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict test set using trained MLP models"
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help="Path to saved_models directory")
    parser.add_argument('--tf_test', type=str, required=True,
                        help="TF test feather file")
    parser.add_argument('--target_test', type=str, required=True,
                        help="Target gene test feather file")
    parser.add_argument('--ground_test', type=str, required=True,
                        help="Ground truth test feather file")
    parser.add_argument('--out_pred', type=str, required=True,
                        help="Output path for predicted matrix (.feather)")
    parser.add_argument('--out_truth', type=str, required=True,
                        help="Output path for true input matrix (.feather)")
    parser.add_argument('--out_ground_truth', type=str, required=True,
                        help="Output path for ground truth matrix (.feather)")
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    main(args)