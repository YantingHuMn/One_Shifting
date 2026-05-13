#!/usr/bin/env python3
import argparse
import os
import json
import glob
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, n_tokens=64, d_model=128,
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.n_tokens = n_tokens
        self.d_model = d_model

        self.chunk_size = math.ceil(input_dim / n_tokens)
        self.padded_input_dim = self.chunk_size * n_tokens

        self.input_proj = nn.Linear(self.chunk_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Sequential(
            nn.Linear(n_tokens * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.size(0)

        if self.input_dim < self.padded_input_dim:
            padding = torch.zeros(batch_size, self.padded_input_dim - self.input_dim,
                                  device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)

        x = x.view(batch_size, self.n_tokens, self.chunk_size)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, -1)
        out = self.output_proj(x)
        return out


def apply_transformation(df, trans):
    df_copy = df.copy()
    if trans == "sqrt+1":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:] + 1)
    elif trans == "sqrt":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:])
    elif trans == "log2":
        df_copy.iloc[:, 1:] = np.log2(df_copy.iloc[:, 1:] + 1)
    elif trans == "log2_then_add_1":
        df_copy.iloc[:, 1:] = np.log2(df_copy.iloc[:, 1:] + 1) + 1
    elif trans == "sqrt+0.00001":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:] + 0.00001)
    elif trans == "sqrt+10":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:] + 10)
    elif trans == "sqrt+1_then_minus_1":
        df_copy.iloc[:, 1:] = np.sqrt(df_copy.iloc[:, 1:] + 1) - 1
    elif trans == "count+1":
        df_copy.iloc[:, 1:] = df_copy.iloc[:, 1:] + 1
    elif trans == "log2(count+2)":
        df_copy.iloc[:, 1:] = np.log2(df_copy.iloc[:, 1:] + 2)
    elif trans == "log2(count+1)+1":
        df_copy.iloc[:, 1:] = np.log2(df_copy.iloc[:, 1:] + 1) + 1
    elif trans == "no_trans":
        pass
    else:
        raise ValueError(f"Unknown transformation: {trans}")
    return df_copy


def filter_and_transform(df1, threshold_value, trans1,
                         transformed_out_dir=None, data_path1=None, save=False):
    data_cols = df1.columns[1:]
    zero_percentage = (df1[data_cols] == 0).mean()
    keep_cols = zero_percentage < threshold_value

    pos_col = df1.columns[0]
    cols_to_keep = [pos_col] + data_cols[keep_cols].tolist()

    filtered_df1 = df1[cols_to_keep].copy()

    filtered_df1 = apply_transformation(filtered_df1, trans1)

    first_col_name = filtered_df1.columns[0]
    filtered_df1 = filtered_df1.rename(columns={first_col_name: 'pos'})

    if save and transformed_out_dir is not None:
        transformed_out_dir = Path(transformed_out_dir)
        transformed_out_dir.mkdir(parents=True, exist_ok=True)
        if data_path1 is not None:
            save_path1 = transformed_out_dir / "Count_matrix_transformed_rep1.feather"
            filtered_df1.to_feather(save_path1)
            print(f"[SAVE] Saved transformed df1 ({trans1}) to: {save_path1}")

    return filtered_df1


def find_weights_file(directory):
    """Find model weights file (.pt or .pth) in directory."""
    for ext in ("*.pt", "*.pth"):
        matches = glob.glob(os.path.join(directory, ext))
        if matches:
            if len(matches) > 1:
                print(f"  [WARNING] Multiple weight files found in {directory}, using: {matches[0]}")
            return matches[0]
    raise FileNotFoundError(f"No .pt or .pth file found in {directory}")


def find_config_file(directory):
    """Find config file (.json) in directory."""
    matches = glob.glob(os.path.join(directory, "*.json"))
    if matches:
        if len(matches) > 1:
            print(f"  [WARNING] Multiple JSON files found in {directory}, using: {matches[0]}")
        return matches[0]
    raise FileNotFoundError(f"No .json file found in {directory}")


def find_best_fold(saved_models_dir, criterion="inner_val_loss"):
    """Automatically find best fold."""
    fold_dirs = glob.glob(os.path.join(saved_models_dir, "fold_*"))
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {saved_models_dir}")

    best_fold = None
    best_score = float("inf")

    for fold_dir in fold_dirs:
        try:
            config_path = find_config_file(fold_dir)
        except FileNotFoundError:
            print(f"Warning: no JSON config in {fold_dir}, skipping")
            continue

        with open(config_path, "r") as f:
            cfg = json.load(f)

        score = cfg.get(criterion, float("inf"))
        fold_name = os.path.basename(fold_dir)
        print(f"{fold_name}: {criterion}={score:.6f}")

        if score != score:  # NaN check
            if best_fold is None:
                best_fold = fold_dir
        elif score < best_score:
            best_score = score
            best_fold = fold_dir

    if best_fold is None:
        raise ValueError("No valid fold found")

    print(f"\n[BEST] {os.path.basename(best_fold)} with {criterion}={best_score:.6f}")
    return best_fold


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Locate model weights and config
    final_weights = None
    final_config = None
    for ext in ("*.pt", "*.pth"):
        for f in glob.glob(os.path.join(args.saved_models_dir, ext)):
            if "final" in os.path.basename(f).lower():
                final_weights = f
                break
        if final_weights:
            break
    for f in glob.glob(os.path.join(args.saved_models_dir, "*.json")):
        if "final" in os.path.basename(f).lower():
            final_config = f
            break

    if final_weights and final_config:
        print(f"[INFO] Using final model: {os.path.basename(final_weights)}")
        model_path = final_weights
        config_path = final_config
    else:
        print("[INFO] Final model not found, using best fold")
        fold_dir = find_best_fold(args.saved_models_dir, args.criterion)
        model_path = find_weights_file(fold_dir)
        config_path = find_config_file(fold_dir)
        print(f"  weights: {os.path.basename(model_path)}")
        print(f"  config:  {os.path.basename(config_path)}")

    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    print(f"\n[CONFIG]")
    print(f"  threshold: {cfg['threshold']}")

    if 'trans1' in cfg:
        trans1 = cfg['trans1']
        print(f"  trans1: {trans1}")
    elif 'trans' in cfg:
        trans1 = cfg['trans']
        print(f"  trans (legacy): {trans1}")
    else:
        raise ValueError("Configuration must contain 'trans1' or 'trans'")

    print(f"  architecture: n_tokens={cfg['n_tokens']}, d_model={cfg['d_model']}, "
          f"nhead={cfg['nhead']}, num_layers={cfg['num_layers']}")

    # Load and preprocess data
    print("\n[LOAD DATA]")
    df1_raw = pd.read_feather(args.data_path1)
    print(f"  Original df1: {df1_raw.shape}")

    print("\n[PREPROCESS] Applying filter_and_transform...")
    df1_transformed = filter_and_transform(
        df1_raw,
        cfg['threshold'],
        trans1,
        transformed_out_dir=args.transformed_out_dir,
        data_path1=args.data_path1,
        save=True
    )
    print(f"  Transformed df1: {df1_transformed.shape}")

    # Prepare input
    if 'pos' in df1_transformed.columns:
        pos_col = df1_transformed['pos'].copy()
        df1_transformed = df1_transformed.drop(columns=['pos'])
    else:
        pos_col = None

    input_dim = int(cfg["input_dim"])
    if df1_transformed.shape[1] != input_dim:
        raise ValueError(f"Dimension mismatch! Expected {input_dim}, got {df1_transformed.shape[1]}")

    X = torch.tensor(df1_transformed.to_numpy(), dtype=torch.float32, device=device)
    print(f"  Input tensor shape: {X.shape}")
    print(f"  Input range: [{X.min().item():.4f}, {X.max().item():.4f}]")

    # Load model
    print("\n[MODEL] Loading TransformerAutoencoder...")
    model = TransformerAutoencoder(
        input_dim=input_dim,
        n_tokens=cfg["n_tokens"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"]
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Reconstruct in batches
    batch_size = cfg.get("batch_size", 64)
    all_recon = []

    print(f"\n[RECONSTRUCT] Processing in batches of {batch_size}...")
    for i in range(0, X.size(0), batch_size):
        x_batch = X[i:i + batch_size]
        recon_batch = model(x_batch)
        all_recon.append(recon_batch.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, X.size(0))}/{X.size(0)} samples")

    recon_array = np.vstack(all_recon)

    # Save
    print(f"\n[CHECK RECONSTRUCTED]")
    print(f"  Shape: {recon_array.shape}")
    print(f"  Range: [{recon_array.min():.4f}, {recon_array.max():.4f}]")
    print(f"  Mean: {recon_array.mean():.4f}")
    print(f"  Zeros: {(recon_array == 0).sum()} / {recon_array.size}")

    recon_df = pd.DataFrame(recon_array, columns=df1_transformed.columns.tolist())

    if pos_col is not None:
        recon_df.insert(0, 'pos', pos_col.values)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recon_df.to_feather(out_path)

    print(f"\n[DONE] Reconstructed data saved to: {out_path}")
    print(f"       Shape: {recon_df.shape}")
    print(f"       Used transformation: {trans1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transformer Autoencoder Reconstruct: denoise V1 data (single input)")
    ap.add_argument("--data_path1", required=True, help="V1 feather file (to denoise)")
    ap.add_argument("--transformed_out_dir", required=True, help="Directory to save transformed data")
    ap.add_argument("--saved_models_dir", required=True, help="saved_models directory")
    ap.add_argument("--criterion", default="outer_test_loss",
                    choices=["inner_val_loss", "outer_test_loss"],
                    help="Criterion for selecting best fold")
    ap.add_argument("--out_path", required=True, help="Output path for reconstructed data (.feather)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)