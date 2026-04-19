#!/usr/bin/env python3
import argparse, os, json, glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class DCA(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim, dropout_rate=0.0):
        super(DCA, self).__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn_enc1 = nn.BatchNorm1d(hidden_dim1)
        self.bn_enc1.weight.requires_grad = False

        self.enc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn_enc2 = nn.BatchNorm1d(hidden_dim2)
        self.bn_enc2.weight.requires_grad = False

        self.bottleneck = nn.Linear(hidden_dim2, latent_dim)
        self.bn_bottleneck = nn.BatchNorm1d(latent_dim)
        self.bn_bottleneck.weight.requires_grad = False

        self.dec1 = nn.Linear(latent_dim, hidden_dim2)
        self.bn_dec1 = nn.BatchNorm1d(hidden_dim2)
        self.bn_dec1.weight.requires_grad = False

        self.dec2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.bn_dec2 = nn.BatchNorm1d(hidden_dim1)
        self.bn_dec2.weight.requires_grad = False

        self.output = nn.Linear(hidden_dim1, input_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def encode(self, x):
        h = F.relu(self.bn_enc1(self.enc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn_enc2(self.enc2(h)))
        h = self.dropout(h)
        z = F.relu(self.bn_bottleneck(self.bottleneck(h)))
        return z

    def decode(self, z):
        h = F.relu(self.bn_dec1(self.dec1(z)))
        h = self.dropout(h)
        h = F.relu(self.bn_dec2(self.dec2(h)))
        h = self.dropout(h)
        return F.relu(self.output(h))

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

def _apply_trans(df, trans):
    if trans == "sqrt+1":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:] + 1)
    elif trans == "sqrt":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:])
    elif trans == "log2":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 1)
    elif trans == "log2_then_add_1":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 1) + 1
    elif trans == "sqrt+0.00001":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:] + 0.00001)
    elif trans == "sqrt+10":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:] + 10)
    elif trans == "sqrt+1_then_minus_1":
        df.iloc[:, 1:] = np.sqrt(df.iloc[:, 1:] + 1) - 1
    elif trans == "count+1":
        df.iloc[:, 1:] = df.iloc[:, 1:] + 1
    elif trans == "log2(count+2)":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 2)
    elif trans == "log2(count+1)+1":
        df.iloc[:, 1:] = np.log2(df.iloc[:, 1:] + 1) + 1
    elif trans == "no_trans":
        pass
    return df

def filter_and_transform(df1, df2, threshold_value, trans1, trans2):
    data_cols = df1.columns[1:]
    zero_percentage = (df1[data_cols] == 0).mean()
    keep_cols = zero_percentage < threshold_value

    pos_col = df1.columns[0]
    cols_to_keep = [pos_col] + data_cols[keep_cols].tolist()

    filtered_df1 = df1[cols_to_keep].copy()
    filtered_df2 = df2[cols_to_keep].copy()

    filtered_df1 = _apply_trans(filtered_df1, trans1)
    filtered_df2 = _apply_trans(filtered_df2, trans2)
    return filtered_df1, filtered_df2


def find_best_fold(saved_models_dir, criterion="inner_val_loss"):
    fold_dirs = glob.glob(os.path.join(saved_models_dir, "fold_*"))
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {saved_models_dir}")

    best_fold = None
    best_score = float("inf")

    for fold_dir in fold_dirs:
        config_path = os.path.join(fold_dir, "dca_config.json")
        if not os.path.exists(config_path):
            print(f"Warning: {config_path} not found, skipping")
            continue

        with open(config_path, "r") as f:
            cfg = json.load(f)

        score = cfg.get(criterion, float("inf"))
        fold_name = os.path.basename(fold_dir)
        print(f"{fold_name}: {criterion}={score:.6f}")

        if score != score:
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

    # 1) Find best model
    final_model_path = os.path.join(args.saved_models_dir, "final_model.pt")
    final_config_path = os.path.join(args.saved_models_dir, "final_config.json")

    if os.path.exists(final_model_path) and os.path.exists(final_config_path):
        print("[INFO] Using final_model (trained on all data)")
        model_path = final_model_path
        config_path = final_config_path
    else:
        print("[INFO] final_model not found, using best fold")
        fold_dir = find_best_fold(args.saved_models_dir, args.criterion)
        model_path = os.path.join(fold_dir, "dca_weights.pt")
        config_path = os.path.join(fold_dir, "dca_config.json")

    # 2) Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    dropout = cfg.get("dropout", 0.0)

    print(f"\n[CONFIG]")
    print(f"  threshold: {cfg['threshold']}")
    print(f"  trans1: {cfg['trans1']}")
    print(f"  trans2: {cfg['trans2']}")
    print(f"  architecture: {cfg['hidden_dim1']}-{cfg['hidden_dim2']}-{cfg['latent_dim']}")
    print(f"  dropout: {dropout}")

    # 3) Load and preprocess
    print("\n[LOAD DATA]")
    df1_raw = pd.read_feather(args.data_path1)
    df2_raw = pd.read_feather(args.data_path2)
    print(f"  Original df1: {df1_raw.shape}")
    print(f"  Original df2: {df2_raw.shape}")

    print("\n[PREPROCESS] Applying filter_and_transform...")
    df1_transformed, df2_transformed = filter_and_transform(
        df1_raw, df2_raw, cfg['threshold'], cfg['trans1'], cfg['trans2']
    )
    print(f"  Transformed df1: {df1_transformed.shape}")
    print(f"  Transformed df2: {df2_transformed.shape}")

    # 4) Save transformed files to --transformed_out_dir with _rep1/_rep2 naming
    out_dir = Path(args.transformed_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path1 = out_dir / "Count_matrix_transformed_rep1.feather"
    save_path2 = out_dir / "Count_matrix_transformed_rep2.feather"
    df1_transformed.to_feather(save_path1)
    df2_transformed.to_feather(save_path2)
    print(f"[SAVE] Transformed rep1 -> {save_path1}")
    print(f"[SAVE] Transformed rep2 -> {save_path2}")

    # 5) Prepare model input
    if 'pos' in df1_transformed.columns:
        pos_col = df1_transformed['pos'].copy()
        df1_transformed = df1_transformed.drop(columns=['pos'])
    else:
        pos_col = None

    X = torch.tensor(df1_transformed.to_numpy(), dtype=torch.float32, device=device)

    input_dim = int(cfg["input_dim"])
    if X.shape[1] != input_dim:
        raise ValueError(f"Dimension mismatch! Expected {input_dim}, got {X.shape[1]}")

    # 6) Load model
    print("\n[MODEL] Loading DCA...")
    model = DCA(input_dim, cfg["hidden_dim1"], cfg["hidden_dim2"], cfg["latent_dim"],
                dropout_rate=dropout).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 7) Reconstruct
    print("[RECONSTRUCT] Running DCA...")
    Z = model.encode(X)
    X_recon = model.decode(Z)

    # 8) Save reconstruction
    recon_df = pd.DataFrame(X_recon.cpu().numpy(), columns=df1_transformed.columns.tolist())
    if pos_col is not None:
        recon_df.insert(0, 'pos', pos_col.values)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recon_df.to_feather(out_path)

    print(f"\n[DONE] Reconstruction saved to: {out_path}")
    print(f"       Shape: {recon_df.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DCA Reconstruction / Inference")
    ap.add_argument("--data_path1", required=True, help="Input feather rep1 (to reconstruct)")
    ap.add_argument("--data_path2", required=True, help="Input feather rep2 (for column filtering)")
    ap.add_argument("--transformed_out_dir", required=True,
                    help="Directory to save Count_matrix_transformed_rep1/rep2.feather")
    ap.add_argument("--saved_models_dir", required=True, help="saved_models directory")
    ap.add_argument("--criterion", default="inner_val_loss",
                    choices=["inner_val_loss", "outer_test_loss"])
    ap.add_argument("--out_path", required=True, help="Output reconstruction feather path")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)